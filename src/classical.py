"""Classical computer vision detector using multi-scale template matching + SIFT feature matching."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .schema import Detection, DetectionResult
from .utils import (
    load_image_cv2,
    non_max_suppression,
    validate_image_path,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multi-scale template matching
# ---------------------------------------------------------------------------


def _multi_scale_template_match(
    scene_gray: np.ndarray,
    template_gray: np.ndarray,
    scale_range: tuple[float, float] = (0.15, 2.5),
    num_scales: int = 30,
    match_threshold: float = 0.55,
    max_detections_per_scale: int = 3,
) -> list[tuple[list[float], float]]:
    """Run template matching across multiple scales.

    Uses cv2.minMaxLoc per scale, then finds additional peaks by masking.
    Returns list of (bbox_xywh, score) tuples.
    """
    t_h, t_w = template_gray.shape[:2]
    s_h, s_w = scene_gray.shape[:2]

    candidates: list[tuple[list[float], float]] = []
    scales = np.linspace(scale_range[0], scale_range[1], num_scales)

    for scale in scales:
        new_w = int(t_w * scale)
        new_h = int(t_h * scale)

        # Skip if scaled template is larger than scene or too small
        if new_w < 10 or new_h < 10 or new_w > s_w or new_h > s_h:
            continue

        scaled_template = cv2.resize(template_gray, (new_w, new_h))
        result = cv2.matchTemplate(
            scene_gray, scaled_template, cv2.TM_CCOEFF_NORMED
        )

        # Find multiple peaks by iterative masking
        result_copy = result.copy()
        for _ in range(max_detections_per_scale):
            _, max_val, _, max_loc = cv2.minMaxLoc(result_copy)
            if max_val < match_threshold:
                break

            pt_x, pt_y = max_loc
            bbox = [float(pt_x), float(pt_y), float(new_w), float(new_h)]
            candidates.append((bbox, float(max_val)))

            # Mask out the region around the found match
            mask_x1 = max(0, pt_x - new_w // 2)
            mask_y1 = max(0, pt_y - new_h // 2)
            mask_x2 = min(result_copy.shape[1], pt_x + new_w // 2)
            mask_y2 = min(result_copy.shape[0], pt_y + new_h // 2)
            result_copy[mask_y1:mask_y2, mask_x1:mask_x2] = 0

    return candidates


# ---------------------------------------------------------------------------
# SIFT feature matching
# ---------------------------------------------------------------------------


def _sift_feature_match(
    scene_gray: np.ndarray,
    template_gray: np.ndarray,
    min_matches: int = 8,
    ratio_threshold: float = 0.75,
) -> list[tuple[list[float], float]]:
    """Use SIFT keypoint matching + homography to locate the template.

    Returns list of (bbox_xywh, score) tuples.
    """
    sift = cv2.SIFT_create(nfeatures=2000)

    kp_t, des_t = sift.detectAndCompute(template_gray, None)
    kp_s, des_s = sift.detectAndCompute(scene_gray, None)

    if des_t is None or des_s is None or len(kp_t) < 4 or len(kp_s) < 4:
        logger.debug("Insufficient keypoints for SIFT matching")
        return []

    # FLANN matcher
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_t, des_s, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m_pair in matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    if len(good_matches) < min_matches:
        logger.debug(
            f"Only {len(good_matches)} good matches (need {min_matches})"
        )
        return []

    # Compute homography
    src_pts = np.float32(
        [kp_t[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp_s[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is None:
        logger.debug("Homography estimation failed")
        return []

    inlier_ratio = float(np.sum(mask)) / len(mask) if len(mask) > 0 else 0.0

    if inlier_ratio < 0.3:
        logger.debug(f"Low inlier ratio: {inlier_ratio:.2f}")
        return []

    # Project template corners to scene
    t_h, t_w = template_gray.shape[:2]
    corners = np.float32(
        [[0, 0], [t_w, 0], [t_w, t_h], [0, t_h]]
    ).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, M)
    projected = projected.reshape(-1, 2)

    # Get axis-aligned bounding box from projected corners
    x_min = float(np.min(projected[:, 0]))
    y_min = float(np.min(projected[:, 1]))
    x_max = float(np.max(projected[:, 0]))
    y_max = float(np.max(projected[:, 1]))

    # Validate the bounding box is within reasonable scene bounds
    s_h, s_w = scene_gray.shape[:2]
    if x_min < -50 or y_min < -50 or x_max > s_w + 50 or y_max > s_h + 50:
        logger.debug("Projected bbox is out of scene bounds")
        return []

    # Clamp to scene boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(s_w, x_max)
    y_max = min(s_h, y_max)

    w = x_max - x_min
    h = y_max - y_min

    # Check for degenerate boxes
    if w < 5 or h < 5:
        return []

    # Score based on inlier ratio and number of matches
    match_score = min(1.0, len(good_matches) / 50)
    score = 0.5 * inlier_ratio + 0.5 * match_score
    score = min(1.0, score)

    bbox = [x_min, y_min, w, h]
    return [(bbox, score)]


# ---------------------------------------------------------------------------
# Color histogram verification
# ---------------------------------------------------------------------------


def _verify_color_match(
    scene: np.ndarray,
    template: np.ndarray,
    bbox: list[float],
) -> float:
    """Verify a detection by comparing color histograms.

    Returns a similarity score in [0, 1].
    """
    x, y, w, h = [int(v) for v in bbox]
    s_h, s_w = scene.shape[:2]

    # Clamp
    x = max(0, x)
    y = max(0, y)
    w = min(w, s_w - x)
    h = min(h, s_h - y)

    if w < 5 or h < 5:
        return 0.0

    crop = scene[y : y + h, x : x + w]
    crop_resized = cv2.resize(crop, (64, 64))
    template_resized = cv2.resize(template, (64, 64))

    # Convert to HSV for better color comparison
    crop_hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)
    tmpl_hsv = cv2.cvtColor(template_resized, cv2.COLOR_BGR2HSV)

    # Calculate histograms (H and S channels)
    hist_crop = cv2.calcHist(
        [crop_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256]
    )
    hist_tmpl = cv2.calcHist(
        [tmpl_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256]
    )

    cv2.normalize(hist_crop, hist_crop)
    cv2.normalize(hist_tmpl, hist_tmpl)

    distance = cv2.compareHist(hist_crop, hist_tmpl, cv2.HISTCMP_BHATTACHARYYA)
    similarity = max(0.0, 1.0 - float(distance))

    return similarity


def _validate_aspect_ratio(
    bbox: list[float],
    template_shape: tuple[int, ...],
    tolerance: float = 3.0,
) -> bool:
    """Check that a detection's aspect ratio is roughly compatible with the template.

    Args:
        bbox: [x, y, w, h] of the candidate.
        template_shape: (height, width, ...) of the template.
        tolerance: Maximum allowed ratio between candidate and template aspect ratios.

    Returns:
        True if the aspect ratio is plausible, False otherwise.
    """
    _, _, w, h = bbox
    if w < 1 or h < 1:
        return False
    t_h, t_w = template_shape[:2]
    if t_w < 1 or t_h < 1:
        return False

    det_ar = w / h
    tmpl_ar = t_w / t_h

    ratio = max(det_ar, tmpl_ar) / max(min(det_ar, tmpl_ar), 1e-6)
    return ratio <= tolerance


# ---------------------------------------------------------------------------
# Main classical detector
# ---------------------------------------------------------------------------


def detect_classical(
    template_path: str | Path,
    scene_path: str | Path,
    confidence_threshold: float = 0.4,
    nms_iou_threshold: float = 0.3,
    use_sift: bool = True,
    use_template_match: bool = True,
    use_color_verify: bool = True,
    color_reject_threshold: float = 0.3,
) -> DetectionResult:
    """Detect template in scene using classical CV methods.

    Combines multi-scale template matching and SIFT feature matching,
    then applies NMS, aspect ratio filtering, and color histogram verification.

    The color verification acts as a hard gate: candidates whose colour
    histogram similarity falls below *color_reject_threshold* are discarded
    entirely. Survivors have their confidence adjusted by blending match
    score with colour score.
    """
    template = load_image_cv2(template_path)
    scene = load_image_cv2(scene_path)

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    all_candidates: list[tuple[list[float], float]] = []

    # Stage 1: Multi-scale template matching
    if use_template_match:
        tm_candidates = _multi_scale_template_match(
            scene_gray,
            template_gray,
            match_threshold=max(confidence_threshold, 0.5),
        )
        logger.info(f"Template matching found {len(tm_candidates)} candidates")
        all_candidates.extend(tm_candidates)

    # Stage 2: SIFT feature matching
    if use_sift:
        sift_candidates = _sift_feature_match(scene_gray, template_gray)
        logger.info(f"SIFT matching found {len(sift_candidates)} candidates")
        all_candidates.extend(sift_candidates)

    if not all_candidates:
        return DetectionResult.empty(method="classical")

    # Filter by aspect ratio before NMS
    filtered: list[tuple[list[float], float]] = []
    for bbox, score in all_candidates:
        if _validate_aspect_ratio(bbox, template_gray.shape):
            filtered.append((bbox, score))
        else:
            logger.debug(f"Aspect ratio reject: {bbox}")
    all_candidates = filtered

    if not all_candidates:
        return DetectionResult.empty(method="classical")

    # Extract boxes and scores for NMS
    boxes = [c[0] for c in all_candidates]
    scores = [c[1] for c in all_candidates]

    # NMS
    keep_indices = non_max_suppression(boxes, scores, nms_iou_threshold)

    # Build detections with color verification as a hard gate
    detections: list[Detection] = []
    for idx in keep_indices:
        bbox = boxes[idx]
        score = scores[idx]

        if use_color_verify:
            color_score = _verify_color_match(scene, template, bbox)

            # Hard reject: if colour is very different, skip this candidate
            if color_score < color_reject_threshold:
                logger.debug(
                    f"Color reject: bbox={bbox}, color_sim={color_score:.3f}"
                )
                continue

            # Blend: 60% structural match + 40% color similarity
            score = 0.6 * score + 0.4 * color_score

        if score >= confidence_threshold:
            detections.append(
                Detection(
                    bbox=[round(v, 1) for v in bbox],
                    confidence=round(min(1.0, score), 4),
                )
            )

    return DetectionResult(
        found=len(detections) > 0,
        detections=detections,
        method="classical",
    )
