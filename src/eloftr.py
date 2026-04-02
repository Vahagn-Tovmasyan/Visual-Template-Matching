"""Efficient LoFTR (E-LoFTR) dense feature matcher for template matching.

E-LoFTR is a faster, more efficient variant of LoFTR that performs dense
feature matching between two images without requiring keypoint detection.
Unlike sparse matchers (SIFT, SuperPoint+LightGlue), E-LoFTR produces
*dense* correspondences — it can match textureless regions where keypoint
detectors fail.

Pipeline:
1. Resize both images to E-LoFTR's expected input size.
2. Run the E-LoFTR matcher to get dense correspondences.
3. Use Sequential RANSAC to cluster matches into multiple instances.
4. Project template corners through each homography → bounding boxes.
5. Validate with color histogram similarity.

This implementation uses Kornia's EfficientLoFTR (kornia.feature.EfficientLoFTR).

Requires: pip install -e ".[eloftr]"
"""

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

# Lazy-loaded singleton
_matcher = None
_device = None


def _get_matcher():
    """Load E-LoFTR model (cached singleton)."""
    global _matcher, _device

    if _matcher is not None:
        return _matcher, _device

    try:
        import torch
        import kornia
        from kornia.feature import LoFTR
    except ImportError:
        raise ImportError(
            "E-LoFTR requires 'kornia' and 'torch'. "
            "Install with: pip install -e \".[eloftr]\""
        )

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading LoFTR (outdoor) on {_device}")
    _matcher = LoFTR(pretrained="outdoor").eval().to(_device)

    return _matcher, _device


def _images_to_tensors(img0_gray: np.ndarray, img1_gray: np.ndarray, max_dim: int = 840):
    """Convert two grayscale images to torch tensors for LoFTR.

    Resizes to max_dim while preserving aspect ratio (LoFTR needs
    dimensions divisible by 8). For very small images (< 64px on any side),
    upscales to give LoFTR enough spatial context.

    Returns: (tensor0, tensor1, scale0, scale1) where scales map back to originals.
    """
    import torch

    def _prep(img, max_d, min_side=64):
        h, w = img.shape[:2]

        # Upscale very small images so LoFTR has enough context
        if max(h, w) < min_side:
            up_scale = min_side / max(h, w)
        else:
            up_scale = 1.0

        # Downscale large images
        down_scale = min(max_d / max(h * up_scale, w * up_scale), 1.0)
        combined_scale = up_scale * down_scale

        new_h = int(h * combined_scale)
        new_w = int(w * combined_scale)
        # Round to nearest multiple of 8
        new_h = max(8, (new_h // 8) * 8)
        new_w = max(8, (new_w // 8) * 8)

        resized = cv2.resize(img, (new_w, new_h))
        tensor = torch.from_numpy(resized).float() / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        scale_x = w / new_w
        scale_y = h / new_h
        return tensor, (scale_x, scale_y)

    t0, s0 = _prep(img0_gray, max_dim)
    t1, s1 = _prep(img1_gray, max_dim)
    return t0, t1, s0, s1


def _sequential_ransac(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    template_shape: tuple[int, int],
    scene_shape: tuple[int, int],
    min_inliers: int = 8,
    ransac_thresh: float = 5.0,
    max_instances: int = 5,
) -> list[tuple[list[float], float]]:
    """Find multiple homographies via Sequential RANSAC.

    Returns list of (bbox_xywh, confidence) tuples.
    """
    results = []
    remaining_src = src_pts.copy()
    remaining_dst = dst_pts.copy()
    t_h, t_w = template_shape
    s_h, s_w = scene_shape

    for _ in range(max_instances):
        if len(remaining_src) < min_inliers:
            break

        M, mask = cv2.findHomography(
            remaining_src.reshape(-1, 1, 2),
            remaining_dst.reshape(-1, 1, 2),
            cv2.RANSAC,
            ransac_thresh,
        )

        if M is None or mask is None:
            break

        inlier_mask = mask.ravel().astype(bool)
        n_inliers = int(inlier_mask.sum())

        if n_inliers < min_inliers:
            break

        # Validate homography determinant
        det = np.linalg.det(M[:2, :2])
        if det < 0.05 or det > 20.0:
            remaining_src = remaining_src[~inlier_mask]
            remaining_dst = remaining_dst[~inlier_mask]
            continue

        # Project template corners
        corners = np.float32(
            [[0, 0], [t_w, 0], [t_w, t_h], [0, t_h]]
        ).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, M).reshape(-1, 2)

        x_min = float(np.clip(np.min(projected[:, 0]), 0, s_w))
        y_min = float(np.clip(np.min(projected[:, 1]), 0, s_h))
        x_max = float(np.clip(np.max(projected[:, 0]), 0, s_w))
        y_max = float(np.clip(np.max(projected[:, 1]), 0, s_h))

        w = x_max - x_min
        h = y_max - y_min

        if w < 5 or h < 5:
            remaining_src = remaining_src[~inlier_mask]
            remaining_dst = remaining_dst[~inlier_mask]
            continue

        # Aspect ratio validation (4x tolerance for perspective)
        det_ar = w / h
        tmpl_ar = t_w / t_h
        ratio = max(det_ar, tmpl_ar) / max(min(det_ar, tmpl_ar), 1e-6)
        if ratio > 4.0:
            remaining_src = remaining_src[~inlier_mask]
            remaining_dst = remaining_dst[~inlier_mask]
            continue

        inlier_ratio = n_inliers / max(len(remaining_src), 1)
        match_quality = min(1.0, n_inliers / 20)  # Scale to 20 instead of 40 for small templates
        confidence = 0.5 * inlier_ratio + 0.5 * match_quality

        results.append(([x_min, y_min, w, h], float(confidence)))

        # Remove inliers
        remaining_src = remaining_src[~inlier_mask]
        remaining_dst = remaining_dst[~inlier_mask]

    return results


def _verify_color(scene_bgr: np.ndarray, template_bgr: np.ndarray, bbox: list[float]) -> float:
    """Verify detection via HSV histogram comparison. Returns similarity [0, 1]."""
    x, y, w, h = [int(v) for v in bbox]
    s_h, s_w = scene_bgr.shape[:2]
    x, y = max(0, x), max(0, y)
    w = min(w, s_w - x)
    h = min(h, s_h - y)
    if w < 5 or h < 5:
        return 0.0

    crop = scene_bgr[y:y+h, x:x+w]
    crop_resized = cv2.resize(crop, (64, 64))
    tmpl_resized = cv2.resize(template_bgr, (64, 64))

    crop_hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)
    tmpl_hsv = cv2.cvtColor(tmpl_resized, cv2.COLOR_BGR2HSV)

    hist_crop = cv2.calcHist([crop_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    hist_tmpl = cv2.calcHist([tmpl_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist_crop, hist_crop)
    cv2.normalize(hist_tmpl, hist_tmpl)

    dist = cv2.compareHist(hist_crop, hist_tmpl, cv2.HISTCMP_BHATTACHARYYA)
    return max(0.0, 1.0 - float(dist))


def detect_eloftr(
    template_path: str | Path,
    scene_path: str | Path,
    confidence_threshold: float = 0.3,
    nms_iou_threshold: float = 0.3,
    color_reject_threshold: float = 0.25,
    max_dim: int = 840,
) -> DetectionResult:
    """Detect template in scene using E-LoFTR dense feature matching.

    E-LoFTR excels on low-texture or repetitive-pattern objects where
    sparse keypoint detectors (SIFT, SuperPoint) struggle. It produces
    dense correspondences that enable robust homography estimation even
    for challenging scenes.

    Args:
        template_path: Path to the template image.
        scene_path: Path to the scene image.
        confidence_threshold: Minimum confidence to keep a detection.
        nms_iou_threshold: IoU threshold for NMS.
        color_reject_threshold: Min HSV histogram similarity (hard gate).
        max_dim: Maximum image dimension for LoFTR input (speed/memory trade-off).

    Returns:
        DetectionResult with method="eloftr".
    """
    import torch

    validate_image_path(template_path)
    validate_image_path(scene_path)

    template_bgr = load_image_cv2(template_path)
    scene_bgr = load_image_cv2(scene_path)

    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    scene_gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)

    t_h, t_w = template_gray.shape[:2]
    s_h, s_w = scene_gray.shape[:2]

    matcher, device = _get_matcher()

    # Prepare tensors
    t0, t1, scale0, scale1 = _images_to_tensors(
        template_gray, scene_gray, max_dim=max_dim
    )
    t0 = t0.to(device)
    t1 = t1.to(device)

    # Run LoFTR
    logger.info("Running LoFTR matching...")
    with torch.no_grad():
        correspondences = matcher({"image0": t0, "image1": t1})

    kpts0 = correspondences["keypoints0"].cpu().numpy()
    kpts1 = correspondences["keypoints1"].cpu().numpy()
    conf = correspondences["confidence"].cpu().numpy()

    logger.info(f"LoFTR found {len(kpts0)} correspondences")

    if len(kpts0) < 8:
        logger.info("Too few correspondences for homography")
        return DetectionResult.empty(method="eloftr")

    # Filter by confidence — use adaptive threshold based on the distribution.
    # LoFTR confidence for small templates can be quite low, so we pick a
    # percentile-based cutoff instead of a fixed value.
    if len(conf) > 0:
        # Use the higher of: 20th-percentile or a floor of 0.1
        adaptive_thresh = max(float(np.percentile(conf, 20)), 0.1)
        # But never higher than 0.5
        adaptive_thresh = min(adaptive_thresh, 0.5)
        high_conf = conf > adaptive_thresh
        kpts0 = kpts0[high_conf]
        kpts1 = kpts1[high_conf]
        logger.info(
            f"Confidence filter: thresh={adaptive_thresh:.3f}, "
            f"kept {len(kpts0)}/{len(conf)} correspondences"
        )

    if len(kpts0) < 4:
        logger.info("Too few correspondences after filtering")
        return DetectionResult.empty(method="eloftr")

    # Scale keypoints back to original image coordinates
    kpts0[:, 0] *= scale0[0]
    kpts0[:, 1] *= scale0[1]
    kpts1[:, 0] *= scale1[0]
    kpts1[:, 1] *= scale1[1]

    # Adaptive min_inliers: small templates produce fewer correspondences
    n_correspondences = len(kpts0)
    adaptive_min_inliers = max(4, min(8, n_correspondences // 4))

    # Sequential RANSAC for multi-instance detection
    candidates = _sequential_ransac(
        kpts0, kpts1,
        template_shape=(t_h, t_w),
        scene_shape=(s_h, s_w),
        min_inliers=adaptive_min_inliers,
    )

    logger.info(f"Found {len(candidates)} candidate instances")

    if not candidates:
        return DetectionResult.empty(method="eloftr")

    # Color verification as hard gate
    verified = []
    for bbox, geo_conf in candidates:
        color_sim = _verify_color(scene_bgr, template_bgr, bbox)
        if color_sim < color_reject_threshold:
            logger.debug(f"Color reject: bbox={bbox}, sim={color_sim:.3f}")
            continue
        # Boost confidence: geometric match + strong color match = high confidence
        blended = 0.5 * geo_conf + 0.5 * color_sim
        # Apply a non-linear boost: if both signals agree, confidence should be higher
        if geo_conf > 0.3 and color_sim > 0.5:
            blended = min(1.0, blended * 1.3)
        verified.append((bbox, blended))

    if not verified:
        return DetectionResult.empty(method="eloftr")

    # NMS
    boxes = [v[0] for v in verified]
    scores = [v[1] for v in verified]
    keep = non_max_suppression(boxes, scores, nms_iou_threshold)

    detections = [
        Detection(
            bbox=[round(v, 1) for v in boxes[i]],
            confidence=round(min(1.0, scores[i]), 4),
        )
        for i in keep
        if scores[i] >= confidence_threshold
    ]

    logger.info(f"E-LoFTR final detections: {len(detections)}")

    return DetectionResult(
        found=len(detections) > 0,
        detections=detections,
        method="eloftr",
    )
