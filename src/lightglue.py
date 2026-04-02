"""LightGlue sparse feature matcher for template matching.

LightGlue is a lightweight, accurate feature matcher that uses an
attention-based mechanism for adaptive matching. This implementation
pairs it with DISK feature extractor via kornia's LocalFeatureMatcher.

Pipeline:
1. Extract DISK keypoints + descriptors from both template and scene.
2. Match them with LightGlue (attention-based, adaptive stopping).
3. Use RANSAC on matched points to estimate a homography.
4. Project template corners through the homography -> bounding box.
5. Validate with color histogram similarity.
6. For multi-instance: use Sequential RANSAC to find multiple homographies.

Requires: pip install kornia torch torchvision
  (Uses kornia's built-in DISK + LightGlueMatcher — no separate package needed.)
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

# Lazy-loaded singletons
_matcher = None
_device = None


def _get_matcher():
    """Load kornia LocalFeatureMatcher with DISK + LightGlue (cached singleton)."""
    global _matcher, _device

    if _matcher is not None:
        return _matcher, _device

    try:
        import torch
        import kornia.feature as KF
    except ImportError:
        raise ImportError(
            "LightGlue requires 'kornia' and 'torch'. "
            "Install with: pip install kornia torch torchvision"
        )

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading kornia LocalFeatureMatcher (DISK + LightGlue) on {_device}")
    _matcher = KF.LocalFeatureMatcher(
        local_feature=KF.DISK.from_pretrained("depth"),
        matcher=KF.LightGlueMatcher("disk"),
    ).eval().to(_device)

    return _matcher, _device


def _prepare_tensor(bgr_img: np.ndarray, device: str, max_dim: int = 1024):
    """Convert BGR numpy image to torch tensor [1,3,H,W] in [0,1].

    DISK expects 3-channel (RGB) input, not grayscale.
    Resizes if larger than max_dim to keep inference fast.
    For very small images, upscales to give the detector enough context.
    Returns (tensor, scale_x, scale_y).
    """
    import torch

    h, w = bgr_img.shape[:2]

    # Upscale very small images
    min_side = 64
    if max(h, w) < min_side:
        up = min_side / max(h, w)
    else:
        up = 1.0

    # Downscale large images
    down = min(max_dim / max(h * up, w * up), 1.0)
    combined = up * down

    new_h = max(8, int(h * combined))
    new_w = max(8, int(w * combined))

    resized = cv2.resize(bgr_img, (new_w, new_h))
    # Convert BGR -> RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # [H,W,3] -> [3,H,W] -> [1,3,H,W]
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    tensor = tensor.unsqueeze(0).to(device)  # [1,3,H,W]

    scale_x = w / new_w
    scale_y = h / new_h
    return tensor, scale_x, scale_y


def _find_homographies(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    template_shape: tuple[int, int],
    scene_shape: tuple[int, int],
    min_inliers: int = 6,
    ransac_thresh: float = 5.0,
) -> list[tuple[list[float], float]]:
    """Sequential RANSAC to find multiple homographies.

    Returns list of (bbox_xywh, confidence) tuples.
    """
    results = []
    remaining_src = src_pts.copy()
    remaining_dst = dst_pts.copy()
    t_h, t_w = template_shape
    s_h, s_w = scene_shape

    for _ in range(5):  # Max 5 instances
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

        # Validate the homography determinant
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

        bw = x_max - x_min
        bh = y_max - y_min

        if bw < 5 or bh < 5:
            remaining_src = remaining_src[~inlier_mask]
            remaining_dst = remaining_dst[~inlier_mask]
            continue

        # Aspect ratio validation (4x tolerance)
        det_ar = bw / bh
        tmpl_ar = t_w / t_h
        ratio = max(det_ar, tmpl_ar) / max(min(det_ar, tmpl_ar), 1e-6)
        if ratio > 4.0:
            remaining_src = remaining_src[~inlier_mask]
            remaining_dst = remaining_dst[~inlier_mask]
            continue

        # Confidence from inlier ratio + match count
        inlier_ratio = n_inliers / max(len(remaining_src), 1)
        match_quality = min(1.0, n_inliers / 30)
        confidence = 0.5 * inlier_ratio + 0.5 * match_quality

        bbox = [x_min, y_min, bw, bh]
        results.append((bbox, float(confidence)))

        # Remove inliers for next iteration (Sequential RANSAC)
        remaining_src = remaining_src[~inlier_mask]
        remaining_dst = remaining_dst[~inlier_mask]

    return results


def _verify_color(scene_bgr: np.ndarray, template_bgr: np.ndarray, bbox: list[float]) -> float:
    """Verify a detection by comparing HSV histograms. Returns similarity [0, 1]."""
    x, y, w, h = [int(v) for v in bbox]
    s_h, s_w = scene_bgr.shape[:2]
    x = max(0, x)
    y = max(0, y)
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

    distance = cv2.compareHist(hist_crop, hist_tmpl, cv2.HISTCMP_BHATTACHARYYA)
    return max(0.0, 1.0 - float(distance))


def detect_lightglue(
    template_path: str | Path,
    scene_path: str | Path,
    confidence_threshold: float = 0.3,
    nms_iou_threshold: float = 0.3,
    color_reject_threshold: float = 0.25,
) -> DetectionResult:
    """Detect template in scene using DISK + LightGlue feature matching.

    Uses kornia's built-in LocalFeatureMatcher with DISK extractor
    and LightGlue matcher. No separate package needed — just kornia.

    Args:
        template_path: Path to the template image.
        scene_path: Path to the scene image.
        confidence_threshold: Minimum confidence to keep a detection.
        nms_iou_threshold: IoU threshold for NMS.
        color_reject_threshold: Min HSV histogram similarity (hard gate).

    Returns:
        DetectionResult with method="lightglue".
    """
    import torch

    validate_image_path(template_path)
    validate_image_path(scene_path)

    template_bgr = load_image_cv2(template_path)
    scene_bgr = load_image_cv2(scene_path)

    t_h, t_w = template_bgr.shape[:2]
    s_h, s_w = scene_bgr.shape[:2]

    matcher, device = _get_matcher()

    # Prepare tensors — DISK needs 3-channel RGB input
    t0, scale0_x, scale0_y = _prepare_tensor(template_bgr, device)
    t1, scale1_x, scale1_y = _prepare_tensor(scene_bgr, device)

    # Run the matcher — kornia's LocalFeatureMatcher handles extraction + matching
    logger.info("Running DISK + LightGlue matching via kornia...")
    with torch.no_grad():
        result = matcher({"image0": t0, "image1": t1})

    # Extract matched keypoints — same output format as LoFTR
    kpts0 = result["keypoints0"].cpu().numpy()  # [N, 2]
    kpts1 = result["keypoints1"].cpu().numpy()  # [N, 2]

    # Optional confidence filtering
    if "confidence" in result:
        conf = result["confidence"].cpu().numpy()
        # Keep matches above adaptive threshold
        if len(conf) > 0:
            thresh = max(float(np.percentile(conf, 20)), 0.1)
            thresh = min(thresh, 0.5)
            good = conf > thresh
            kpts0 = kpts0[good]
            kpts1 = kpts1[good]

    n_matches = len(kpts0)
    logger.info(f"LightGlue matches: {n_matches}")

    if n_matches < 6:
        logger.info("Too few matches for homography estimation")
        return DetectionResult.empty(method="lightglue")

    # Scale keypoints back to original image coordinates
    kpts0[:, 0] *= scale0_x
    kpts0[:, 1] *= scale0_y
    kpts1[:, 0] *= scale1_x
    kpts1[:, 1] *= scale1_y

    # Adaptive min_inliers for small match counts
    adaptive_min = max(4, min(6, n_matches // 4))

    # Find homographies (Sequential RANSAC for multi-instance)
    homography_results = _find_homographies(
        kpts0, kpts1,
        template_shape=(t_h, t_w),
        scene_shape=(s_h, s_w),
        min_inliers=adaptive_min,
    )
    logger.info(f"Found {len(homography_results)} candidate instances")

    if not homography_results:
        return DetectionResult.empty(method="lightglue")

    # Apply color verification as hard gate
    candidates = []
    for bbox, confidence in homography_results:
        color_sim = _verify_color(scene_bgr, template_bgr, bbox)
        if color_sim < color_reject_threshold:
            logger.debug(f"Color reject: bbox={bbox}, color_sim={color_sim:.3f}")
            continue
        # Blend: 60% geometric + 40% color
        blended = 0.6 * confidence + 0.4 * color_sim
        candidates.append((bbox, blended))

    if not candidates:
        return DetectionResult.empty(method="lightglue")

    # NMS
    boxes = [c[0] for c in candidates]
    scores = [c[1] for c in candidates]
    keep = non_max_suppression(boxes, scores, nms_iou_threshold)

    detections = [
        Detection(
            bbox=[round(v, 1) for v in boxes[i]],
            confidence=round(min(1.0, scores[i]), 4),
        )
        for i in keep
        if scores[i] >= confidence_threshold
    ]

    logger.info(f"LightGlue final detections: {len(detections)}")

    return DetectionResult(
        found=len(detections) > 0,
        detections=detections,
        method="lightglue",
    )
