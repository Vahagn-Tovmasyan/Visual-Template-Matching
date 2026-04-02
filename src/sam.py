"""SAM 2 (Segment Anything Model 2) based template matching.

SAM 2 is a faster, more accurate successor to SAM for image segmentation.
Unlike feature matchers (SIFT, LightGlue, LoFTR) or object detectors
(YOLO, DINO), SAM 2 produces high-quality segmentation masks for *any* object.

Strategy for template matching with SAM 2:
1. Use HuggingFace's mask-generation pipeline to segment the scene into
   all candidate masks in a single optimized pass.
2. Extract features from the template (color histogram, Hu moments, AR).
3. For each scene segment, compare it against the template using:
   - Color histogram similarity (HSV space)
   - Shape similarity (Hu moments)
   - Size ratio compatibility
4. Return bounding boxes for the best matches.

This approach is especially powerful for:
- Objects with distinct shapes that feature matchers miss
- Occluded objects where only partial shape is visible
- Multi-instance detection (SAM 2 finds all distinct objects)

Requires: pip install torch torchvision transformers
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
_pipeline = None


def _get_pipeline():
    """Load SAM 2 mask-generation pipeline (cached singleton)."""
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    try:
        import torch
        from transformers import pipeline
    except ImportError:
        raise ImportError(
            "SAM 2 requires 'torch' and 'transformers'. "
            "Install with: pip install torch torchvision transformers"
        )

    device = 0 if torch.cuda.is_available() else -1

    # Try SAM 2 first, fall back to SAM 1 if not available
    model_id = "facebook/sam2.1-hiera-base-plus"
    try:
        logger.info(f"Loading SAM 2 pipeline: {model_id} (device={device})")
        _pipeline = pipeline(
            "mask-generation",
            model=model_id,
            device=device,
        )
    except Exception as e:
        logger.warning(f"SAM 2 load failed ({e}), falling back to SAM 1")
        model_id = "facebook/sam-vit-base"
        logger.info(f"Loading SAM 1 pipeline: {model_id} (device={device})")
        _pipeline = pipeline(
            "mask-generation",
            model=model_id,
            device=device,
        )

    return _pipeline


def _get_template_features(template_bgr: np.ndarray) -> dict:
    """Extract features from the template for comparison.

    Returns a dict with color histogram, Hu moments, aspect ratio, and area.
    """
    h, w = template_bgr.shape[:2]

    # Color histogram in HSV
    hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)

    # Also compute a dominant hue for quick filtering
    hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    dominant_hue = int(np.argmax(hue_hist))

    # Shape features: convert to grayscale, threshold, find contour
    gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hu_moments = np.zeros(7)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest)
        hu_moments = cv2.HuMoments(moments).flatten()

    aspect_ratio = w / max(h, 1)

    return {
        "color_hist": hist,
        "dominant_hue": dominant_hue,
        "hu_moments": hu_moments,
        "aspect_ratio": aspect_ratio,
        "area": w * h,
        "width": w,
        "height": h,
    }


def _compare_segment_to_template(
    scene_bgr: np.ndarray,
    mask: np.ndarray,
    template_features: dict,
) -> float:
    """Score how similar a scene segment is to the template.

    Returns a similarity score in [0, 1].
    """
    # Get bounding box of the mask
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0.0

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    w = x_max - x_min
    h = y_max - y_min

    if w < 5 or h < 5:
        return 0.0

    # Size ratio check: reject segments that are way too big or too small
    tmpl_area = template_features["area"]
    seg_area = w * h
    size_ratio = seg_area / max(tmpl_area, 1)
    if size_ratio > 25.0 or size_ratio < 0.01:
        return 0.0

    # 1. Color histogram similarity (most important signal)
    crop = scene_bgr[y_min:y_max, x_min:x_max]
    crop_mask = mask[y_min:y_max, x_min:x_max]

    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    seg_hist = cv2.calcHist([crop_hsv], [0, 1], crop_mask, [30, 32], [0, 180, 0, 256])
    cv2.normalize(seg_hist, seg_hist)

    color_dist = cv2.compareHist(
        template_features["color_hist"], seg_hist, cv2.HISTCMP_BHATTACHARYYA
    )
    color_sim = max(0.0, 1.0 - float(color_dist))

    # 2. Aspect ratio similarity
    seg_ar = w / max(h, 1)
    tmpl_ar = template_features["aspect_ratio"]
    ar_diff = abs(seg_ar - tmpl_ar) / max(tmpl_ar, 0.1)
    ar_sim = max(0.0, 1.0 - ar_diff)

    # 3. Shape similarity via Hu moments
    contours, _ = cv2.findContours(
        crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    hu_sim = 0.5  # default if we can't compute
    if contours:
        largest = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest)
        seg_hu = cv2.HuMoments(moments).flatten()

        # Log-scale comparison (standard approach for Hu moments)
        tmpl_hu = template_features["hu_moments"]
        eps = 1e-10
        log_tmpl = np.sign(tmpl_hu) * np.log10(np.abs(tmpl_hu) + eps)
        log_seg = np.sign(seg_hu) * np.log10(np.abs(seg_hu) + eps)
        diff = np.abs(log_tmpl - log_seg)
        hu_sim = max(0.0, 1.0 - np.mean(diff) / 10.0)

    # 4. Size similarity — penalize segments very different in size
    size_sim = 1.0 - min(1.0, abs(np.log(size_ratio + 1e-6)) / 3.0)

    # Combined score: color dominates, then shape, size, AR
    score = 0.50 * color_sim + 0.20 * hu_sim + 0.15 * size_sim + 0.15 * ar_sim

    return float(score)


def detect_sam(
    template_path: str | Path,
    scene_path: str | Path,
    confidence_threshold: float = 0.3,
    nms_iou_threshold: float = 0.3,
    points_per_batch: int = 64,
) -> DetectionResult:
    """Detect template in scene using SAM 2 segmentation + feature comparison.

    Uses HuggingFace's mask-generation pipeline which generates all masks
    in a single optimized pass (much faster than per-point inference).

    Args:
        template_path: Path to the template image.
        scene_path: Path to the scene image.
        confidence_threshold: Minimum confidence to keep a detection.
        nms_iou_threshold: IoU threshold for NMS.
        points_per_batch: Number of points processed in parallel by SAM.
            Higher = faster but more memory. Default 64 is good for CPU.

    Returns:
        DetectionResult with method="sam".
    """
    from PIL import Image

    validate_image_path(template_path)
    validate_image_path(scene_path)

    template_bgr = load_image_cv2(template_path)
    scene_bgr = load_image_cv2(scene_path)

    # Resize scene for speed if very large
    s_h, s_w = scene_bgr.shape[:2]
    scale = 1.0
    max_side = 1024
    if max(s_h, s_w) > max_side:
        scale = max_side / max(s_h, s_w)
        scene_resized = cv2.resize(
            scene_bgr, (int(s_w * scale), int(s_h * scale))
        )
    else:
        scene_resized = scene_bgr

    # Convert to PIL RGB for the pipeline
    scene_rgb = cv2.cvtColor(scene_resized, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(scene_rgb)

    # Extract template features
    template_features = _get_template_features(template_bgr)
    logger.info(
        f"Template: {template_features['width']}x{template_features['height']}, "
        f"AR={template_features['aspect_ratio']:.2f}, "
        f"dominant_hue={template_features['dominant_hue']}"
    )

    # Generate all masks using the pipeline — single optimized pass
    # Use relaxed thresholds to ensure we don't miss valid segments
    pipe = _get_pipeline()
    logger.info(f"Running SAM mask generation (points_per_batch={points_per_batch})...")
    outputs = pipe(
        pil_image,
        points_per_batch=points_per_batch,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.80,
    )

    masks_data = outputs.get("masks", [])
    scores_data = outputs.get("scores", [])
    logger.info(f"SAM generated {len(masks_data)} masks")

    if not masks_data:
        logger.info("SAM found no valid segments")
        return DetectionResult.empty(method="sam")

    # Score each mask against the template
    candidates = []
    for i, mask_np in enumerate(masks_data):
        # The pipeline returns masks as numpy arrays or PIL images
        if not isinstance(mask_np, np.ndarray):
            mask_np = np.array(mask_np)

        # Ensure uint8 binary mask
        if mask_np.dtype == bool:
            mask_np = mask_np.astype(np.uint8) * 255
        elif mask_np.max() <= 1:
            mask_np = (mask_np * 255).astype(np.uint8)
        else:
            mask_np = mask_np.astype(np.uint8)

        # Quick area filter
        mask_area = int((mask_np > 0).sum())
        total_area = mask_np.shape[0] * mask_np.shape[1]
        if mask_area < 100 or mask_area > int(total_area * 0.5):
            continue

        similarity = _compare_segment_to_template(
            scene_resized, mask_np, template_features
        )

        if similarity < 0.15:  # Very low bar — let final threshold do the work
            continue

        # Get bounding box from mask
        ys, xs = np.where(mask_np > 0)
        if len(xs) == 0:
            continue
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        # Scale back to original resolution
        bbox = [
            x_min / scale,
            y_min / scale,
            (x_max - x_min) / scale,
            (y_max - y_min) / scale,
        ]

        # Blend SAM IoU prediction with feature similarity
        # Give more weight to our similarity metric since SAM IoU
        # measures segmentation quality, not template match quality
        iou_pred = float(scores_data[i]) if i < len(scores_data) else 0.5
        combined_conf = 0.3 * iou_pred + 0.7 * similarity
        candidates.append((bbox, combined_conf))

    logger.info(f"SAM scored {len(candidates)} candidate segments")

    if not candidates:
        return DetectionResult.empty(method="sam")

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

    logger.info(f"SAM final detections: {len(detections)}")

    return DetectionResult(
        found=len(detections) > 0,
        detections=detections,
        method="sam",
    )
