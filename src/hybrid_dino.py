"""Hybrid pipeline: Grounding DINO pre-filter -> VLM re-ranker.

Strategy:
1. Use VLM to describe the template in natural language.
2. Run Grounding DINO on the scene with that description to find candidates.
3. For each candidate, crop the region and send to VLM for verification.
4. Keep only VLM-confirmed detections.

This combines DINO's excellent multi-instance detection and precise
bounding boxes with the VLM's semantic understanding for verification.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PIL import Image

from .schema import Detection, DetectionResult
from .utils import (
    image_to_base64,
    load_image_cv2,
    load_image_pil,
    non_max_suppression,
)
from .vlm import _describe_template, OPENROUTER_API_URL, DEFAULT_MODEL
from .hybrid import _verify_crop_with_vlm
from .dino import detect_dino, _get_model_and_processor, DEFAULT_DINO_MODEL

logger = logging.getLogger(__name__)


def detect_hybrid_dino(
    template_path: str | Path,
    scene_path: str | Path,
    confidence_threshold: float = 0.3,
    nms_iou_threshold: float = 0.3,
    dino_threshold: float = 0.2,
    model_id: str = DEFAULT_DINO_MODEL,
    vlm_model: str = DEFAULT_MODEL,
    crop_expand: float = 1.5,
) -> DetectionResult:
    """Detect template using DINO pre-filter + VLM re-ranker.

    1. Describe the template via VLM (natural language).
    2. Run Grounding DINO on the scene with that description (lower threshold).
    3. For each DINO candidate, crop and verify with VLM.
    4. Return only VLM-confirmed detections.

    Args:
        template_path: Path to the template image.
        scene_path: Path to the scene image.
        confidence_threshold: Final minimum confidence to keep a detection.
        nms_iou_threshold: IoU threshold for NMS.
        dino_threshold: DINO detection threshold (lower = more candidates).
        model_id: HuggingFace model ID for Grounding DINO.
        vlm_model: OpenRouter model for VLM verification.
        crop_expand: Factor to expand crop regions for context.

    Returns:
        DetectionResult with method="hybrid-dino".
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key or api_key == "your-api-key-here":
        raise ValueError(
            "OPENROUTER_API_KEY not set. "
            "The DINO-Hybrid method requires an API key for VLM verification. "
            "Copy .env.example to .env and add your key."
        )

    # Step 1: Get DINO candidates (lower threshold to over-generate)
    dino_result = detect_dino(
        template_path,
        scene_path,
        confidence_threshold=dino_threshold,
        nms_iou_threshold=nms_iou_threshold,
        model_id=model_id,
    )

    if not dino_result.detections:
        logger.info("Hybrid-DINO: no DINO candidates found")
        return DetectionResult(
            found=False,
            detections=[],
            method="hybrid-dino",
        )

    logger.info(
        f"Hybrid-DINO: {len(dino_result.detections)} DINO candidates "
        f"to verify with VLM"
    )

    # Step 2: Prepare images for VLM verification
    scene = load_image_cv2(scene_path)
    template_pil = load_image_pil(template_path)
    template_b64 = image_to_base64(template_pil)
    s_h, s_w = scene.shape[:2]

    # Get template description from VLM
    description = _describe_template(template_b64, api_key, vlm_model)

    # Step 3: Verify each DINO candidate with VLM
    verified: list[Detection] = []

    for det in dino_result.detections:
        x, y, w, h = det.bbox

        # Expand the crop region for context
        cx, cy = x + w / 2, y + h / 2
        ew, eh = w * crop_expand, h * crop_expand
        ex1 = max(0, int(cx - ew / 2))
        ey1 = max(0, int(cy - eh / 2))
        ex2 = min(s_w, int(cx + ew / 2))
        ey2 = min(s_h, int(cy + eh / 2))

        crop = scene[ey1:ey2, ex1:ex2]
        if crop.size == 0:
            continue

        crop_pil = Image.fromarray(crop[:, :, ::-1])  # BGR -> RGB
        crop_b64 = image_to_base64(crop_pil)

        is_match, vlm_confidence = _verify_crop_with_vlm(
            template_b64, crop_b64, description, api_key, vlm_model
        )

        if is_match:
            # Blend DINO and VLM confidence
            blended = 0.4 * det.confidence + 0.6 * vlm_confidence

            if blended >= confidence_threshold:
                verified.append(
                    Detection(
                        bbox=[round(v, 1) for v in det.bbox],
                        confidence=round(min(1.0, blended), 4),
                    )
                )

    # NMS on the final set
    if verified:
        boxes = [d.bbox for d in verified]
        scores = [d.confidence for d in verified]
        keep = non_max_suppression(boxes, scores, nms_iou_threshold)
        verified = [verified[i] for i in keep]

    logger.info(f"Hybrid-DINO: {len(verified)} verified detections")

    return DetectionResult(
        found=len(verified) > 0,
        detections=verified,
        method="hybrid-dino",
    )
