"""Grounding DINO detector using HuggingFace transformers.

Uses IDEA-Research/grounding-dino-tiny for zero-shot object detection.
The model runs locally (no API key needed) and supports text-prompted
detection, making it ideal for multi-instance scenarios where the VLM
struggles with precise bounding boxes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .schema import Detection, DetectionResult
from .utils import (
    load_image_pil,
    non_max_suppression,
    validate_image_path,
)

logger = logging.getLogger(__name__)

DEFAULT_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"

# Lazy-loaded singleton — model + processor are loaded once on first call
_model = None
_processor = None


def _get_model_and_processor(model_id: str = DEFAULT_DINO_MODEL):
    """Load the Grounding DINO model and processor (cached singleton)."""
    global _model, _processor

    if _model is not None and _processor is not None:
        return _model, _processor

    try:
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    except ImportError:
        raise ImportError(
            "Grounding DINO requires 'torch' and 'transformers'. "
            "Install with: pip install -e \".[dino]\""
        )

    logger.info(f"Loading Grounding DINO model: {model_id}")
    _processor = AutoProcessor.from_pretrained(model_id)
    _model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    _model.eval()

    # Use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = _model.to(device)
    logger.info(f"Grounding DINO loaded on {device}")

    return _model, _processor


def _generate_text_prompt(template_path: str | Path) -> str:
    """Generate a text prompt for DINO from the template image.
    
    Tries to use the VLM to describe the template (MAX 3 WORDS),
    otherwise falls back to a generic prompt based on colour analysis.
    Grounding DINO works best with very simple class names.
    """
    # Try VLM description first
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if api_key and api_key != "your-api-key-here":
        try:
            from .vlm import _describe_template, DEFAULT_MODEL
            from .utils import image_to_base64

            template_pil = load_image_pil(template_path)

            # Resize template for API efficiency
            t_max_dim = 512
            tw, th = template_pil.size
            if max(tw, th) > t_max_dim:
                t_scale = t_max_dim / max(tw, th)
                template_pil = template_pil.resize(
                    (int(tw * t_scale), int(th * t_scale)), Image.LANCZOS
                )

            template_b64 = image_to_base64(template_pil)
            description = _describe_template(template_b64, api_key, DEFAULT_MODEL)
            # DINO works best with short prompts ending with a period
            prompt = description.rstrip(".") + "."
            logger.info(f"DINO text prompt (from VLM): {prompt}")
            return prompt
        except Exception as e:
            logger.warning(f"VLM description failed, using fallback: {e}")

    # Fallback: analyse dominant colour of template
    prompt = _colour_based_prompt(template_path)
    logger.info(f"DINO text prompt (colour analysis): {prompt}")
    return prompt


def _colour_based_prompt(template_path: str | Path) -> str:
    """Generate a simple text prompt based on the template's dominant colour."""
    import cv2
    from .utils import load_image_cv2

    img = load_image_cv2(template_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_hsv = hsv.mean(axis=(0, 1))

    h, s, v = mean_hsv
    # Simple colour classification from HSV hue
    if s < 50:
        colour = "grey" if v < 200 else "white"
    elif h < 10 or h >= 160:  # OpenCV Red wraps from 160-180
        colour = "red"
    elif h < 25:
        colour = "orange"
    elif h < 35:
        colour = "yellow"
    elif h < 85:
        colour = "green"
    elif h < 130:
        colour = "blue"
    elif h < 170:
        colour = "purple"
    else:
        colour = "red"

    return f"a {colour} round object."


def detect_dino(
    template_path: str | Path,
    scene_path: str | Path,
    confidence_threshold: float = 0.3,
    nms_iou_threshold: float = 0.3,
    model_id: str = DEFAULT_DINO_MODEL,
    text_prompt: str | None = None,
) -> DetectionResult:
    """Detect template in scene using Grounding DINO (zero-shot object detection).

    Two-step approach:
    1. Generate a text description of the template (via VLM or colour analysis).
    2. Run Grounding DINO with that text prompt on the scene image.

    Runs locally — no API key required for detection itself (VLM description
    is optional and enhances accuracy).

    Args:
        template_path: Path to the template image.
        scene_path: Path to the scene image.
        confidence_threshold: Minimum confidence to keep a detection.
        nms_iou_threshold: IoU threshold for non-max suppression.
        model_id: HuggingFace model ID for Grounding DINO.
        text_prompt: Optional explicit text prompt (skips auto-generation).

    Returns:
        DetectionResult with method="dino".
    """
    import torch

    validate_image_path(template_path)
    validate_image_path(scene_path)

    model, processor = _get_model_and_processor(model_id)
    device = next(model.parameters()).device

    # Step 1: Generate text prompt
    if text_prompt is None:
        text_prompt = _generate_text_prompt(template_path)

    # Step 2: Load and process scene image
    scene_pil = load_image_pil(scene_path)
    scene_width, scene_height = scene_pil.size

    inputs = processor(images=scene_pil, text=text_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Step 3: Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Step 4: Post-process
    # DINO defaults to text_threshold=0.25 and box_threshold=0.25.
    # Tiny, distant objects often fall below these text similarity scores.
    # We drop the internal thresholds to 0.05 to over-generate candidates,
    # and rely on our own NMS and confidence_threshold filtering to clean them up.
    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=0.05,
        text_threshold=0.05,
        target_sizes=[(scene_height, scene_width)],
    )

    result = results[0]
    boxes = result["boxes"].cpu().numpy()   # [x1, y1, x2, y2]
    scores = result["scores"].cpu().numpy()

    logger.info(f"DINO found {len(boxes)} raw candidates")

    # Convert [x1, y1, x2, y2] → [x, y, w, h]
    detections_raw = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        if w < 5 or h < 5:
            continue
        detections_raw.append({
            "bbox": [float(x1), float(y1), float(w), float(h)],
            "confidence": float(score),
        })

    # NMS
    if detections_raw:
        nms_boxes = [d["bbox"] for d in detections_raw]
        nms_scores = [d["confidence"] for d in detections_raw]
        keep = non_max_suppression(nms_boxes, nms_scores, nms_iou_threshold)
        detections_raw = [detections_raw[i] for i in keep]

    # Build final detections
    detections = [
        Detection(
            bbox=[round(v, 1) for v in d["bbox"]],
            confidence=round(d["confidence"], 4),
        )
        for d in detections_raw
        if d["confidence"] >= confidence_threshold
    ]

    logger.info(f"DINO final detections: {len(detections)}")

    return DetectionResult(
        found=len(detections) > 0,
        detections=detections,
        method="dino",
    )
