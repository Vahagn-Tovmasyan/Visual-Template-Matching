"""YOLO11 detector using Ultralytics.

Uses YOLO11 as a high-speed pseudo-template matcher. Since YOLO is a closed-set
detector (COCO classes), we first run it on the template image to deduce the 
target's class ID (e.g., class 29 for frisbee), then we extract all instances
of that specific class from the scene.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .schema import Detection, DetectionResult
from .utils import (
    non_max_suppression,
    validate_image_path,
)

logger = logging.getLogger(__name__)

DEFAULT_YOLO_MODEL = "yolo11s.pt"

# Lazy-loaded dictionary of models to avoid reloading weights
_models = {}

def _get_model(model_id: str = DEFAULT_YOLO_MODEL):
    """Load the YOLO model (cached singleton per model_id)."""
    if model_id in _models:
        return _models[model_id]

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "YOLO requires 'ultralytics'. "
            "Install with: pip install -e \".[yolo]\""
        )

    logger.info(f"Loading YOLO model: {model_id}")
    # Initialize YOLO model. It automatically downloads the weights if missing.
    model = YOLO(model_id)
    _models[model_id] = model
    return model


def _infer_template_class(model, template_path: str | Path) -> int | None:
    """Run YOLO on the template image to figure out which COCO class it is."""
    # We use a very low conf threshold here because tight template crops 
    # can confuse YOLO. We just want its best guess!
    results = model(str(template_path), conf=0.01, verbose=False)
    
    if not results or not len(results[0].boxes):
        return None
        
    # Get the bounding box with the highest confidence in the template
    boxes = results[0].boxes
    best_idx = boxes.conf.argmax().item()
    best_class_id = int(boxes.cls[best_idx].item())
    
    class_name = model.names[best_class_id]
    confidence = boxes.conf[best_idx].item()
    logger.info(f"YOLO inferred template class: {class_name} (ID: {best_class_id}) with conf {confidence:.2f}")
    
    return best_class_id


def detect_yolo(
    template_path: str | Path,
    scene_path: str | Path,
    confidence_threshold: float = 0.3,
    nms_iou_threshold: float = 0.3, # YOLO actually does its own NMS, but we'll apply it just in case
    model_id: str = DEFAULT_YOLO_MODEL,
) -> DetectionResult:
    """Detect template in scene using YOLO11.

    1. Infer the COCO class of the template image.
    2. Search the scene image.
    3. Filter the scene detections to only match the template's class.

    Args:
        template_path: Path to the template image.
        scene_path: Path to the scene image.
        confidence_threshold: Minimum confidence to keep a detection.
        nms_iou_threshold: IoU threshold for non-max suppression.
        model_id: YOLO model to use (e.g., 'yolo11n.pt', 'yolo11s.pt').

    Returns:
        DetectionResult with method="yolo".
    """
    validate_image_path(template_path)
    validate_image_path(scene_path)

    model = _get_model(model_id)

    # Step 1: Infer the template's class
    target_class_id = _infer_template_class(model, template_path)
    
    if target_class_id is None:
        logger.warning(
            "YOLO could not detect any object in the template. "
            "The template might not be one of the COCO classes."
        )
        return DetectionResult(found=False, detections=[], method="yolo")

    # Step 2: Run inference on the scene
    # We pass conf=confidence_threshold directly to YOLO for speed
    results = model(
        str(scene_path), 
        conf=confidence_threshold,
        iou=nms_iou_threshold,
        verbose=False
    )
    
    if not results or not len(results[0].boxes):
        logger.info("YOLO found 0 candidates in scene.")
        return DetectionResult(found=False, detections=[], method="yolo")

    boxes = results[0].boxes
    
    # Step 3: Filter detections by the target class
    detections_raw = []
    
    # boxes.xywh returns center x, center y, width, height
    # We need top-left x, y, width, height for our schema
    xywh_tensors = boxes.xywh.cpu().numpy()
    conf_tensors = boxes.conf.cpu().numpy()
    cls_tensors = boxes.cls.cpu().numpy()
    
    for xywh, conf, cls_id in zip(xywh_tensors, conf_tensors, cls_tensors):
        if int(cls_id) != target_class_id:
            continue
            
        cx, cy, w, h = xywh
        x1 = cx - (w / 2)
        y1 = cy - (h / 2)
        
        detections_raw.append({
            "bbox": [float(x1), float(y1), float(w), float(h)],
            "confidence": float(conf),
        })

    logger.info(f"YOLO found {len(detections_raw)} candidates of class {model.names[target_class_id]}")

    # Build final detections
    detections = [
        Detection(
            bbox=[round(v, 1) for v in d["bbox"]],
            confidence=round(d["confidence"], 4),
        )
        for d in detections_raw
    ]

    return DetectionResult(
        found=len(detections) > 0,
        detections=detections,
        method="yolo",
    )
