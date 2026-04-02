"""Shared utility functions for template matching."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Image loading & validation
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def validate_image_path(path: str | Path) -> Path:
    """Validate and return a resolved image path."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{p.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return p


def load_image_cv2(path: str | Path) -> np.ndarray:
    """Load an image as a BGR numpy array (OpenCV convention)."""
    p = validate_image_path(path)
    img = cv2.imread(str(p))
    if img is None:
        raise IOError(f"Failed to read image: {p}")
    return img


def load_image_pil(path: str | Path) -> Image.Image:
    """Load an image as a PIL Image."""
    p = validate_image_path(path)
    return Image.open(str(p)).convert("RGB")


def image_to_base64(
    img: Image.Image | np.ndarray, fmt: str = "JPEG"
) -> str:
    """Encode an image to a base64 string."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    img.save(buf, format=fmt, quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def image_hash(img: Image.Image | np.ndarray) -> str:
    """Compute a fast hash of an image for caching."""
    if isinstance(img, np.ndarray):
        data = img.tobytes()
    else:
        buf = BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()
    return hashlib.md5(data).hexdigest()


# ---------------------------------------------------------------------------
# Bounding box utilities
# ---------------------------------------------------------------------------


def bbox_xyxy_to_xywh(bbox: Sequence[float]) -> list[float]:
    """Convert [x1, y1, x2, y2] → [x, y, w, h]."""
    x1, y1, x2, y2 = bbox
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def bbox_xywh_to_xyxy(bbox: Sequence[float]) -> list[float]:
    """Convert [x, y, w, h] → [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return [float(x), float(y), float(x + w), float(y + h)]


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute standard IoU between two boxes in [x, y, w, h] format."""
    a = bbox_xywh_to_xyxy(box_a)
    b = bbox_xywh_to_xyxy(box_b)

    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def _compute_ioa(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute Intersection-over-Minimum-Area to detect contained boxes."""
    a = bbox_xywh_to_xyxy(box_a)
    b = bbox_xywh_to_xyxy(box_b)

    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    min_area = min(area_a, area_b)

    if min_area <= 0:
        return 0.0
    return inter_area / min_area


def non_max_suppression(
    boxes: list[list[float]],
    scores: list[float],
    iou_threshold: float = 0.5,
    ioa_threshold: float = 0.8,
) -> list[int]:
    """Non-maximum suppression on [x, y, w, h] boxes.

    Uses IoU to prune generic overlaps and IoA to prune fully contained boxes.
    Returns indices of kept boxes, sorted by descending confidence.
    """
    if not boxes:
        return []

    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []

    while indices:
        current = indices.pop(0)
        keep.append(current)
        remaining: list[int] = []
        for idx in indices:
            # Drop the box if IoU is too high OR if it is mostly enclosed
            iou = compute_iou(boxes[current], boxes[idx])
            ioa = _compute_ioa(boxes[current], boxes[idx])
            
            if iou < iou_threshold and ioa < ioa_threshold:
                remaining.append(idx)
        indices = remaining

    return keep


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw_detections(
    scene: np.ndarray,
    detections: list[dict],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
) -> np.ndarray:
    """Draw bounding boxes and confidence scores on a scene image.

    Args:
        scene: BGR image (will be copied, not modified in-place).
        detections: List of dicts with 'bbox' [x,y,w,h] and 'confidence'.
        color: BGR color for boxes.
        thickness: Line thickness.

    Returns:
        Annotated copy of the scene.
    """
    annotated = scene.copy()

    for det in detections:
        bbox = det["bbox"]
        conf = det.get("confidence", 0.0)
        x, y, w, h = [int(v) for v in bbox]

        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)

        label = f"{conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        label_thickness = 2
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, label_thickness)

        # Draw label background
        cv2.rectangle(
            annotated,
            (x, y - th - 10),
            (x + tw + 6, y),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x + 3, y - 5),
            font,
            font_scale,
            (0, 0, 0),
            label_thickness,
        )

    return annotated


# ---------------------------------------------------------------------------
# Caching helper
# ---------------------------------------------------------------------------

CACHE_DIR = Path(os.environ.get("VTM_CACHE_DIR", ".cache"))


def get_cache_path(prefix: str, *keys: str) -> Path:
    """Get a cache file path based on a hash of the keys."""
    h = hashlib.md5("_".join(keys).encode()).hexdigest()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{prefix}_{h}.json"


def load_from_cache(path: Path) -> dict | None:
    """Load JSON from cache if it exists."""
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def save_to_cache(path: Path, data: dict) -> None:
    """Save JSON to cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
