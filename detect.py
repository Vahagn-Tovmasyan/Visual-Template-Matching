"""CLI entry point for visual template matching."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2

from src.schema import DetectionResult
from src.classical import detect_classical
from src.vlm import detect_vlm
from src.hybrid import detect_hybrid
from src.utils import draw_detections, load_image_cv2


def _build_detectors() -> dict:
    """Build detector registry, gracefully skipping unavailable backends."""
    registry = {
        "classical": detect_classical,
        "vlm": detect_vlm,
        "hybrid": detect_hybrid,
    }
    try:
        from src.dino import detect_dino
        registry["dino"] = detect_dino
    except ImportError:
        pass
    try:
        from src.hybrid_dino import detect_hybrid_dino
        registry["hybrid-dino"] = detect_hybrid_dino
    except ImportError:
        pass
    try:
        from src.yolo import detect_yolo
        registry["yolo"] = detect_yolo
    except ImportError:
        pass
    try:
        from src.hybrid_yolo import detect_hybrid_yolo
        registry["hybrid-yolo"] = detect_hybrid_yolo
    except ImportError:
        pass
    try:
        from src.lightglue import detect_lightglue
        registry["lightglue"] = detect_lightglue
    except ImportError:
        pass
    try:
        from src.eloftr import detect_eloftr
        registry["eloftr"] = detect_eloftr
    except ImportError:
        pass
    try:
        from src.sam import detect_sam
        registry["sam"] = detect_sam
    except ImportError:
        pass
    return registry


DETECTORS = _build_detectors()


def main():
    parser = argparse.ArgumentParser(
        description="Visual Template Matching — locate templates in scene images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect.py --template template.jpg --scene scene.jpg
  python detect.py --template template.jpg --scene scene.jpg --method vlm --threshold 0.3
  python detect.py --template template.jpg --scene scene.jpg --output annotated.jpg
        """,
    )
    parser.add_argument(
        "--template", "-t", required=True, help="Path to the template image"
    )
    parser.add_argument(
        "--scene", "-s", required=True, help="Path to the scene image"
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=list(DETECTORS.keys()),
        default="classical",
        help="Detection method (default: classical)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Confidence threshold (default: 0.6)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to save the annotated image (optional)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    detector = DETECTORS[args.method]
    result: DetectionResult = detector(
        args.template,
        args.scene,
        confidence_threshold=args.threshold,
    )

    print(json.dumps(result.model_dump(), indent=2))

    if args.output:
        scene_img = load_image_cv2(args.scene)
        det_dicts = [d.model_dump() for d in result.detections]
        annotated = draw_detections(scene_img, det_dicts)
        cv2.imwrite(args.output, annotated)
        print(f"Annotated image saved to {args.output}")


if __name__ == "__main__":
    main()
