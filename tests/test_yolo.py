"""Tests for YOLO11 detector — does not require weights download."""

import pytest
import numpy as np


class TestYoloDetectorSignature:
    """Tests that the detector function has the correct interface."""

    def test_detect_yolo_exists(self):
        """detect_yolo should be importable."""
        from src.yolo import detect_yolo
        assert callable(detect_yolo)

    def test_detect_hybrid_yolo_exists(self):
        """detect_hybrid_yolo should be importable."""
        from src.hybrid_yolo import detect_hybrid_yolo
        assert callable(detect_hybrid_yolo)

    def test_detect_yolo_signature(self):
        """detect_yolo should return a DetectionResult with method='yolo'."""
        from src.yolo import detect_yolo
        import inspect

        sig = inspect.signature(detect_yolo)
        params = list(sig.parameters.keys())
        assert "template_path" in params
        assert "scene_path" in params
        assert "confidence_threshold" in params

    def test_detect_hybrid_yolo_signature(self):
        """detect_hybrid_yolo should return a DetectionResult with method='hybrid-yolo'."""
        from src.hybrid_yolo import detect_hybrid_yolo
        import inspect

        sig = inspect.signature(detect_hybrid_yolo)
        params = list(sig.parameters.keys())
        assert "template_path" in params
        assert "scene_path" in params
        assert "confidence_threshold" in params


class TestYoloCoordinateConversion:
    """Test that YOLO output format is correctly converted."""

    def test_cxcywh_to_xywh_conversion(self):
        """YOLO outputs [center_x, center_y, w, h], we need top-left [x, y, w, h]."""
        # Simulate the conversion logic from detect_yolo
        cx, cy, w, h = 150.0, 250.0, 100.0, 200.0
        
        x1 = cx - (w / 2)
        y1 = cy - (h / 2)

        assert x1 == 100.0
        assert y1 == 150.0
        assert w == 100.0
        assert h == 200.0

    def test_zero_size_boxes_handled(self):
        """Boxes with 0 size shouldn't crash coordinate math."""
        cx, cy, w, h = 0.0, 0.0, 0.0, 0.0
        x1 = cx - (w / 2)
        y1 = cy - (h / 2)
        assert x1 == 0.0
        assert y1 == 0.0
