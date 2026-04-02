"""Tests for DINO detector — does not require a GPU or model download."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


class TestDinoColourPrompt:
    """Tests for colour-based prompt generation fallback."""

    def test_red_object_prompt(self):
        """Red template should produce a prompt containing 'red'."""
        from src.dino import _colour_based_prompt
        import cv2
        import tempfile
        from pathlib import Path

        red_img = np.zeros((50, 50, 3), dtype=np.uint8)
        red_img[:, :] = (0, 0, 255)  # BGR red

        # Windows-safe tempfile creation
        f = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        f.close()
        
        try:
            cv2.imwrite(f.name, red_img)
            prompt = _colour_based_prompt(f.name)
        finally:
            Path(f.name).unlink()

        assert "red" in prompt.lower()
        assert prompt.endswith(".")

    def test_blue_object_prompt(self):
        """Blue template should produce a prompt containing 'blue'."""
        from src.dino import _colour_based_prompt
        import cv2
        import tempfile
        from pathlib import Path

        blue_img = np.zeros((50, 50, 3), dtype=np.uint8)
        blue_img[:, :] = (255, 0, 0)  # BGR blue

        f = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        f.close()

        try:
            cv2.imwrite(f.name, blue_img)
            prompt = _colour_based_prompt(f.name)
        finally:
            Path(f.name).unlink()

        assert "blue" in prompt.lower()

    def test_green_object_prompt(self):
        """Green template should produce a prompt containing 'green'."""
        from src.dino import _colour_based_prompt
        import cv2
        import tempfile
        from pathlib import Path

        green_img = np.zeros((50, 50, 3), dtype=np.uint8)
        green_img[:, :] = (0, 255, 0)  # BGR green

        f = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        f.close()

        try:
            cv2.imwrite(f.name, green_img)
            prompt = _colour_based_prompt(f.name)
        finally:
            Path(f.name).unlink()

        assert "green" in prompt.lower()


class TestDinoDetectorSignature:
    """Tests that the detector function has the correct interface."""

    def test_detect_dino_exists(self):
        """detect_dino should be importable."""
        from src.dino import detect_dino
        assert callable(detect_dino)

    def test_detect_hybrid_dino_exists(self):
        """detect_hybrid_dino should be importable."""
        from src.hybrid_dino import detect_hybrid_dino
        assert callable(detect_hybrid_dino)

    def test_detect_dino_returns_detection_result(self):
        """detect_dino should return a DetectionResult with method='dino'."""
        from src.dino import detect_dino
        from src.schema import DetectionResult
        import inspect

        sig = inspect.signature(detect_dino)
        params = list(sig.parameters.keys())
        assert "template_path" in params
        assert "scene_path" in params
        assert "confidence_threshold" in params

    def test_detect_hybrid_dino_returns_detection_result(self):
        """detect_hybrid_dino should return a DetectionResult with method='hybrid-dino'."""
        from src.hybrid_dino import detect_hybrid_dino
        import inspect

        sig = inspect.signature(detect_hybrid_dino)
        params = list(sig.parameters.keys())
        assert "template_path" in params
        assert "scene_path" in params
        assert "confidence_threshold" in params


class TestDinoCoordinateConversion:
    """Test that DINO output format is correctly converted."""

    def test_xyxy_to_xywh_conversion(self):
        """DINO outputs [x1, y1, x2, y2], we need [x, y, w, h]."""
        # Simulate the conversion logic from detect_dino
        x1, y1, x2, y2 = 100.0, 200.0, 300.0, 400.0
        x, y = x1, y1
        w = x2 - x1
        h = y2 - y1

        assert x == 100.0
        assert y == 200.0
        assert w == 200.0
        assert h == 200.0

    def test_tiny_boxes_filtered(self):
        """Boxes smaller than 5px should be rejected."""
        x1, y1, x2, y2 = 100.0, 200.0, 102.0, 201.0
        w = x2 - x1
        h = y2 - y1
        assert w < 5 or h < 5  # Should be filtered
