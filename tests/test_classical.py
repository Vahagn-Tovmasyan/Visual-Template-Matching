"""Tests for the classical CV detector."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.classical import detect_classical, _validate_aspect_ratio
from src.schema import DetectionResult
from src.utils import compute_iou


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_IMAGES_DIR = Path("test_images")


@pytest.fixture
def synthetic_pair(tmp_path):
    """Create a synthetic template + scene pair with known ground truth.

    Pastes a coloured rectangle onto a neutral scene at a known location.
    """
    # Template: 50x50 red square with some texture
    template = np.zeros((50, 50, 3), dtype=np.uint8)
    template[:, :] = (0, 0, 200)  # Red (BGR)
    cv2.circle(template, (25, 25), 10, (0, 0, 255), -1)

    # Scene: 400x600 grey background
    scene = np.full((400, 600, 3), 128, dtype=np.uint8)
    noise = np.random.RandomState(42).randint(0, 30, scene.shape, dtype=np.uint8)
    scene = cv2.add(scene, noise)

    # Paste template at known location (100, 150)
    gt_x, gt_y = 100, 150
    scene[gt_y : gt_y + 50, gt_x : gt_x + 50] = template

    template_path = tmp_path / "template.png"
    scene_path = tmp_path / "scene.png"
    cv2.imwrite(str(template_path), template)
    cv2.imwrite(str(scene_path), scene)

    return template_path, scene_path, [gt_x, gt_y, 50, 50]


@pytest.fixture
def negative_pair(tmp_path):
    """Create a template + unrelated scene (no match expected)."""
    # Template: blue square
    template = np.zeros((50, 50, 3), dtype=np.uint8)
    template[:, :] = (200, 0, 0)  # Blue

    # Scene: green pattern, no blue at all
    scene = np.zeros((300, 400, 3), dtype=np.uint8)
    scene[:, :] = (0, 180, 0)
    for i in range(0, 300, 20):
        cv2.line(scene, (0, i), (400, i), (0, 200, 0), 1)

    template_path = tmp_path / "template.png"
    scene_path = tmp_path / "scene.png"
    cv2.imwrite(str(template_path), template)
    cv2.imwrite(str(scene_path), scene)

    return template_path, scene_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClassicalDetector:
    """Tests for the classical CV detector."""

    def test_positive_detection_synthetic(self, synthetic_pair):
        """Detector finds a template pasted into a simple synthetic scene."""
        template_path, scene_path, gt_bbox = synthetic_pair
        result = detect_classical(
            template_path, scene_path, confidence_threshold=0.3
        )

        assert isinstance(result, DetectionResult)
        assert result.found is True
        assert len(result.detections) >= 1
        assert result.method == "classical"

        # Best detection should be near the ground truth
        best = max(result.detections, key=lambda d: d.confidence)
        dx = abs(best.bbox[0] - gt_bbox[0])
        dy = abs(best.bbox[1] - gt_bbox[1])
        assert dx < 20 and dy < 20, (
            f"Detection at {best.bbox[:2]} too far from GT {gt_bbox[:2]}"
        )

    def test_negative_detection(self, negative_pair):
        """No false positive on an unrelated scene with high threshold."""
        template_path, scene_path = negative_pair
        result = detect_classical(
            template_path, scene_path, confidence_threshold=0.6
        )
        assert isinstance(result, DetectionResult)
        # With a high threshold, should find zero or very few matches
        assert len(result.detections) <= 1

    def test_empty_result_format(self, negative_pair):
        """Empty result should have found=False and an empty detections list."""
        template_path, scene_path = negative_pair
        result = detect_classical(
            template_path, scene_path, confidence_threshold=0.95
        )
        assert result.found is False
        assert result.detections == []

    def test_output_json_format(self, synthetic_pair):
        """Output can be serialised to valid JSON matching the required schema."""
        template_path, scene_path, _ = synthetic_pair
        result = detect_classical(template_path, scene_path)
        data = json.loads(result.model_dump_json())

        assert "found" in data
        assert "detections" in data
        assert isinstance(data["found"], bool)
        assert isinstance(data["detections"], list)

    def test_confidence_within_range(self, synthetic_pair):
        """All detection confidences must be in [0, 1]."""
        template_path, scene_path, _ = synthetic_pair
        result = detect_classical(
            template_path, scene_path, confidence_threshold=0.1
        )
        for det in result.detections:
            assert 0.0 <= det.confidence <= 1.0

    def test_invalid_template_path(self):
        with pytest.raises(FileNotFoundError):
            detect_classical("nonexistent.jpg", "nonexistent2.jpg")

    @pytest.mark.skipif(
        not TEST_IMAGES_DIR.exists(),
        reason="Test images not available",
    )
    def test_real_image_scenario_a(self):
        """Scenario A: clean detection on the original frisbee image."""
        template = TEST_IMAGES_DIR / "000000017029_Template.jpg"
        scene = TEST_IMAGES_DIR / "000000017029.jpg"
        if not template.exists() or not scene.exists():
            pytest.skip("Frisbee test images not present")

        result = detect_classical(template, scene, confidence_threshold=0.3)
        assert isinstance(result, DetectionResult)
        assert result.found is True
        assert len(result.detections) >= 1
        assert result.detections[0].confidence > 0.3


class TestAspectRatioValidation:
    """Tests for the aspect ratio filter."""

    def test_matching_ratio(self):
        # Template 100x50 (AR=2.0), detection 200x100 (AR=2.0)
        assert _validate_aspect_ratio([0, 0, 200, 100], (50, 100)) is True

    def test_wildly_different_ratio(self):
        # Template 100x50 (AR=2.0), detection 10x200 (AR=0.05)
        assert _validate_aspect_ratio([0, 0, 10, 200], (50, 100)) is False

    def test_tolerance_boundary(self):
        # Template 100x100 (AR=1.0), detection 300x100 (AR=3.0) with tolerance=3
        assert _validate_aspect_ratio(
            [0, 0, 300, 100], (100, 100), tolerance=3.0
        ) is True
        assert _validate_aspect_ratio(
            [0, 0, 400, 100], (100, 100), tolerance=3.0
        ) is False

    def test_zero_dimension_rejected(self):
        assert _validate_aspect_ratio([0, 0, 0, 100], (50, 100)) is False
