"""Tests for the output schema models."""

import json

import pytest
from pydantic import ValidationError

from src.schema import Detection, DetectionResult


class TestDetection:
    """Tests for the Detection model."""

    def test_valid_detection(self):
        d = Detection(bbox=[100, 200, 50, 60], confidence=0.85)
        assert d.bbox == [100, 200, 50, 60]
        assert d.confidence == 0.85

    def test_bbox_must_have_four_elements(self):
        with pytest.raises(ValidationError):
            Detection(bbox=[100, 200, 50], confidence=0.5)

    def test_bbox_five_elements_rejected(self):
        with pytest.raises(ValidationError):
            Detection(bbox=[1, 2, 3, 4, 5], confidence=0.5)

    def test_confidence_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            Detection(bbox=[0, 0, 10, 10], confidence=-0.1)

    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValidationError):
            Detection(bbox=[0, 0, 10, 10], confidence=1.1)

    def test_confidence_boundary_zero(self):
        d = Detection(bbox=[0, 0, 10, 10], confidence=0.0)
        assert d.confidence == 0.0

    def test_confidence_boundary_one(self):
        d = Detection(bbox=[0, 0, 10, 10], confidence=1.0)
        assert d.confidence == 1.0

    def test_float_bbox_values(self):
        d = Detection(bbox=[10.5, 20.3, 100.7, 200.1], confidence=0.5)
        assert d.bbox[0] == pytest.approx(10.5)

    def test_serialization_roundtrip(self):
        d = Detection(bbox=[10.5, 20.3, 100.0, 200.0], confidence=0.92)
        data = json.loads(d.model_dump_json())
        d2 = Detection(**data)
        assert d == d2


class TestDetectionResult:
    """Tests for the DetectionResult model."""

    def test_empty_result(self):
        r = DetectionResult.empty(method="classical")
        assert r.found is False
        assert r.detections == []
        assert r.method == "classical"

    def test_empty_result_json(self):
        r = DetectionResult.empty()
        data = json.loads(r.model_dump_json())
        assert data["found"] is False
        assert data["detections"] == []

    def test_result_with_detections(self):
        detections = [
            Detection(bbox=[100, 200, 50, 60], confidence=0.85),
            Detection(bbox=[300, 400, 70, 80], confidence=0.72),
        ]
        r = DetectionResult(found=True, detections=detections, method="vlm")
        assert r.found is True
        assert len(r.detections) == 2
        assert r.method == "vlm"

    def test_json_schema_structure(self):
        """Verify the JSON output matches the required structure:
        { "found": bool, "detections": [{ "bbox": [x,y,w,h], "confidence": float }] }
        """
        r = DetectionResult(
            found=True,
            detections=[Detection(bbox=[10, 20, 30, 40], confidence=0.9)],
            method="hybrid",
        )
        data = json.loads(r.model_dump_json())

        assert "found" in data
        assert "detections" in data
        assert isinstance(data["found"], bool)
        assert isinstance(data["detections"], list)

        det = data["detections"][0]
        assert "bbox" in det
        assert "confidence" in det
        assert len(det["bbox"]) == 4
        assert isinstance(det["confidence"], float)

    def test_serialization_roundtrip(self):
        r = DetectionResult(
            found=True,
            detections=[Detection(bbox=[10, 20, 30, 40], confidence=0.9)],
            method="classical",
        )
        data = json.loads(r.model_dump_json())
        r2 = DetectionResult(**data)
        assert r == r2

    def test_multiple_methods(self):
        """Ensure each method string is accepted."""
        for method in ("classical", "vlm", "hybrid"):
            r = DetectionResult.empty(method=method)
            assert r.method == method
