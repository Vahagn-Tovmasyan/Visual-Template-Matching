"""Tests for VLM response parsing — does not call the API."""

import pytest

from src.vlm import _parse_vlm_response


class TestVLMResponseParsing:
    """Tests for parsing various VLM response formats."""

    def test_normalised_coordinates(self):
        content = '{"detections": [{"bbox": [0.1, 0.2, 0.5, 0.6], "confidence": 0.9}]}'
        result = _parse_vlm_response(content, scene_width=1000, scene_height=800)
        assert len(result) == 1
        bbox = result[0]["bbox"]
        # [0.1*1000, 0.2*800, (0.5-0.1)*1000, (0.6-0.2)*800]
        assert bbox[0] == pytest.approx(100, abs=1)
        assert bbox[1] == pytest.approx(160, abs=1)
        assert bbox[2] == pytest.approx(400, abs=1)  # width
        assert bbox[3] == pytest.approx(320, abs=1)  # height

    def test_qwen_1000_scale(self):
        content = '{"detections": [{"bbox": [100, 200, 500, 600], "confidence": 0.8}]}'
        result = _parse_vlm_response(content, scene_width=1920, scene_height=1080)
        assert len(result) == 1
        bbox = result[0]["bbox"]
        # Qwen 0-1000 scale: x1=100/1000*1920=192
        assert bbox[0] == pytest.approx(192, abs=1)

    def test_pixel_coordinates(self):
        # Values > 1000, treated as absolute pixels
        content = '{"detections": [{"bbox": [100, 200, 1500, 900], "confidence": 0.7}]}'
        result = _parse_vlm_response(content, scene_width=1920, scene_height=1080)
        assert len(result) == 1
        assert result[0]["bbox"][0] == pytest.approx(100, abs=1)

    def test_empty_detections(self):
        content = '{"detections": []}'
        result = _parse_vlm_response(content, 800, 600)
        assert result == []

    def test_markdown_wrapped_json(self):
        content = '```json\n{"detections": [{"bbox": [0.1, 0.2, 0.5, 0.6], "confidence": 0.9}]}\n```'
        result = _parse_vlm_response(content, 1000, 1000)
        assert len(result) == 1

    def test_no_json_in_response(self):
        content = "I could not find the object in the scene."
        result = _parse_vlm_response(content, 800, 600)
        assert result == []

    def test_invalid_json(self):
        content = '{"detections": [{"bbox": [broken}'
        result = _parse_vlm_response(content, 800, 600)
        assert result == []

    def test_invalid_bbox_length(self):
        content = '{"detections": [{"bbox": [0.1, 0.2], "confidence": 0.9}]}'
        result = _parse_vlm_response(content, 800, 600)
        assert result == []

    def test_tiny_box_filtered(self):
        """Boxes smaller than 5px should be filtered out."""
        content = '{"detections": [{"bbox": [0.5, 0.5, 0.501, 0.501], "confidence": 0.9}]}'
        result = _parse_vlm_response(content, 1000, 1000)
        assert result == []

    def test_confidence_clamped(self):
        content = '{"detections": [{"bbox": [0.1, 0.1, 0.5, 0.5], "confidence": 1.5}]}'
        result = _parse_vlm_response(content, 1000, 1000)
        assert len(result) == 1
        assert result[0]["confidence"] == 1.0

    def test_swapped_coordinates_corrected(self):
        """If x1 > x2 or y1 > y2, they should be swapped automatically."""
        content = '{"detections": [{"bbox": [0.5, 0.6, 0.1, 0.2], "confidence": 0.8}]}'
        result = _parse_vlm_response(content, 1000, 1000)
        assert len(result) == 1
        # After swap and conversion: x=100, y=200, w=400, h=400
        assert result[0]["bbox"][2] > 0  # width positive
        assert result[0]["bbox"][3] > 0  # height positive

    def test_multiple_detections(self):
        content = (
            '{"detections": ['
            '{"bbox": [0.1, 0.1, 0.3, 0.3], "confidence": 0.9},'
            '{"bbox": [0.6, 0.6, 0.8, 0.8], "confidence": 0.7}'
            "]}"
        )
        result = _parse_vlm_response(content, 1000, 1000)
        assert len(result) == 2
