"""Tests for utility functions."""

import numpy as np
import pytest

from src.utils import (
    bbox_xywh_to_xyxy,
    bbox_xyxy_to_xywh,
    compute_iou,
    non_max_suppression,
    validate_image_path,
)


class TestBboxConversion:
    """Tests for bounding box format conversions."""

    def test_xywh_to_xyxy(self):
        assert bbox_xywh_to_xyxy([10, 20, 30, 40]) == [10, 20, 40, 60]

    def test_xyxy_to_xywh(self):
        assert bbox_xyxy_to_xywh([10, 20, 40, 60]) == [10, 20, 30, 40]

    def test_roundtrip(self):
        original = [15.5, 25.0, 100.0, 200.0]
        converted = bbox_xyxy_to_xywh(bbox_xywh_to_xyxy(original))
        for a, b in zip(original, converted):
            assert abs(a - b) < 1e-6

    def test_zero_size_box(self):
        assert bbox_xywh_to_xyxy([5, 10, 0, 0]) == [5, 10, 5, 10]


class TestIoU:
    """Tests for IoU computation."""

    def test_identical_boxes(self):
        box = [10, 20, 30, 40]
        assert abs(compute_iou(box, box) - 1.0) < 1e-6

    def test_no_overlap(self):
        box_a = [0, 0, 10, 10]
        box_b = [20, 20, 10, 10]
        assert compute_iou(box_a, box_b) == 0.0

    def test_partial_overlap(self):
        box_a = [0, 0, 20, 20]
        box_b = [10, 10, 20, 20]
        # Intersection: 10x10=100, Union: 400+400-100=700
        expected = 100 / 700
        assert abs(compute_iou(box_a, box_b) - expected) < 1e-6

    def test_contained_box(self):
        outer = [0, 0, 100, 100]
        inner = [25, 25, 50, 50]
        # Intersection: 50x50=2500, Union: 10000+2500-2500=10000
        expected = 2500 / 10000
        assert abs(compute_iou(outer, inner) - expected) < 1e-6

    def test_zero_area(self):
        box_a = [0, 0, 0, 0]
        box_b = [0, 0, 10, 10]
        assert compute_iou(box_a, box_b) == 0.0

    def test_touching_boxes_no_overlap(self):
        box_a = [0, 0, 10, 10]
        box_b = [10, 0, 10, 10]
        assert compute_iou(box_a, box_b) == 0.0


class TestNMS:
    """Tests for non-maximum suppression."""

    def test_empty_input(self):
        assert non_max_suppression([], []) == []

    def test_single_box(self):
        result = non_max_suppression([[0, 0, 10, 10]], [0.9])
        assert result == [0]

    def test_no_overlap_keeps_all(self):
        boxes = [[0, 0, 10, 10], [50, 50, 10, 10], [100, 100, 10, 10]]
        scores = [0.9, 0.8, 0.7]
        result = non_max_suppression(boxes, scores, iou_threshold=0.5)
        assert len(result) == 3

    def test_high_overlap_suppresses(self):
        boxes = [[0, 0, 100, 100], [5, 5, 100, 100]]
        scores = [0.9, 0.8]
        result = non_max_suppression(boxes, scores, iou_threshold=0.5)
        assert len(result) == 1
        assert result[0] == 0  # Highest score kept

    def test_ordering_by_score(self):
        boxes = [[0, 0, 10, 10], [50, 50, 10, 10]]
        scores = [0.5, 0.9]
        result = non_max_suppression(boxes, scores, iou_threshold=0.5)
        assert result[0] == 1  # Higher score first

    def test_contained_box_suppressed(self):
        """Small box fully inside a larger box should be suppressed via IoA."""
        boxes = [[0, 0, 100, 100], [30, 30, 20, 20]]
        scores = [0.9, 0.8]
        result = non_max_suppression(
            boxes, scores, iou_threshold=0.5, ioa_threshold=0.8
        )
        assert len(result) == 1


class TestValidation:
    """Tests for image path validation."""

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            validate_image_path("/nonexistent/image.jpg")

    def test_unsupported_format(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not an image")
        with pytest.raises(ValueError, match="Unsupported"):
            validate_image_path(txt_file)

    def test_valid_image(self, tmp_path):
        """A valid .jpg file should pass validation."""
        img_file = tmp_path / "test.jpg"
        # Write a minimal valid JPEG (just needs to exist for path validation)
        import cv2
        cv2.imwrite(str(img_file), np.zeros((10, 10, 3), dtype=np.uint8))
        path = validate_image_path(img_file)
        assert path.exists()
