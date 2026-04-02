"""Output schema for template matching detections."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Detection(BaseModel):
    """A single detected template instance."""

    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box as [x, y, width, height] in pixels",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score between 0 and 1",
    )


class DetectionResult(BaseModel):
    """Complete result from a template matching run."""

    found: bool = Field(
        ...,
        description="Whether at least one match was found",
    )
    detections: list[Detection] = Field(
        default_factory=list,
        description="List of detected template instances",
    )
    method: str = Field(
        default="unknown",
        description="Detection method used (classical, vlm, hybrid)",
    )

    @classmethod
    def empty(cls, method: str = "unknown") -> DetectionResult:
        """Create an empty (no match) result."""
        return cls(found=False, detections=[], method=method)
