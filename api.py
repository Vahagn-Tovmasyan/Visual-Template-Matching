"""FastAPI endpoint for visual template matching."""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.classical import detect_classical
from src.schema import DetectionResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Visual Template Matching API",
    description="Locate small visual templates inside larger scene images",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def _build_detector_registry() -> dict:
    """Build detector registry, gracefully skipping unavailable backends."""
    from src.vlm import detect_vlm
    from src.hybrid import detect_hybrid

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


DETECTORS = _build_detector_registry()

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _validate_upload(file: UploadFile, name: str) -> None:
    """Validate an uploaded file."""
    if not file.filename:
        raise HTTPException(400, f"No filename provided for {name}")
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED:
        raise HTTPException(
            400,
            f"Unsupported format '{ext}' for {name}. "
            f"Supported: {', '.join(sorted(SUPPORTED))}",
        )


@app.post("/detect", response_model=DetectionResult)
async def detect(
    template: UploadFile = File(..., description="Template image to find"),
    scene: UploadFile = File(..., description="Scene image to search in"),
    method: str = Form("classical", description="Detection method"),
    threshold: float = Form(0.4, description="Confidence threshold"),
):
    """Run template matching on uploaded images."""
    # Validate inputs
    _validate_upload(template, "template")
    _validate_upload(scene, "scene")

    if method not in DETECTORS:
        raise HTTPException(
            400,
            f"Unknown method '{method}'. Choose from: {list(DETECTORS.keys())}",
        )

    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(400, "Threshold must be between 0.0 and 1.0")

    detector = DETECTORS[method]

    # Save uploads to temp files
    with tempfile.TemporaryDirectory() as tmp_dir:
        t_ext = Path(template.filename).suffix
        s_ext = Path(scene.filename).suffix
        template_path = Path(tmp_dir) / f"template{t_ext}"
        scene_path = Path(tmp_dir) / f"scene{s_ext}"

        template_data = await template.read()
        scene_data = await scene.read()

        template_path.write_bytes(template_data)
        scene_path.write_bytes(scene_data)

        try:
            result = detector(
                str(template_path),
                str(scene_path),
                confidence_threshold=threshold,
            )
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise HTTPException(500, f"Detection failed: {str(e)}")

    return result


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
