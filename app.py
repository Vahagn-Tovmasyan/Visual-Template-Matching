"""Interactive Gradio UI for visual template matching."""

from __future__ import annotations

import json
import logging
import tempfile
import traceback
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from src.classical import detect_classical
from src.vlm import detect_vlm
from src.hybrid import detect_hybrid
from src.utils import draw_detections

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Core detectors (always available)
DETECTORS = {
    "Classical (OpenCV + SIFT)": detect_classical,
    "VLM (Qwen2.5-VL via OpenRouter)": detect_vlm,
    "Hybrid (Classical -> VLM re-ranker)": detect_hybrid,
}

# Optional detectors — register if dependencies are installed
try:
    from src.dino import detect_dino
    DETECTORS["DINO (Grounding DINO)"] = detect_dino
except ImportError:
    pass
try:
    from src.hybrid_dino import detect_hybrid_dino
    DETECTORS["Hybrid-DINO (DINO -> VLM verifier)"] = detect_hybrid_dino
except ImportError:
    pass
try:
    from src.yolo import detect_yolo
    DETECTORS["YOLO (Ultralytics yolo11s)"] = detect_yolo
except ImportError:
    pass
try:
    from src.hybrid_yolo import detect_hybrid_yolo
    DETECTORS["Hybrid-YOLO (YOLO -> VLM verifier)"] = detect_hybrid_yolo
except ImportError:
    pass
try:
    from src.lightglue import detect_lightglue
    DETECTORS["LightGlue (DISK + LightGlue)"] = detect_lightglue
except ImportError:
    pass
try:
    from src.eloftr import detect_eloftr
    DETECTORS["E-LoFTR (Dense Feature Matching)"] = detect_eloftr
except ImportError:
    pass
try:
    from src.sam import detect_sam
    DETECTORS["SAM 2 (Segment Anything)"] = detect_sam
except ImportError:
    pass

# Color palette for each detector
_PALETTE = [
    (0, 255, 0), (255, 165, 0), (0, 191, 255), (148, 103, 189),
    (220, 20, 60), (0, 128, 128), (255, 69, 0), (75, 0, 130),
    (255, 215, 0), (0, 255, 127),
]
COLORS = {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(DETECTORS)}


def _save_temp_image(img: np.ndarray, name: str, tmp_dir: Path) -> Path:
    """Save a numpy (RGB) array to a JPEG file and return its path."""
    path = tmp_dir / name
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


def run_detection(
    template_img: np.ndarray | None,
    scene_img: np.ndarray | None,
    method: str,
    threshold: float,
) -> tuple[np.ndarray | None, str, str]:
    """Run template matching and return (annotated_image, json_str, summary)."""

    if template_img is None:
        return None, '{"error": "Please upload a template image."}', ""
    if scene_img is None:
        return None, '{"error": "Please upload a scene image."}', ""

    try:
        tmp_dir = Path(tempfile.mkdtemp(prefix="vtm_"))
        template_path = _save_temp_image(template_img, "template.jpg", tmp_dir)
        scene_path = _save_temp_image(scene_img, "scene.jpg", tmp_dir)

        detector = DETECTORS[method]
        result = detector(
            str(template_path),
            str(scene_path),
            confidence_threshold=threshold,
        )

        # Draw results
        scene_bgr = cv2.cvtColor(scene_img, cv2.COLOR_RGB2BGR)
        det_dicts = [d.model_dump() for d in result.detections]
        color = COLORS.get(method, (0, 255, 0))
        annotated_bgr = draw_detections(
            scene_bgr, det_dicts, color=color, thickness=3
        )
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        json_output = result.model_dump_json(indent=2)

        # Build human-readable summary
        n = len(result.detections)
        if n == 0:
            summary = "No matches found."
        else:
            confs = [d.confidence for d in result.detections]
            summary = (
                f"Found {n} match{'es' if n > 1 else ''}. "
                f"Confidence: {', '.join(f'{c:.2f}' for c in confs)}"
            )

        # Clean up temp files
        template_path.unlink(missing_ok=True)
        scene_path.unlink(missing_ok=True)
        tmp_dir.rmdir()

        return annotated_rgb, json_output, summary

    except FileNotFoundError as e:
        return None, f'{{"error": "File not found: {e}"}}', ""
    except ValueError as e:
        return None, f'{{"error": "{e}"}}', str(e)
    except Exception as e:
        logger.error(f"Detection failed: {traceback.format_exc()}")
        return None, f'{{"error": "Detection failed: {e}"}}', f"Error: {e}"


def run_comparison(
    template_img: np.ndarray | None,
    scene_img: np.ndarray | None,
    threshold: float,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Run Classical vs VLM side-by-side and return both annotated images + summary."""

    if template_img is None or scene_img is None:
        msg = "Please upload both template and scene images."
        return None, None, msg

    try:
        tmp_dir = Path(tempfile.mkdtemp(prefix="vtm_cmp_"))
        template_path = _save_temp_image(template_img, "template.jpg", tmp_dir)
        scene_path = _save_temp_image(scene_img, "scene.jpg", tmp_dir)

        scene_bgr = cv2.cvtColor(scene_img, cv2.COLOR_RGB2BGR)
        results = {}

        for label, detector, color in [
            ("Classical", detect_classical, (0, 255, 0)),
            ("VLM", detect_vlm, (255, 165, 0)),
        ]:
            try:
                result = detector(
                    str(template_path),
                    str(scene_path),
                    confidence_threshold=threshold,
                )
                det_dicts = [d.model_dump() for d in result.detections]
                ann = draw_detections(
                    scene_bgr, det_dicts, color=color, thickness=3
                )
                ann_rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
                results[label] = (ann_rgb, result)
            except Exception as e:
                logger.error(f"{label} failed: {e}")
                results[label] = (scene_img, None)

        template_path.unlink(missing_ok=True)
        scene_path.unlink(missing_ok=True)
        tmp_dir.rmdir()

        # Build comparison summary
        lines = []
        for label in ("Classical", "VLM"):
            ann, res = results.get(label, (scene_img, None))
            if res is None:
                lines.append(f"{label}: error")
            else:
                n = len(res.detections)
                if n == 0:
                    lines.append(f"{label}: no matches")
                else:
                    confs = [f"{d.confidence:.2f}" for d in res.detections]
                    lines.append(
                        f"{label}: {n} match{'es' if n > 1 else ''} "
                        f"(conf: {', '.join(confs)})"
                    )

        classical_img = results.get("Classical", (scene_img, None))[0]
        vlm_img = results.get("VLM", (scene_img, None))[0]

        return classical_img, vlm_img, "\n".join(lines)

    except Exception as e:
        logger.error(f"Comparison failed: {traceback.format_exc()}")
        return None, None, f"Error: {e}"


# ---------------------------------------------------------------------------
# Build Gradio interface
# ---------------------------------------------------------------------------

DESCRIPTION = """
# Visual Template Matching

Locate small visual templates inside larger scene images using **Classical CV**,
**Vision Language Models**, or a **Hybrid** approach.

**How to use:**
1. Upload a **template** image (the small object to find).
2. Upload a **scene** image (the larger image to search in).
3. Choose a detection method and set the confidence threshold.
4. Click **Run Detection** to see results.
"""

with gr.Blocks(
    title="Visual Template Matching",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
) as app:
    gr.Markdown(DESCRIPTION)

    # ---- Tab: Single Detection ----
    with gr.Tab("Detection"):
        with gr.Row():
            with gr.Column(scale=1):
                template_input = gr.Image(
                    label="Template Image",
                    type="numpy",
                    height=250,
                )
            with gr.Column(scale=1):
                scene_input = gr.Image(
                    label="Scene Image",
                    type="numpy",
                    height=250,
                )

        with gr.Row():
            method_dropdown = gr.Dropdown(
                choices=list(DETECTORS.keys()),
                value="Classical (OpenCV + SIFT)",
                label="Detection Method",
                interactive=True,
            )
            threshold_slider = gr.Slider(
                minimum=0.05,
                maximum=1.0,
                value=0.4,
                step=0.05,
                label="Confidence Threshold",
                interactive=True,
            )

        run_btn = gr.Button("Run Detection", variant="primary", size="lg")

        summary_box = gr.Textbox(label="Summary", interactive=False)

        with gr.Row():
            with gr.Column(scale=2):
                result_image = gr.Image(
                    label="Detection Results",
                    type="numpy",
                    height=500,
                )
            with gr.Column(scale=1):
                result_json = gr.Textbox(
                    label="JSON Output",
                    lines=20,
                    max_lines=30,
                )

        # Examples
        example_dir = Path("test_images")
        if example_dir.exists():
            template = str(example_dir / "000000017029_Template.jpg")
            gr.Examples(
                examples=[
                    [template, str(example_dir / "000000017029.jpg"),
                     "Classical (OpenCV + SIFT)", 0.4],
                    [template, str(example_dir / "gettyimages-2196099952-2048x2048.jpg"),
                     "Classical (OpenCV + SIFT)", 0.35],
                    [template, str(example_dir / "istockphoto-183252423-2048x2048.jpg"),
                     "Classical (OpenCV + SIFT)", 0.3],
                    [template, str(example_dir / "Generated Image.jpg"),
                     "Classical (OpenCV + SIFT)", 0.3],
                ],
                inputs=[template_input, scene_input, method_dropdown, threshold_slider],
                label="Click an example to load it",
            )

        run_btn.click(
            fn=run_detection,
            inputs=[template_input, scene_input, method_dropdown, threshold_slider],
            outputs=[result_image, result_json, summary_box],
        )

    # ---- Tab: Side-by-Side Comparison ----
    with gr.Tab("Compare Methods"):
        gr.Markdown(
            "Run **Classical** and **VLM** on the same inputs and compare results side-by-side."
        )
        with gr.Row():
            cmp_template = gr.Image(label="Template Image", type="numpy", height=250)
            cmp_scene = gr.Image(label="Scene Image", type="numpy", height=250)

        with gr.Row():
            cmp_threshold = gr.Slider(
                minimum=0.05,
                maximum=1.0,
                value=0.4,
                step=0.05,
                label="Confidence Threshold",
            )
            cmp_run_btn = gr.Button("Run Comparison", variant="primary", size="lg")

        cmp_summary = gr.Textbox(label="Summary", interactive=False)

        with gr.Row():
            cmp_classical = gr.Image(
                label="Classical Results",
                type="numpy",
                height=400,
            )
            cmp_vlm = gr.Image(
                label="VLM Results",
                type="numpy",
                height=400,
            )

        cmp_run_btn.click(
            fn=run_comparison,
            inputs=[cmp_template, cmp_scene, cmp_threshold],
            outputs=[cmp_classical, cmp_vlm, cmp_summary],
        )

if __name__ == "__main__":
    app.launch(share=False)
