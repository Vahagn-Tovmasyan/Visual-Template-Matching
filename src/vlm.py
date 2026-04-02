"""VLM-based detector using Qwen2.5-VL via OpenRouter API."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import cv2
import requests
from dotenv import load_dotenv
from PIL import Image

from .schema import Detection, DetectionResult
from .utils import (
    get_cache_path,
    image_hash,
    image_to_base64,
    load_from_cache,
    load_image_cv2,
    load_image_pil,
    non_max_suppression,
    save_to_cache,
    validate_image_path,
)

logger = logging.getLogger(__name__)

load_dotenv()

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen2.5-vl-72b-instruct"


# ---------------------------------------------------------------------------
# Template description
# ---------------------------------------------------------------------------


def _describe_template(template_b64: str, api_key: str, model: str) -> str:
    """Ask the VLM to produce a concise natural-language description of the template.

    This forces a two-step reasoning process: the model first understands
    *what* to look for before searching. This consistently outperforms
    single-shot prompting for grounding tasks.
    """
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{template_b64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Describe the core object in this image in MAXIMUM 3 WORDS (e.g., 'red plastic frisbee' or 'blue coffee mug'). "
                            "Focus strictly on the base colour and object category. "
                            "Completely ignore lighting, glare, shadows, background, size, orientation, and surface text/logos."
                        ),
                    },
                ],
            }
        ],
        "max_tokens": 120,
        "temperature": 0.1,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(
        OPENROUTER_API_URL, json=payload, headers=headers, timeout=30
    )
    resp.raise_for_status()
    data = resp.json()
    description = data["choices"][0]["message"]["content"].strip()
    logger.info(f"Template description: {description}")
    return description


# ---------------------------------------------------------------------------
# Grounding via VLM
# ---------------------------------------------------------------------------

GROUNDING_PROMPT = """You are a precise visual object detector.

TEMPLATE DESCRIPTION: {description}

YOUR TASK:
Look at the scene image carefully and find ALL occurrences of the object
described above (which also matches the template image I provided).

INSTRUCTIONS — follow these exactly:
1. Scan the entire scene systematically from left to right, top to bottom.
2. For each candidate region, compare its shape, colour, and features against
   the template. Only include it if you are confident it is the same type of
   object.
3. Return bounding boxes as HIGH-PRECISION NORMALISED coordinates in [0, 1] (use at least 4 decimal places):
     x1 = left_edge / image_width
     y1 = top_edge  / image_height
     x2 = right_edge / image_width
     y2 = bottom_edge / image_height
4. Also return a confidence score (0.0 to 1.0) for each detection. This must reflect your actual visual certainty (e.g., 0.99 for a definite match, 0.72 for an ambiguous or blurry one). Do NOT default to 0.95.
5. If the object is NOT present in the scene, return an empty list.

IMPORTANT:
- Coordinates MUST be normalised floats between 0.0 and 1.0.
- Do NOT return pixel coordinates.
- Think step by step before answering.
- Be precise — avoid including nearby background in the box.
- Similar objects (e.g., a plate vs a frisbee) should NOT be matched.

Respond ONLY with valid JSON (no other text, no markdown fences):
{{"detections": [{{"bbox": [x1, y1, x2, y2], "confidence": <belief_score>}}]}}

If no matches found:
{{"detections": []}}"""


def _call_vlm_grounding(
    template_b64: str,
    scene_b64: str,
    description: str,
    scene_width: int,
    scene_height: int,
    api_key: str,
    model: str,
) -> list[dict]:
    """Call VLM to get bounding box detections."""
    prompt = GROUNDING_PROMPT.format(description=description)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"TEMPLATE IMAGE (this is the object to find — {description}):",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{template_b64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            f"SCENE IMAGE (search for the template object in "
                            f"this {scene_width}x{scene_height} px image):"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{scene_b64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.1,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(
        OPENROUTER_API_URL, json=payload, headers=headers, timeout=60
    )
    resp.raise_for_status()
    data = resp.json()

    content = data["choices"][0]["message"]["content"].strip()
    logger.info(f"VLM raw response: {content[:500]}")

    # Log token usage for cost tracking
    usage = data.get("usage", {})
    if usage:
        logger.info(
            f"Tokens — prompt: {usage.get('prompt_tokens', '?')}, "
            f"completion: {usage.get('completion_tokens', '?')}, "
            f"total: {usage.get('total_tokens', '?')}"
        )

    return _parse_vlm_response(content, scene_width, scene_height)


def _parse_vlm_response(
    content: str, scene_width: int, scene_height: int
) -> list[dict]:
    """Parse the VLM response, extracting JSON detections.

    Handles three coordinate conventions that VLMs commonly produce:
    1. Normalised [0, 1] — our preferred format.
    2. Qwen 0-1000 scale — internal convention of Qwen-VL models.
    3. Absolute pixel values — fallback.
    """
    # Try to extract JSON from markdown code blocks or raw text
    json_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL
    )
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON object
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            logger.warning("No JSON found in VLM response")
            return []

    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse VLM JSON: {e}")
        return []

    raw_detections = result.get("detections", [])
    detections = []

    for det in raw_detections:
        bbox = det.get("bbox", [])
        confidence = det.get("confidence", 0.5)

        if len(bbox) != 4:
            logger.warning(f"Invalid bbox format: {bbox}")
            continue

        x1, y1, x2, y2 = [float(v) for v in bbox]

        # Determine coordinate system and convert to absolute pixels
        if all(0 <= v <= 1.05 for v in [x1, y1, x2, y2]):
            # Normalised [0, 1] — multiply by image dimensions
            x1 *= scene_width
            y1 *= scene_height
            x2 *= scene_width
            y2 *= scene_height
        elif all(0 <= v <= 1000 for v in [x1, y1, x2, y2]) and max(
            x1, y1, x2, y2
        ) > 1.05:
            # Qwen 0-1000 convention
            x1 = x1 / 1000 * scene_width
            y1 = y1 / 1000 * scene_height
            x2 = x2 / 1000 * scene_width
            y2 = y2 / 1000 * scene_height
        # else: assume absolute pixel values, leave as-is

        # Ensure proper ordering (x1 < x2, y1 < y2)
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Clamp to image bounds
        x1 = max(0, min(x1, scene_width))
        y1 = max(0, min(y1, scene_height))
        x2 = max(0, min(x2, scene_width))
        y2 = max(0, min(y2, scene_height))

        w = x2 - x1
        h = y2 - y1

        if w < 5 or h < 5:
            continue

        confidence = max(0.0, min(1.0, float(confidence)))
        detections.append({"bbox": [x1, y1, w, h], "confidence": confidence})

    return detections


# ---------------------------------------------------------------------------
# Main VLM detector
# ---------------------------------------------------------------------------


def detect_vlm(
    template_path: str | Path,
    scene_path: str | Path,
    confidence_threshold: float = 0.3,
    nms_iou_threshold: float = 0.3,
    model: str = DEFAULT_MODEL,
    use_cache: bool = True,
) -> DetectionResult:
    """Detect template in scene using a Vision Language Model (Qwen2.5-VL).

    Two-step approach:
    1. Ask the VLM to describe the template in natural language.
    2. Send both images + description and ask for grounding coordinates.

    This works better than single-shot because it forces the model to reason
    about what makes the template distinctive before searching for it.

    Requires OPENROUTER_API_KEY environment variable.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key or api_key == "your-api-key-here":
        raise ValueError(
            "OPENROUTER_API_KEY not set. "
            "Copy .env.example to .env and add your key."
        )

    template_pil = load_image_pil(template_path)
    scene_pil = load_image_pil(scene_path)
    scene_width, scene_height = scene_pil.size

    # Resize for API efficiency (keep under 1536px on longest side)
    max_dim = 1536
    scale_factor = 1.0
    if max(scene_width, scene_height) > max_dim:
        scale_factor = max_dim / max(scene_width, scene_height)
        new_w = int(scene_width * scale_factor)
        new_h = int(scene_height * scale_factor)
        scene_pil_resized = scene_pil.resize((new_w, new_h), Image.LANCZOS)
    else:
        scene_pil_resized = scene_pil
        new_w, new_h = scene_width, scene_height

    # Resize template proportionally (keep it recognisable)
    t_max_dim = 512
    tw, th = template_pil.size
    if max(tw, th) > t_max_dim:
        t_scale = t_max_dim / max(tw, th)
        template_pil = template_pil.resize(
            (int(tw * t_scale), int(th * t_scale)), Image.LANCZOS
        )

    template_b64 = image_to_base64(template_pil)
    scene_b64 = image_to_base64(scene_pil_resized)

    # Check cache
    t_hash = image_hash(template_pil)
    s_hash = image_hash(scene_pil_resized)
    cache_path = get_cache_path("vlm", t_hash, s_hash, model)

    detections_raw = None

    if use_cache:
        cached = load_from_cache(cache_path)
        if cached is not None:
            logger.info("Using cached VLM result")
            detections_raw = cached.get("detections", [])
            # Scale back if the scene was resized
            if scale_factor != 1.0:
                for det in detections_raw:
                    det["bbox"] = [v / scale_factor for v in det["bbox"]]

    if detections_raw is None:
        # Step 1: Describe the template
        description = _describe_template(template_b64, api_key, model)

        # Step 2: Ground the template in the scene
        detections_raw = _call_vlm_grounding(
            template_b64,
            scene_b64,
            description,
            new_w,
            new_h,
            api_key,
            model,
        )

        # Cache the result (before scaling back)
        if use_cache:
            save_to_cache(
                cache_path,
                {"detections": detections_raw, "description": description},
            )

        # Scale coordinates back to original image size
        if scale_factor != 1.0:
            for det in detections_raw:
                det["bbox"] = [v / scale_factor for v in det["bbox"]]

    # NMS
    if detections_raw:
        boxes = [d["bbox"] for d in detections_raw]
        scores = [d["confidence"] for d in detections_raw]
        keep = non_max_suppression(boxes, scores, nms_iou_threshold)
        detections_raw = [detections_raw[i] for i in keep]

    # Post-hoc confidence calibration: VLMs like Qwen tend to output a fixed
    # confidence (often 0.95) regardless of actual certainty.  We recalibrate
    # by verifying each detection with a local colour-histogram check and
    # template matching against the template.  The calibrated score is driven
    # primarily by visual verification, not the VLM's raw score.
    if detections_raw:
        import numpy as np
        scene_cv = load_image_cv2(scene_path)
        template_cv = load_image_cv2(template_path)
        for det in detections_raw:
            bx, by, bw, bh = det["bbox"]
            x, y, w, h = int(bx), int(by), int(bw), int(bh)
            sh, sw = scene_cv.shape[:2]
            x = max(0, x)
            y = max(0, y)
            w = min(w, sw - x)
            h = min(h, sh - y)
            if w > 5 and h > 5:
                crop = scene_cv[y:y+h, x:x+w]
                crop_r = cv2.resize(crop, (64, 64))
                tmpl_r = cv2.resize(template_cv, (64, 64))

                # 1. HSV color histogram similarity
                crop_hsv = cv2.cvtColor(crop_r, cv2.COLOR_BGR2HSV)
                tmpl_hsv = cv2.cvtColor(tmpl_r, cv2.COLOR_BGR2HSV)
                hc = cv2.calcHist([crop_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
                ht = cv2.calcHist([tmpl_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
                cv2.normalize(hc, hc)
                cv2.normalize(ht, ht)
                dist = cv2.compareHist(hc, ht, cv2.HISTCMP_BHATTACHARYYA)
                color_sim = max(0.0, 1.0 - float(dist))

                # 2. Template matching score (NCC) on the crop
                crop_gray = cv2.cvtColor(crop_r, cv2.COLOR_BGR2GRAY)
                tmpl_gray = cv2.cvtColor(tmpl_r, cv2.COLOR_BGR2GRAY)
                ncc = cv2.matchTemplate(crop_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
                tm_score = float(np.clip(ncc.max(), 0.0, 1.0))

                # Calibrated confidence: heavily weighted toward visual verification
                # VLM's raw score only contributes 20% (since it's often constant 0.95)
                visual_score = 0.6 * color_sim + 0.4 * tm_score
                det["confidence"] = 0.2 * det["confidence"] + 0.8 * visual_score
            else:
                det["confidence"] *= 0.3  # Penalise tiny/degenerate boxes

    # Filter by confidence
    detections = [
        Detection(
            bbox=[round(v, 1) for v in d["bbox"]],
            confidence=round(d["confidence"], 4),
        )
        for d in detections_raw
        if d["confidence"] >= confidence_threshold
    ]

    return DetectionResult(
        found=len(detections) > 0,
        detections=detections,
        method="vlm",
    )