"""Hybrid pipeline: classical CV pre-filter -> VLM re-ranker.

Strategy:
1. Run classical detection with a *lower* threshold to over-generate candidates.
2. For each candidate, crop the scene region (expanded by a factor) and send
   the crop + template to the VLM for verification.
3. Keep only VLM-confirmed detections.

If classical detection finds nothing (common with heavily occluded or
transformed targets), fall back to full VLM grounding on the entire scene.

This typically reduces API calls from 1 full-scene analysis to N small-crop
verifications, where N is usually 1-5.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image

from .schema import Detection, DetectionResult
from .utils import (
    image_to_base64,
    load_image_cv2,
    load_image_pil,
    non_max_suppression,
)
from .classical import detect_classical
from .vlm import _describe_template, OPENROUTER_API_URL, DEFAULT_MODEL

logger = logging.getLogger(__name__)
load_dotenv()


RERANK_PROMPT = """You are a precise visual verification system.

I will show you:
1. A TEMPLATE image — the exact object you need to identify.
2. A CANDIDATE CROP from a larger scene — a region that might contain the template.

TEMPLATE DESCRIPTION: {description}

YOUR TASK:
Determine whether the candidate crop contains the same object as the template.

Focus on: overall shape, base colour, and general category of the object.
Ignore: background, lighting differences, translucency, glare, and surface details like stamped text or logos.

Consider:
- The object may appear at a different scale or viewing angle.
- It may be partially occluded (hidden behind something).
- Surface textures (like moulded text or logos) may be visible in one image but invisible in the other due to lighting or distance. This is normal.
- Similar-looking objects (e.g., a red plate vs. a red frisbee) should be distinguished by their functional shape (e.g., rims/aerodynamics), not surface decals.

Your confidence score must reflect your actual visual certainty (e.g., 0.99 for a definite match, 0.65 for a plausible but ambiguous one). Use the full range of 0.0 to 1.0. Do NOT use 0.95 as a default.

Respond ONLY with this JSON (no other text):
{{"match": true, "confidence": <belief_score>, "reasoning": "brief explanation"}}

Or if it does not match:
{{"match": false, "confidence": <belief_score>, "reasoning": "brief explanation"}}"""


def _verify_crop_with_vlm(
    template_b64: str,
    crop_b64: str,
    description: str,
    api_key: str,
    model: str,
) -> tuple[bool, float]:
    """Use VLM to verify if a crop contains the template object.

    Returns (is_match, confidence).
    """
    prompt = RERANK_PROMPT.format(description=description)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "TEMPLATE:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{template_b64}"
                        },
                    },
                    {"type": "text", "text": "CANDIDATE CROP:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{crop_b64}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 200,
        "temperature": 0.1,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(
            OPENROUTER_API_URL, json=payload, headers=headers, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()

        # Parse JSON response
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            is_match = result.get("match", False)
            confidence = float(result.get("confidence", 0.0))
            reasoning = result.get("reasoning", "")
            logger.info(
                f"VLM re-rank: match={is_match}, conf={confidence:.2f}, "
                f"reason={reasoning}"
            )
            return is_match, confidence
    except Exception as e:
        logger.warning(f"VLM re-ranking failed: {e}")

    return False, 0.0


def detect_hybrid(
    template_path: str | Path,
    scene_path: str | Path,
    confidence_threshold: float = 0.3,
    nms_iou_threshold: float = 0.3,
    classical_threshold: float = 0.25,
    model: str = DEFAULT_MODEL,
    crop_expand: float = 1.5,
) -> DetectionResult:
    """Detect template using classical pre-filter + VLM re-ranker.

    1. Run classical detector with a lower threshold to get candidates.
    2. For each candidate, crop the scene region (expanded by crop_expand).
    3. Send each crop + template to VLM for verification.
    4. Keep only VLM-confirmed detections.
    5. If classical finds nothing, fall back to full VLM grounding.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key or api_key == "your-api-key-here":
        raise ValueError(
            "OPENROUTER_API_KEY not set. "
            "Copy .env.example to .env and add your key."
        )

    # Step 1: Get classical candidates (lower threshold to over-generate)
    classical_result = detect_classical(
        template_path,
        scene_path,
        confidence_threshold=classical_threshold,
        nms_iou_threshold=nms_iou_threshold,
    )

    if not classical_result.detections:
        logger.info(
            "Hybrid: no classical candidates, falling back to full VLM"
        )
        from .vlm import detect_vlm

        return detect_vlm(
            template_path,
            scene_path,
            confidence_threshold=confidence_threshold,
            model=model,
        )

    logger.info(
        f"Hybrid: {len(classical_result.detections)} classical candidates "
        f"to re-rank with VLM"
    )

    # Step 2: Prepare images
    scene = load_image_cv2(scene_path)
    template_pil = load_image_pil(template_path)
    template_b64 = image_to_base64(template_pil)
    s_h, s_w = scene.shape[:2]

    # Get template description from VLM
    description = _describe_template(template_b64, api_key, model)

    # Step 3: Re-rank each candidate via VLM
    reranked: list[Detection] = []

    for det in classical_result.detections:
        x, y, w, h = det.bbox

        # Expand the crop region for context
        cx, cy = x + w / 2, y + h / 2
        ew, eh = w * crop_expand, h * crop_expand
        ex1 = max(0, int(cx - ew / 2))
        ey1 = max(0, int(cy - eh / 2))
        ex2 = min(s_w, int(cx + ew / 2))
        ey2 = min(s_h, int(cy + eh / 2))

        crop = scene[ey1:ey2, ex1:ex2]
        if crop.size == 0:
            continue

        crop_pil = Image.fromarray(crop[:, :, ::-1])  # BGR -> RGB
        crop_b64 = image_to_base64(crop_pil)

        is_match, vlm_confidence = _verify_crop_with_vlm(
            template_b64, crop_b64, description, api_key, model
        )

        if is_match:
            # Blend classical and VLM confidence
            blended = 0.4 * det.confidence + 0.6 * vlm_confidence

            if blended >= confidence_threshold:
                reranked.append(
                    Detection(
                        bbox=[round(v, 1) for v in det.bbox],
                        confidence=round(min(1.0, blended), 4),
                    )
                )

    if not reranked:
        # No candidates survived VLM re-ranking — fall back to full VLM
        logger.info("Hybrid: no candidates survived re-ranking, falling back to VLM")
        from .vlm import detect_vlm

        return detect_vlm(
            template_path,
            scene_path,
            confidence_threshold=confidence_threshold,
            model=model,
        )

    # NMS on reranked detections
    boxes = [d.bbox for d in reranked]
    scores = [d.confidence for d in reranked]
    keep = non_max_suppression(boxes, scores, nms_iou_threshold)

    final = [reranked[i] for i in keep]

    return DetectionResult(
        found=len(final) > 0,
        detections=final,
        method="hybrid",
    )