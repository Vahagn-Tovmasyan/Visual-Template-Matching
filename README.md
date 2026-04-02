# Visual Template Matching

Locate small visual templates inside larger scene images using **Classical Computer Vision**, **Vision Language Models (VLMs)**, or a **Hybrid** approach that combines both.

## Overview

Given a template image T (a small crop of a visual target) and a scene image S (a larger image), the system:

1. **Determines** whether T appears in S.
2. **Returns** the location of each occurrence as a bounding box `[x, y, w, h]` in pixels.
3. **Returns** a confidence score (0-1) per detection.

### Detection Methods

| Method | How it works | Strengths | Weaknesses |
|--------|-------------|-----------|------------|
| **Classical** | Multi-scale template matching + SIFT feature matching + colour histogram hard-gate | Fast (~200 ms), free, no API needed | Brittle to rotation, occlusion, large appearance change |
| **VLM** | Qwen2.5-VL 72B via OpenRouter: describe template, then prompt for grounded bboxes | Understands semantics; handles viewpoint and scale changes | Slower (~5-10 s), costs ~$0.01/call, bbox precision is approximate |
| **Hybrid** | Classical pre-filter (low threshold) -> VLM re-ranker per crop | Best accuracy: CV finds candidates fast, VLM confirms identity | Requires API key, cost proportional to number of candidates |
| **DINO** | Grounding DINO (local model `grounding-dino-tiny`) | Excellent multi-instance detection and high-quality bounding boxes | Requires ~2GB env footprint; slow on CPU-only machines (~4s) |
| **DINO-Hybrid** | DINO pre-filter (low threshold) -> VLM re-ranker per crop | Best in class recall and geometric precision | Slowest combined pipeline; high compute + API cost |
| **YOLO** | YOLO11 Object Detector (local model `yolo11n.pt`) | Ultra-fast; great for COCO classes | Fails on arbitrary custom objects not in the COCO dataset |
| **YOLO-Hybrid** | YOLO pre-filter -> VLM re-ranker per crop | High-speed semantic matching pipeline | Only works if the template is a recognized COCO class |
| **LightGlue** | DISK keypoints + LightGlue attention-based matcher (via kornia) + RANSAC homography | Best sparse matcher; handles large viewpoint/scale changes | Needs kornia install (~500MB); matches pixel patterns, not semantics |
| **E-LoFTR** | Kornia LoFTR dense feature matching + Sequential RANSAC + adaptive confidence | Dense matches on low-texture objects where sparse fails | Best on source image; struggles across different scenes |
| **SAM 2** | SAM 2.1 Hiera — Segments entire scene, matches segments via colour/shape/size scoring | Finds objects by appearance without texture matching | CPU-only is slow (~60-90s); colour-based so may match similar-looking objects |

I evaluated classical matching, learned feature matching, VLM-based semantic matching, and detector-assisted hybrids. On this benchmark, Grounding DINO and YOLO-based methods achieved the strongest overall localization, while VLM-based methods were useful for semantic verification but less reliable as standalone localizers. For arbitrary-object template matching, learned and semantic matching approaches are more general than YOLO, which depends on known classes. DINO + Qwen method performed significantly better than the standalone models.

## Quick Start

### 1. Install

```bash
git clone <repo-url>
cd visual-template-matching
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[api,dev]"

# Optional backends (each adds ~1-2GB for PyTorch + model weights):
pip install -e ".[dino]"       # Grounding DINO zero-shot detector
pip install -e ".[lightglue]"  # SuperPoint + LightGlue sparse matcher
pip install -e ".[eloftr]"     # E-LoFTR dense feature matcher (via Kornia)
pip install -e ".[sam]"       # SAM 2.1 segmentation-based detector (local)
pip install -e ".[all]"        # Everything (API, all models, dev tools)
```

### 2. Configure API Key (for VLM / Hybrid)

```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

The Classical method works without an API key.

### 3. Run CLI

```bash
# Classical detection (no API key needed)
python detect.py -t test_images/000000017029_Template.jpg \
                 -s test_images/000000017029.jpg \
                 -m classical

# VLM detection
python detect.py -t test_images/000000017029_Template.jpg \
                 -s test_images/000000017029.jpg \
                 -m vlm

# Hybrid with annotated output
python detect.py -t template.jpg -s scene.jpg -m hybrid -o result.jpg --verbose
```

### 4. Launch Gradio UI

```bash
python app.py
# Open http://localhost:7860
```

### 5. Run Tests

```bash
pytest tests/ -v
```

### 6. Run Evaluation

```bash
# Evaluate a single method
python evaluate.py --annotations test_images/annotations.json --method classical
python evaluate.py --annotations test_images/annotations.json --method vlm

# Compare ALL installed methods side-by-side (sorted by F1)
python evaluate.py --annotations test_images/annotations.json --method all

# Save results to JSON for further analysis
python evaluate.py -a test_images/annotations.json -m all -o results.json
```

## Project Structure

```
visual-template-matching/
├── src/
│   ├── __init__.py        # Package init
│   ├── schema.py          # Pydantic models (Detection, DetectionResult)
│   ├── utils.py           # Image loading, NMS, IoU, drawing, caching
│   ├── classical.py       # Multi-scale template matching + SIFT + colour gate
│   ├── vlm.py             # Qwen2.5-VL via OpenRouter (two-step grounding)
│   ├── hybrid.py          # Classical pre-filter -> VLM re-ranker
│   ├── dino.py            # Grounding DINO zero-shot detection (local)
│   ├── hybrid_dino.py     # DINO pre-filter -> VLM re-ranker
│   ├── yolo.py            # YOLO11 COCO-class pseudo-template matcher
│   ├── hybrid_yolo.py     # YOLO pre-filter -> VLM re-ranker
│   ├── lightglue.py       # DISK + LightGlue sparse matcher (via kornia)
│   ├── eloftr.py          # E-LoFTR dense feature matcher (via Kornia)
│   └── sam.py             # SAM 2.1 Hiera mask-generation pipeline
├── tests/
│   ├── test_schema.py     # Schema validation tests
│   ├── test_utils.py      # Utility function tests
│   ├── test_classical.py  # Classical detector tests (synthetic + real images)
│   └── test_vlm.py        # VLM response parsing tests (no API calls)
├── test_images/           # Test images for all four scenarios
│   └── annotations.json   # Ground truth bounding boxes
├── detect.py              # CLI entry point
├── app.py                 # Gradio interactive UI (with comparison tab)
├── api.py                 # FastAPI endpoint
├── evaluate.py            # Quantitative evaluation (P/R/F1 at IoU >= 0.5)
├── Dockerfile             # Container deployment
├── pyproject.toml         # Dependencies (pinned)
├── .env.example           # API key template
└── README.md
```

## Design Decisions

### Why Classical CV as baseline?

Classical methods (template matching + SIFT) have zero latency or cost overhead and work well under constrained conditions. They provide a meaningful comparison point: where classical methods succeed, we don't need a VLM. Where they fail (scale change, occlusion, clutter), the VLM adds real value.

**Multi-scale template matching** handles translations and moderate scale changes by sweeping across 30 scale factors (0.15x to 2.5x). At each scale, we use `cv2.TM_CCOEFF_NORMED` and extract up to 3 peaks via iterative masking.

**SIFT feature matching** adds geometric robustness: it finds corresponding keypoints between template and scene, estimates a homography via RANSAC, and projects the template corners to get a bounding box even under perspective distortion.

**Colour histogram hard-gate** is the key false-positive filter. After finding geometric candidates, we compare HSV histograms between the candidate crop and the template. Candidates below a similarity threshold of 0.3 are rejected outright (not merely downweighted). This eliminates most structurally-similar but colour-wrong matches (e.g., a grey patch matching a red frisbee shape). Survivors have their confidence blended: 60% structural match + 40% colour similarity.

**Aspect ratio validation** rejects candidates whose width/height ratio is more than 3x different from the template's. This cheaply eliminates degenerate matches at extreme scales.

### Why Qwen2.5-VL via OpenRouter?

Qwen2.5-VL supports visual grounding and can output bounding box coordinates when prompted. Via OpenRouter, we access the 72B parameter variant without needing a local GPU. The model receives both the template and scene image, along with a natural-language description.

The **two-step approach** (describe template -> ground in scene) works better than single-shot for two reasons. First, it forces the model to reason about what makes the template distinctive before searching. Second, it gives the model an explicit natural-language anchor to match against, which improves recall on semantically similar but visually different instances.

We request **normalised [0,1] coordinates** to avoid confusion between the model's internal resolution and the actual image size. The parser also handles Qwen's 0-1000 scale convention and raw pixel values as fallbacks.

### Why Hybrid?

VLMs are expensive and slow. The hybrid pipeline runs classical detection first (milliseconds), generates a list of candidate regions, then sends only those crops to the VLM for verification. This typically reduces cost by 60-80% compared to full VLM-only analysis, because verifying a small crop is cheaper than grounding across an entire scene.

If classical detection finds nothing (common with occluded or heavily transformed targets), the hybrid falls back to full VLM grounding. This ensures recall stays high even when classical methods fail completely.

### VLM Prompting Strategy

Getting reliable bounding boxes from a VLM required several prompt-engineering iterations:

1. **Two-step reasoning.** We first ask the model to describe the template in ≤3 words (e.g. "red plastic frisbee"), then feed that description back alongside both images when grounding. This forces the model to commit to what it is looking for before searching, and consistently outperforms single-shot prompting.

2. **Normalised [0,1] coordinates.** We request coordinates as floats between 0 and 1 rather than pixel values. This avoids confusion between the model's internal resolution and the actual image dimensions. The parser also handles Qwen's 0-1000 convention and raw pixels as fallbacks.

3. **Step-by-step scanning.** The prompt instructs the model to "scan the entire scene systematically from left to right, top to bottom" and compare each candidate region against the template. This structured approach improves recall on multi-instance scenes.

4. **Post-hoc confidence calibration.** VLMs tend to output a fixed confidence (often 0.95) regardless of actual certainty. We recalibrate each detection by computing an HSV colour histogram similarity and a normalised cross-correlation (NCC) template-matching score against the crop. The calibrated confidence is 20% VLM raw score + 80% visual verification, giving meaningful score variation across detections.

### Why Sparse and Dense Feature Matchers?

**LightGlue (DISK + LightGlue)** excels at finding the exact same object across viewpoint and scale changes. It extracts DISK keypoints, matches them with attention-based LightGlue, then uses Sequential RANSAC to estimate multiple homographies for multi-instance detection. It works best when the template and scene share the same pixel-level texture.

**E-LoFTR** is a dense feature matcher that produces correspondences even on low-texture objects where sparse methods fail. It is strongest when matching the template against its source image, but struggles across different scenes because it matches pixel patterns rather than semantic concepts.

Both matchers are complemented by an HSV colour histogram hard-gate to reject geometrically plausible but colour-wrong matches.

### Why SAM 2?

SAM 2 (Segment Anything Model 2.1) takes a fundamentally different approach: it segments the entire scene into masks, then scores each segment against the template using colour histograms, Hu moment shape similarity, relative size, and aspect ratio. This lets it find objects without any texture matching, which is useful when the template appears at very different scales or viewpoints. The trade-off is that SAM matches by appearance, not identity, it may match a red ball when searching for a red frisbee.

### Scenario D Benchmarks & The "Red Ball" Test

I used Scenario D (AI-generated multi-instance park scene) to stress-test the various methods. I also added a **red ball** as a distractor to test semantic grounding vs. simple colour/shape matching.

| Method | Scenario D (6 Frisbees) | The "Red Ball" Test (Distractor) | Verdict |
| :--- | :--- | :--- | :--- |
| **Hybrid-YOLO** | **The best one** (although not a matcher). Highest confidence ($0.97+$) and perfect count (6/6). | High confidence. | Best for "clean" detection. |
| **Hybrid-DINO** | Really Strong. 6/6 detections with solid confidence ($0.68+$). | **Failed.** High confidence ($0.94$) on the ball. | DINO is too "generous" with semantic similarity. |
| **SAM 2** | Noisy. Found 8 objects (2 false positives) and lower confidence. | **Fail.** Detected the ball as a frisbee. | SAM 2 lacks "class knowledge"; it sees "red round thing" and segments. |
| **ELoFTR** | **Fail.** `found: false`. | N/A | Likely failed due to low resolution or lack of matching keypoints. |
| **VLM (Qwen)** | Accurate but cautious. Found all 6 but with lower confidence ($0.30$). | **Success.** It correctly did not flag the ball. | Qwen has the best "reasoning" but poor "localization." |

---

## Test Scenarios

All scenarios use the same template: a cropped red frisbee from a dog-catching photo.

| Scenario | Image | Condition | What it tests |
|----------|-------|-----------|---------------|
| A — Clean | Dog catching frisbee (source) | Template present once, same scale | Baseline correctness |
| B — Scale/Pose | Frisbee photos | Different scales and viewing angles | Geometric robustness |
| C — Occlusion | Disc golf chains, person holding | Partially hidden by objects/hands | Partial match tolerance |
| D — Multi-instance | AI-generated park scene | Six frisbees at different positions | Recall and duplicate suppression |

## Known Limitations

1. **Template matching is brittle to rotation.** Pure `cv2.matchTemplate` has no rotation invariance. SIFT compensates but only when enough textured keypoints exist. A rotation sweep (testing 36 angles) would add ~36x latency per scale, which is why it's omitted.

2. **VLM bbox precision is approximate.** VLMs were not trained for pixel-precise bounding box regression. Coordinates can be off by 20-50px. Step-by-step prompting and normalised coordinates help but don't eliminate this. The hybrid approach sidesteps this by using classical boxes (pixel-precise) and VLM only for identity verification.

3. **VLM may hallucinate detections.** Occasionally the VLM reports confident matches for objects that merely resemble the template (e.g., a red ball vs. red frisbee). The hybrid pipeline mitigates this by requiring classical corroboration first.

4. **Colour-dependent verification.** The HSV histogram gate is effective for distinctly-coloured objects (red frisbee against green grass) but would struggle with monochrome or textured objects where colour is not discriminative. A structural similarity (SSIM) check could complement this for colour-neutral templates.

5. **No tiled/hierarchical search.** For very large images (>4K) with tiny targets (<30px), the VLM may miss detections because the target occupies too few tokens in the vision encoder. A tiled search that splits the scene into overlapping regions would improve recall at the cost of more API calls.

6. **API cost.** Each VLM call uses ~2000-4000 tokens. At Qwen2.5-VL-72B pricing via OpenRouter, a single scene analysis costs ~$0.01-0.02. The caching system (keyed on image hash + prompt hash) prevents duplicate calls within a session.

7. **YOLO is limited to COCO classes.** YOLO11 can only detect objects belonging to its 80 pre-trained COCO classes. If the template image contains an object not in COCO (e.g., a custom industrial part), YOLO will fail to identify the class and return zero detections. For arbitrary object types, prefer Classical, VLM, or DINO.

8. **Feature matchers need shared texture.** LightGlue and E-LoFTR match pixel-level patterns, not semantic concepts. They perform excellently when the template is cropped from the same photo (Scenario A) but degrade on different scenes where the object has a different texture, lighting, or background. For cross-scene matching, VLM-based methods are more robust.

9. **SAM 2 matches by appearance, not identity.** Because SAM scores segments using colour histograms and shape metrics, it may match visually similar but semantically different objects (e.g., a red ball when searching for a red frisbee). It also runs slowly on CPU-only machines (~60-90s per image).

10. **DISK requires dimensions divisible by 16.** The DISK feature extractor uses a U-Net architecture that requires input dimensions to be multiples of 16. We handle this by padding with zeros, but very small templates (<16px) may not produce enough keypoints for reliable matching.