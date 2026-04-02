"""Quantitative evaluation script — computes Precision, Recall, F1 at IoU ≥ 0.5."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.classical import detect_classical
from src.schema import DetectionResult
from src.utils import compute_iou

# All detector imports are deferred to run_evaluation() so that
# heavy optional dependencies (torch, kornia, ultralytics, etc.)
# are only loaded when actually needed.

logger = logging.getLogger(__name__)


def load_ground_truth(path: Path) -> list[dict]:
    """Load ground truth annotations from a JSON file.

    Expected format:
    {
        "annotations": [
            {
                "template": "path/to/template.jpg",
                "scene": "path/to/scene.jpg",
                "scenario": "A",
                "bboxes": [[x, y, w, h], ...]
            }
        ]
    }
    """
    with open(path) as f:
        data = json.load(f)
    return data.get("annotations", [])


def evaluate_detections(
    predicted: list[list[float]],
    ground_truth: list[list[float]],
    iou_threshold: float = 0.5,
) -> dict:
    """Compute precision, recall, and F1 for a single image pair.

    Args:
        predicted: List of predicted bboxes [x, y, w, h].
        ground_truth: List of ground truth bboxes [x, y, w, h].
        iou_threshold: Minimum IoU to count as a true positive.

    Returns:
        Dict with tp, fp, fn, precision, recall, f1.
    """
    matched_gt = set()
    tp = 0
    fp = 0

    for pred_box in predicted:
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def run_evaluation(
    annotations_path: Path,
    method: str = "classical",
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.5,
) -> dict:
    """Run the full evaluation pipeline.

    Returns per-scenario and aggregate metrics.
    """
    from src.vlm import detect_vlm
    from src.hybrid import detect_hybrid

    detectors = {
        "classical": detect_classical,
        "vlm": detect_vlm,
        "hybrid": detect_hybrid,
    }

    # Optionally register DINO-based detectors
    try:
        from src.dino import detect_dino
        detectors["dino"] = detect_dino
    except ImportError:
        logger.warning("DINO not available (install with: pip install -e '.[dino]')")

    try:
        from src.hybrid_dino import detect_hybrid_dino
        detectors["hybrid-dino"] = detect_hybrid_dino
    except ImportError:
        pass

    # Optionally register YOLO-based detectors
    try:
        from src.yolo import detect_yolo
        detectors["yolo"] = detect_yolo
    except ImportError:
        logger.warning("YOLO not available (install with: pip install -e '.[yolo]')")

    try:
        from src.hybrid_yolo import detect_hybrid_yolo
        detectors["hybrid-yolo"] = detect_hybrid_yolo
    except ImportError:
        pass

    # Optionally register LightGlue
    try:
        from src.lightglue import detect_lightglue
        detectors["lightglue"] = detect_lightglue
    except ImportError:
        logger.warning("LightGlue not available (install with: pip install -e '.[lightglue]')")

    # Optionally register E-LoFTR
    try:
        from src.eloftr import detect_eloftr
        detectors["eloftr"] = detect_eloftr
    except ImportError:
        logger.warning("E-LoFTR not available (install with: pip install -e '.[eloftr]')")

    # Optionally register SAM
    try:
        from src.sam import detect_sam
        detectors["sam"] = detect_sam
    except ImportError:
        logger.warning("SAM not available (install with: pip install -e '.[sam]')")

    if method not in detectors:
        available = ", ".join(sorted(detectors.keys()))
        raise ValueError(
            f"Method '{method}' is not available. "
            f"Installed methods: {available}"
        )
    detector = detectors[method]

    annotations = load_ground_truth(annotations_path)
    base_dir = annotations_path.parent

    per_scenario: dict[str, list[dict]] = {}
    all_results: list[dict] = []

    for ann in annotations:
        template_path = base_dir / ann["template"]
        scene_path = base_dir / ann["scene"]
        scenario = ann.get("scenario", "unknown")
        gt_bboxes = ann["bboxes"]

        try:
            result: DetectionResult = detector(
                str(template_path),
                str(scene_path),
                confidence_threshold=confidence_threshold,
            )
            pred_bboxes = [d.bbox for d in result.detections]
        except Exception as e:
            logger.error(f"Detection failed for {scene_path}: {e}")
            pred_bboxes = []

        metrics = evaluate_detections(pred_bboxes, gt_bboxes, iou_threshold)
        metrics["scene"] = str(scene_path.name)
        metrics["scenario"] = scenario

        if scenario not in per_scenario:
            per_scenario[scenario] = []
        per_scenario[scenario].append(metrics)
        all_results.append(metrics)

    # Aggregate per scenario
    scenario_summary = {}
    for scenario, results in sorted(per_scenario.items()):
        total_tp = sum(r["tp"] for r in results)
        total_fp = sum(r["fp"] for r in results)
        total_fn = sum(r["fn"] for r in results)

        p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        scenario_summary[scenario] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "num_images": len(results),
        }

    # Overall aggregate
    total_tp = sum(r["tp"] for r in all_results)
    total_fp = sum(r["fp"] for r in all_results)
    total_fn = sum(r["fn"] for r in all_results)
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f = (
        2 * overall_p * overall_r / (overall_p + overall_r)
        if (overall_p + overall_r) > 0
        else 0.0
    )

    return {
        "method": method,
        "iou_threshold": iou_threshold,
        "confidence_threshold": confidence_threshold,
        "per_scenario": scenario_summary,
        "overall": {
            "precision": round(overall_p, 4),
            "recall": round(overall_r, 4),
            "f1": round(overall_f, 4),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        },
        "per_image": all_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate template matching across test scenarios"
    )
    parser.add_argument(
        "--annotations",
        "-a",
        required=True,
        help="Path to ground truth annotations JSON",
    )
    ALL_METHODS = [
        "classical", "vlm", "hybrid",
        "dino", "hybrid-dino",
        "yolo", "hybrid-yolo",
        "lightglue", "eloftr", "sam",
        "all",  # special: run every installed method and compare
    ]
    parser.add_argument(
        "--method",
        "-m",
        choices=ALL_METHODS,
        default="classical",
        help="Detection method to evaluate (use 'all' to compare every installed method)",
    )
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--output", "-o", help="Save results to JSON file")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.method == "all":
        _run_comparison(args)
    else:
        results = run_evaluation(
            Path(args.annotations),
            method=args.method,
            confidence_threshold=args.threshold,
            iou_threshold=args.iou,
        )
        _print_single_report(results, args)


def _print_single_report(results: dict, args) -> None:
    """Print the evaluation table for a single method."""
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS — Method: {results['method']} | IoU ≥ {args.iou}")
    print("=" * 70)
    print(f"{'Scenario':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 70)

    for scenario, metrics in sorted(results["per_scenario"].items()):
        print(
            f"{scenario:<12} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f} {metrics['tp']:>5} {metrics['fp']:>5} {metrics['fn']:>5}"
        )

    overall = results["overall"]
    print("-" * 70)
    print(
        f"{'OVERALL':<12} {overall['precision']:>10.4f} {overall['recall']:>10.4f} "
        f"{overall['f1']:>10.4f} {overall['tp']:>5} {overall['fp']:>5} {overall['fn']:>5}"
    )
    print("=" * 70)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results saved to: {args.output}")


def _run_comparison(args) -> None:
    """Run every installed method and print a side-by-side comparison."""
    import time

    # Discover which methods are installed by doing a dummy run_evaluation import check
    # We'll just try each method and skip those that fail to import.
    candidate_methods = [
        "classical", "vlm", "hybrid",
        "dino", "hybrid-dino",
        "yolo", "hybrid-yolo",
        "lightglue", "eloftr", "sam",
    ]

    all_results = {}
    for method in candidate_methods:
        try:
            t0 = time.time()
            results = run_evaluation(
                Path(args.annotations),
                method=method,
                confidence_threshold=args.threshold,
                iou_threshold=args.iou,
            )
            elapsed = time.time() - t0
            results["elapsed_seconds"] = round(elapsed, 1)
            all_results[method] = results
            logger.info(f"Finished {method} in {elapsed:.1f}s")
        except (ValueError, ImportError) as e:
            logger.warning(f"Skipping {method}: {e}")
        except Exception as e:
            logger.error(f"Error running {method}: {e}")

    if not all_results:
        print("No methods could be evaluated.")
        return

    # Print comparison table
    print("\n" + "=" * 90)
    print(f"MULTI-METHOD COMPARISON | IoU ≥ {args.iou} | Confidence ≥ {args.threshold}")
    print("=" * 90)
    print(
        f"{'Method':<16} {'Precision':>10} {'Recall':>10} {'F1':>10} "
        f"{'TP':>5} {'FP':>5} {'FN':>5} {'Time (s)':>10}"
    )
    print("-" * 90)

    # Sort by F1 descending
    sorted_methods = sorted(
        all_results.keys(),
        key=lambda m: all_results[m]["overall"]["f1"],
        reverse=True,
    )
    for method in sorted_methods:
        o = all_results[method]["overall"]
        t = all_results[method].get("elapsed_seconds", 0)
        print(
            f"{method:<16} {o['precision']:>10.4f} {o['recall']:>10.4f} {o['f1']:>10.4f} "
            f"{o['tp']:>5} {o['fp']:>5} {o['fn']:>5} {t:>10.1f}"
        )

    print("=" * 90)

    # Per-scenario breakdown for each method
    scenarios = set()
    for r in all_results.values():
        scenarios.update(r["per_scenario"].keys())

    for scenario in sorted(scenarios):
        print(f"\n--- Scenario {scenario} ---")
        print(f"{'Method':<16} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 50)
        for method in sorted_methods:
            s = all_results[method]["per_scenario"].get(scenario)
            if s:
                print(
                    f"{method:<16} {s['precision']:>10.4f} {s['recall']:>10.4f} {s['f1']:>10.4f}"
                )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    main()
