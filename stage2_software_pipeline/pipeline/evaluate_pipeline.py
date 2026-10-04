import argparse
import json
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Optional, Set

import torch
from tqdm import tqdm
from ultralytics import YOLO

from inference import (
    DEFAULT_MINILM_CKPT,
    DEFAULT_MODEL_NAME,
    DEFAULT_YOLO_WEIGHTS,
    load_minilm_scorer,
    run_yolo_detection,
    score_detections,
    choose_device,
    validate_path,
)
from prepare_yolo_coco_tasks import COCO_ID_TO_YOLO_ID, YOLO_COCO_NAMES


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TEST_JSONL = PROJECT_DIR / "derived" / "coco_tasks_test.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the full COCO-task pipeline: YOLO detects visible COCO objects, "
            "then fine-tuned MiniLM selects the best detected class for the task."
        )
    )
    parser.add_argument("--jsonl", default=str(DEFAULT_TEST_JSONL))
    parser.add_argument("--image-root", default=None, help="Optional folder containing train2014/val2014.")
    parser.add_argument("--yolo-weights", default=str(DEFAULT_YOLO_WEIGHTS))
    parser.add_argument("--minilm-ckpt", default=str(DEFAULT_MINILM_CKPT))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--yolo-conf", type=float, default=0.05)
    parser.add_argument("--none-threshold", type=float, default=0.10)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--rank-by", choices=["minilm", "product"], default="minilm")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-json", default=str(PROJECT_DIR / "pipeline_eval_results.json"))
    return parser.parse_args()


def load_rows(jsonl_path: Path, max_samples: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def resolve_image_path(entry: Dict[str, Any], image_root: Optional[Path]) -> Path:
    raw = str(entry["image_path"])
    candidate = Path(raw)
    if candidate.exists():
        return candidate

    if image_root is None:
        return candidate

    image_name = PureWindowsPath(raw).name
    if image_name == raw:
        image_name = Path(raw).name
    split_dir = "train2014" if entry.get("split") == "train" else "val2014"
    return image_root / split_dir / image_name


def preferred_object_names(entry: Dict[str, Any]) -> Set[str]:
    names: Set[str] = set()
    for obj in entry.get("objects", []):
        if int(obj.get("preferred", 0)) == 1:
            coco_id = int(obj["coco_category_id"])
            yolo_idx = COCO_ID_TO_YOLO_ID.get(coco_id)
            if yolo_idx is not None:
                names.add(YOLO_COCO_NAMES[yolo_idx])
    return names


def summarize(metrics: Dict[str, Any]) -> None:
    positive_total = max(metrics["positive_task_total"], 1)
    all_negative_total = max(metrics["all_negative_task_total"], 1)
    print("\nFull pipeline evaluation")
    print(f"  rows: {metrics['rows']}")
    print(f"  evaluated_images: {metrics['evaluated_images']}")
    print(f"  missing_images: {metrics['missing_images']}")
    print(f"  no_detections: {metrics['no_detections']}")
    print(
        "  positive top1 accuracy: "
        f"{metrics['positive_top1_correct']}/{metrics['positive_task_total']} "
        f"({metrics['positive_top1_correct'] / positive_total:.4f})"
    )
    print(
        "  all-negative rejection accuracy: "
        f"{metrics['all_negative_rejected']}/{metrics['all_negative_task_total']} "
        f"({metrics['all_negative_rejected'] / all_negative_total:.4f})"
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    jsonl_path = Path(args.jsonl)
    yolo_weights = Path(args.yolo_weights)
    minilm_ckpt = Path(args.minilm_ckpt)
    image_root = Path(args.image_root) if args.image_root else None

    validate_path(jsonl_path, "JSONL")
    validate_path(yolo_weights, "YOLO weights")
    validate_path(minilm_ckpt, "MiniLM checkpoint")
    if image_root is not None:
        validate_path(image_root, "Image root")

    device = choose_device(args.device)
    yolo_model = YOLO(str(yolo_weights))
    minilm_model = load_minilm_scorer(minilm_ckpt, args.model_name, device)

    rows = load_rows(jsonl_path, args.max_samples)
    metrics: Dict[str, Any] = {
        "rows": len(rows),
        "evaluated_images": 0,
        "missing_images": 0,
        "no_detections": 0,
        "positive_task_total": 0,
        "positive_top1_correct": 0,
        "all_negative_task_total": 0,
        "all_negative_rejected": 0,
    }
    examples: List[Dict[str, Any]] = []

    for entry in tqdm(rows, desc="Evaluating"):
        image_path = resolve_image_path(entry, image_root)
        if not image_path.exists():
            metrics["missing_images"] += 1
            continue

        detections = run_yolo_detection(yolo_model, image_path, args.yolo_conf, args.max_det)
        metrics["evaluated_images"] += 1
        positives = preferred_object_names(entry)
        has_positive = bool(positives)

        scored = score_detections(entry["task_prompt"], detections, minilm_model, device, args.rank_by)
        selected = None if scored is None else scored["selected_detection"]
        if scored is None:
            metrics["no_detections"] += 1

        status = "no_detections"
        if selected is not None:
            status = "selected" if selected["rank_score"] >= args.none_threshold else "below_threshold"

        selected_name = None if status != "selected" or selected is None else selected["class_name"]
        if has_positive:
            metrics["positive_task_total"] += 1
            metrics["positive_top1_correct"] += int(selected_name in positives)
        else:
            metrics["all_negative_task_total"] += 1
            metrics["all_negative_rejected"] += int(selected_name is None)

        if len(examples) < 25:
            examples.append(
                {
                    "image": str(image_path),
                    "task": entry["task_prompt"],
                    "preferred_objects": sorted(positives),
                    "selected_object": selected_name,
                    "status": status,
                    "num_detections": len(detections),
                    "top_detections": []
                    if scored is None
                    else sorted(scored["all_detections"], key=lambda det: det["rank_score"], reverse=True)[:10],
                }
            )

    output = {
        "jsonl": str(jsonl_path),
        "image_root": str(image_root) if image_root else None,
        "device": str(device),
        "rank_by": args.rank_by,
        "yolo_conf": args.yolo_conf,
        "none_threshold": args.none_threshold,
        "yolo_weights": str(yolo_weights),
        "minilm_checkpoint": str(minilm_ckpt),
        "metrics": metrics,
        "examples": examples,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    summarize(metrics)
    print("  wrote:", output_path)


if __name__ == "__main__":
    main()
