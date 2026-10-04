import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import torch
from tqdm import tqdm
from ultralytics import YOLO

from inference import (
    DEFAULT_MINILM_CKPT,
    DEFAULT_MODEL_NAME,
    choose_device,
    load_minilm_scorer,
    run_yolo_detection,
    score_detections,
    validate_path,
)
from prepare_yolo_coco_tasks import COCO_ID_TO_YOLO_ID, YOLO_COCO_NAMES


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_JSONL = PROJECT_DIR / "derived" / "coco_tasks_test.jsonl"
DEFAULT_IMAGE_ROOT = Path(r"D:\datasets\mscoco")

# Change this one line when you want to test another detector.
#YOLO_WEIGHTS = PROJECT_DIR / "runs" / "detect" / "yolo_coco_tasks" / "weights" / "best.pt"
# YOLO_WEIGHTS = PROJECT_DIR / "runs" / "derived" / "yolo_coco_tasks" / "weights" / "best.pt"
YOLO_WEIGHTS = PROJECT_DIR / "yolov8x.pt"  # general YOLO model weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full pipeline on 1000 labeled rows from derived/*.jsonl. "
            "Accuracy is computed against each row's preferred objects."
        )
    )
    parser.add_argument("--jsonl", default=str(DEFAULT_JSONL))
    parser.add_argument(
        "--image-root",
        default=str(DEFAULT_IMAGE_ROOT),
        help="Folder containing train2014 and val2014. Used when JSONL image paths are from another machine.",
    )
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--balanced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample rows as evenly as possible across task prompts.",
    )
    parser.add_argument("--positive-only", action="store_true", help="Evaluate only rows with preferred objects.")
    parser.add_argument("--yolo-weights", default=str(YOLO_WEIGHTS))
    parser.add_argument("--minilm-ckpt", default=str(DEFAULT_MINILM_CKPT))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--yolo-conf", type=float, default=0.05)
    parser.add_argument("--none-threshold", type=float, default=0.10)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--rank-by", choices=["minilm", "product"], default="minilm")
    parser.add_argument("--output-jsonl", default=str(PROJECT_DIR / "results"/"batch_1000_pipeline_results.jsonl"))
    parser.add_argument("--output-summary", default=str(PROJECT_DIR / "results"/"batch_1000_pipeline_summary.json"))
    parser.add_argument("--output-csv", default=str(PROJECT_DIR /"results" /"batch_1000_pipeline_results.csv"))
    return parser.parse_args()


def load_rows(jsonl_path: Path, positive_only: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if positive_only and not preferred_object_names(row):
                continue
            rows.append(row)
    return rows


def resolve_image_path(entry: Dict[str, Any], image_root: Optional[Path]) -> Path:
    raw = str(entry["image_path"])
    candidate = Path(raw)
    if candidate.exists():
        return candidate

    image_name = PureWindowsPath(raw).name
    if image_name == raw:
        image_name = Path(raw).name
    split_dir = "train2014" if entry.get("split") == "train" else "val2014"
    return candidate if image_root is None else image_root / split_dir / image_name


def preferred_object_names(entry: Dict[str, Any]) -> Set[str]:
    names: Set[str] = set()
    for obj in entry.get("objects", []):
        if int(obj.get("preferred", 0)) != 1:
            continue
        yolo_id = COCO_ID_TO_YOLO_ID.get(int(obj["coco_category_id"]))
        if yolo_id is not None:
            names.add(YOLO_COCO_NAMES[yolo_id])
    return names


def annotated_object_names(entry: Dict[str, Any]) -> Set[str]:
    names: Set[str] = set()
    for obj in entry.get("objects", []):
        yolo_id = COCO_ID_TO_YOLO_ID.get(int(obj["coco_category_id"]))
        if yolo_id is not None:
            names.add(YOLO_COCO_NAMES[yolo_id])
    return names


def sample_rows(rows: Sequence[Dict[str, Any]], num_samples: int, balanced: bool, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    if not balanced:
        shuffled = list(rows)
        rng.shuffle(shuffled)
        return shuffled[:num_samples]

    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_prompt"])].append(row)
    for task_rows in by_task.values():
        rng.shuffle(task_rows)

    sampled: List[Dict[str, Any]] = []
    task_names = sorted(by_task)
    cursor = 0
    while len(sampled) < num_samples and any(by_task.values()):
        task = task_names[cursor % len(task_names)]
        if by_task[task]:
            sampled.append(by_task[task].pop())
        cursor += 1
    return sampled


def top_detections(detections: Iterable[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    ranked = sorted(detections, key=lambda det: det.get("rank_score", 0.0), reverse=True)
    return [
        {
            "class_name": det["class_name"],
            "yolo_conf": det["yolo_conf"],
            "minilm_score": det.get("minilm_score"),
            "rank_score": det.get("rank_score"),
            "bbox_xyxy": det["bbox_xyxy"],
        }
        for det in ranked[:limit]
    ]


def safe_rate(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def add_metric(metrics: Dict[str, Any], task: str, key: str, amount: int = 1) -> None:
    metrics[key] += amount
    metrics["per_task"][task][key] += amount


def build_summary(metrics: Dict[str, Any], selected_counter: Counter) -> Dict[str, Any]:
    summary = {
        "total": metrics["total"],
        "missing_images": metrics["missing_images"],
        "selected": metrics["selected"],
        "below_threshold": metrics["below_threshold"],
        "no_detections": metrics["no_detections"],
        "positive_total": metrics["positive_total"],
        "positive_top1_correct": metrics["positive_top1_correct"],
        "positive_top1_accuracy": safe_rate(metrics["positive_top1_correct"], metrics["positive_total"]),
        "positive_expected_available_by_yolo": metrics["positive_expected_available_by_yolo"],
        "positive_top1_when_expected_detected": safe_rate(
            metrics["positive_top1_correct_when_expected_detected"],
            metrics["positive_expected_available_by_yolo"],
        ),
        "positive_expected_in_top5_when_detected": safe_rate(
            metrics["positive_expected_in_top5_when_detected"],
            metrics["positive_expected_available_by_yolo"],
        ),
        "negative_total": metrics["negative_total"],
        "negative_rejected": metrics["negative_rejected"],
        "negative_rejection_accuracy": safe_rate(metrics["negative_rejected"], metrics["negative_total"]),
        "overall_accuracy": safe_rate(
            metrics["positive_top1_correct"] + metrics["negative_rejected"],
            metrics["positive_total"] + metrics["negative_total"],
        ),
        "prompt_balance": dict(metrics["prompt_balance"]),
        "selected_object_counts": dict(selected_counter.most_common()),
        "per_task": {},
    }

    for task, task_metrics in metrics["per_task"].items():
        task_positive = task_metrics["positive_total"]
        task_negative = task_metrics["negative_total"]
        summary["per_task"][task] = {
            "total": task_metrics["total"],
            "selected": task_metrics["selected"],
            "below_threshold": task_metrics["below_threshold"],
            "no_detections": task_metrics["no_detections"],
            "positive_total": task_positive,
            "positive_top1_correct": task_metrics["positive_top1_correct"],
            "positive_top1_accuracy": safe_rate(task_metrics["positive_top1_correct"], task_positive),
            "negative_total": task_negative,
            "negative_rejected": task_metrics["negative_rejected"],
            "negative_rejection_accuracy": safe_rate(task_metrics["negative_rejected"], task_negative),
            "overall_accuracy": safe_rate(
                task_metrics["positive_top1_correct"] + task_metrics["negative_rejected"],
                task_positive + task_negative,
            ),
        }
    return summary


@torch.no_grad()
def main() -> None:
    args = parse_args()
    jsonl_path = Path(args.jsonl)
    image_root = Path(args.image_root) if args.image_root else None
    yolo_weights = Path(args.yolo_weights)
    minilm_ckpt = Path(args.minilm_ckpt)

    validate_path(jsonl_path, "Derived JSONL")
    validate_path(yolo_weights, "YOLO weights")
    validate_path(minilm_ckpt, "MiniLM checkpoint")
    if image_root is not None:
        validate_path(image_root, "Image root")

    rows = load_rows(jsonl_path, args.positive_only)
    sampled_rows = sample_rows(rows, args.num_samples, args.balanced, args.seed)
    if len(sampled_rows) < args.num_samples:
        print(f"Warning: only sampled {len(sampled_rows)} rows from {jsonl_path}.")

    device = choose_device(args.device)
    yolo_model = YOLO(str(yolo_weights))
    minilm_model = load_minilm_scorer(minilm_ckpt, args.model_name, device)

    output_jsonl = Path(args.output_jsonl)
    output_csv = Path(args.output_csv)
    output_summary = Path(args.output_summary)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, Any] = defaultdict(int)
    metrics["per_task"] = defaultdict(lambda: defaultdict(int))
    metrics["prompt_balance"] = Counter()
    selected_counter: Counter = Counter()

    csv_fields = [
        "index",
        "image",
        "task",
        "status",
        "selected_object",
        "selected_rank_score",
        "selected_minilm_score",
        "selected_yolo_conf",
        "num_detections",
        "preferred_objects",
        "annotated_objects",
        "is_positive_row",
        "correct",
        "expected_available_by_yolo",
        "expected_in_top5",
    ]

    with output_jsonl.open("w", encoding="utf-8") as jsonl_f, output_csv.open(
        "w", encoding="utf-8", newline=""
    ) as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=csv_fields)
        writer.writeheader()

        for idx, entry in enumerate(tqdm(sampled_rows, desc="Batch pipeline test"), start=1):
            task = str(entry["task_prompt"])
            image_path = resolve_image_path(entry, image_root)
            preferred = preferred_object_names(entry)
            annotated = annotated_object_names(entry)

            metrics["total"] += 1
            metrics["prompt_balance"][task] += 1
            metrics["per_task"][task]["total"] += 1

            if not image_path.exists():
                add_metric(metrics, task, "missing_images")
                continue

            detections = run_yolo_detection(yolo_model, image_path, args.yolo_conf, args.max_det)
            scored = score_detections(task, detections, minilm_model, device, args.rank_by)
            selected = None if scored is None else scored["selected_detection"]

            status = "no_detections"
            if selected is not None:
                status = "selected" if selected["rank_score"] >= args.none_threshold else "below_threshold"
            selected_object = None if status != "selected" or selected is None else selected["class_name"]

            ranked_top = [] if scored is None else top_detections(scored["all_detections"], limit=10)
            detected_names = {det["class_name"] for det in detections}
            top5_names = {det["class_name"] for det in ranked_top[:5]}
            expected_available_by_yolo = bool(preferred & detected_names)
            expected_in_top5 = bool(preferred & top5_names)

            is_positive = bool(preferred)
            if is_positive:
                add_metric(metrics, task, "positive_total")
                correct = selected_object in preferred if selected_object else False
                add_metric(metrics, task, "positive_top1_correct", int(correct))
                add_metric(metrics, task, "positive_expected_available_by_yolo", int(expected_available_by_yolo))
                add_metric(
                    metrics,
                    task,
                    "positive_top1_correct_when_expected_detected",
                    int(expected_available_by_yolo and correct),
                )
                add_metric(
                    metrics,
                    task,
                    "positive_expected_in_top5_when_detected",
                    int(expected_available_by_yolo and expected_in_top5),
                )
            else:
                add_metric(metrics, task, "negative_total")
                correct = selected_object is None
                add_metric(metrics, task, "negative_rejected", int(correct))

            add_metric(metrics, task, status)
            if selected_object:
                selected_counter[selected_object] += 1

            row = {
                "index": idx,
                "image": str(image_path),
                "task": task,
                "task_id": entry.get("task_id"),
                "image_id": entry.get("image_id"),
                "status": status,
                "preferred_objects": sorted(preferred),
                "annotated_objects": sorted(annotated),
                "selected_object": selected_object,
                "selected_detection": selected if selected_object else None,
                "num_detections": len(detections),
                "top_detections": ranked_top,
                "is_positive_row": is_positive,
                "correct": correct,
                "expected_available_by_yolo": expected_available_by_yolo,
                "expected_in_top5": expected_in_top5,
            }
            jsonl_f.write(json.dumps(row) + "\n")

            writer.writerow(
                {
                    "index": idx,
                    "image": str(image_path),
                    "task": task,
                    "status": status,
                    "selected_object": selected_object,
                    "selected_rank_score": None if not selected_object else selected["rank_score"],
                    "selected_minilm_score": None if not selected_object else selected["minilm_score"],
                    "selected_yolo_conf": None if not selected_object else selected["yolo_conf"],
                    "num_detections": len(detections),
                    "preferred_objects": ";".join(sorted(preferred)),
                    "annotated_objects": ";".join(sorted(annotated)),
                    "is_positive_row": is_positive,
                    "correct": correct,
                    "expected_available_by_yolo": expected_available_by_yolo,
                    "expected_in_top5": expected_in_top5,
                }
            )

    summary = build_summary(metrics, selected_counter)
    summary.update(
        {
            "jsonl": str(jsonl_path),
            "image_root": str(image_root) if image_root else None,
            "device": str(device),
            "yolo_weights": str(yolo_weights),
            "minilm_checkpoint": str(minilm_ckpt),
            "rank_by": args.rank_by,
            "yolo_conf": args.yolo_conf,
            "none_threshold": args.none_threshold,
            "balanced": args.balanced,
            "positive_only": args.positive_only,
            "outputs": {
                "jsonl": str(output_jsonl),
                "csv": str(output_csv),
                "summary": str(output_summary),
            },
        }
    )
    output_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\nBatch evaluation summary")
    print(f"  total: {summary['total']}")
    print(f"  positive_top1_accuracy: {summary['positive_top1_accuracy']:.4f}")
    print(f"  negative_rejection_accuracy: {summary['negative_rejection_accuracy']:.4f}")
    print(f"  overall_accuracy: {summary['overall_accuracy']:.4f}")
    print(f"  missing_images: {summary['missing_images']}")
    print(f"  wrote JSONL: {output_jsonl}")
    print(f"  wrote CSV: {output_csv}")
    print(f"  wrote summary: {output_summary}")


if __name__ == "__main__":
    main()
