import argparse
import csv
import json
from pathlib import Path

from inference import (
    DEFAULT_MINILM_CKPT,
    DEFAULT_MODEL_NAME,
    DEFAULT_YOLO_WEIGHTS,
    choose_device,
    draw_pipeline_result,
    load_minilm_scorer,
    print_summary,
    run_yolo_detection,
    score_detections,
    validate_path,
)
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results" / "batch_inference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run task-aware object selection on a batch of image/task pairs from a CSV file. "
            "CSV columns: image_path, task_prompt"
        )
    )
    parser.add_argument("--csv", required=True, help="CSV file with columns: image_path, task_prompt")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--yolo-weights", default=str(DEFAULT_YOLO_WEIGHTS))
    parser.add_argument("--minilm-ckpt", default=str(DEFAULT_MINILM_CKPT))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--yolo-conf", type=float, default=0.05)
    parser.add_argument("--none-threshold", type=float, default=0.10)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--rank-by", choices=["minilm", "product"], default="minilm")
    parser.add_argument("--no-image-output", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value.lower())
    return "_".join(part for part in cleaned.split("_") if part)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.csv)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    json_dir = output_dir / "json"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    yolo_weights = Path(args.yolo_weights)
    minilm_ckpt = Path(args.minilm_ckpt)
    validate_path(input_csv, "Input CSV")
    validate_path(yolo_weights, "YOLO weights")
    validate_path(minilm_ckpt, "MiniLM checkpoint")

    device = choose_device(args.device)
    yolo_model = YOLO(str(yolo_weights))
    minilm_model = load_minilm_scorer(minilm_ckpt, args.model_name, device)

    summary_rows = []
    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_path", "task_prompt"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required column(s): {', '.join(sorted(missing))}")

        for index, row in enumerate(reader, start=1):
            image_path = row["image_path"].strip().strip('"').strip("'")
            task_prompt = row["task_prompt"].strip()
            if not image_path or not task_prompt:
                print(f"[{index}] Skipped: missing image_path or task_prompt")
                continue

            stem = f"{index:04d}_{safe_name(task_prompt)}_{Path(image_path).stem}"
            output_image = image_dir / f"{stem}.jpg"
            output_json = json_dir / f"{stem}.json"

            print(f"\n[{index}] {image_path} | task: {task_prompt}")
            image = Path(image_path)
            validate_path(image, "Image")
            detections = run_yolo_detection(yolo_model, image, args.yolo_conf, args.max_det)
            scored = score_detections(task_prompt, detections, minilm_model, device, args.rank_by)

            output = {
                "image": str(image),
                "task": task_prompt,
                "device": str(device),
                "rank_by": args.rank_by,
                "text_mode": "finetuned",
                "yolo_weights": str(yolo_weights),
                "minilm_checkpoint": str(minilm_ckpt),
                "minilm_model_name": args.model_name,
                "num_detections": len(detections),
                "selected_object": None,
                "selected_detection": None,
                "all_detections": detections,
            }

            if scored is None:
                output["status"] = "no_detections"
            else:
                selected = scored["selected_detection"]
                output["selected_object"] = selected["class_name"]
                output["selected_detection"] = selected
                output["all_detections"] = scored["all_detections"]
                output["status"] = "selected" if selected["rank_score"] >= args.none_threshold else "below_threshold"
                if not args.no_image_output and output["status"] == "selected":
                    draw_pipeline_result(image, selected, output_image)
                    output["output_image"] = str(output_image)

            print_summary(output)

            output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            selected = output.get("selected_detection") or {}
            summary_rows.append(
                {
                    "index": index,
                    "image_path": image_path,
                    "task_prompt": task_prompt,
                    "status": output.get("status"),
                    "selected_object": output.get("selected_object"),
                    "rank_score": selected.get("rank_score"),
                    "minilm_score": selected.get("minilm_score"),
                    "yolo_conf": selected.get("yolo_conf"),
                    "num_detections": output.get("num_detections"),
                    "output_image": output.get("output_image"),
                    "output_json": str(output_json),
                }
            )

    summary_csv = output_dir / "batch_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "index",
            "image_path",
            "task_prompt",
            "status",
            "selected_object",
            "rank_score",
            "minilm_score",
            "yolo_conf",
            "num_detections",
            "output_image",
            "output_json",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nBatch complete. Summary CSV: {summary_csv}")
    print(f"Detailed JSON files: {json_dir}")
    if not args.no_image_output:
        print(f"Annotated images: {image_dir}")


if __name__ == "__main__":
    main()
