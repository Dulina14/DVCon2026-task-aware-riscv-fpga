import argparse
import json
from pathlib import Path

from inference import (
    DEFAULT_MINILM_CKPT,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_IMAGE,
    DEFAULT_YOLO_WEIGHTS,
    print_summary,
    run_pipeline,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_JSON = PROJECT_DIR /"results"/ "pipeline_result.json"


def clean_input_path(raw_path: str) -> str:
    return raw_path.strip().strip('"').strip("'")


def prompt_required(prompt: str) -> str:
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("Input cannot be empty. Please try again.")


def build_args(image_path: str, task_prompt: str) -> argparse.Namespace:
    return argparse.Namespace(
        image=image_path,
        task=task_prompt,
        yolo_weights=str(DEFAULT_YOLO_WEIGHTS),
        minilm_ckpt=str(DEFAULT_MINILM_CKPT),
        model_name=DEFAULT_MODEL_NAME,
        text_mode="finetuned",
        device="auto",
        yolo_conf=0.05,
        none_threshold=0.10,
        max_det=300,
        rank_by="minilm",
        output_image=str(DEFAULT_OUTPUT_IMAGE),
        output_json=str(DEFAULT_OUTPUT_JSON),
        no_image_output=False,
        print_json=False,
    )


def main() -> None:
    print("COCO-Task YOLO + MiniLM Interactive Pipeline")
    image_path = clean_input_path(prompt_required("Enter image path: "))
    task_prompt = prompt_required('Enter task prompt, for example "serve wine": ')

    args = build_args(image_path, task_prompt)
    output = run_pipeline(args)
    text = json.dumps(output, indent=2)

    print_summary(output)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(text + "\n", encoding="utf-8")

    print(f"\nJSON output saved to: {output_json}")
    if output.get("output_image"):
        print(f"Annotated image saved to: {output['output_image']}")


if __name__ == "__main__":
    main()
