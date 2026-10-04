import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoTokenizer
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
#DEFAULT_YOLO_WEIGHTS = PROJECT_DIR / "runs" / "detect" / "yolo_coco_tasks" / "weights" / "best.pt"
DEFAULT_YOLO_WEIGHTS = PROJECT_DIR / "yolov8x.pt"  # general YOLO model weights
DEFAULT_MINILM_CKPT = PROJECT_DIR / "checkpoints" / "finetuned_text" / "minilm_task_text_best.pt"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUTPUT_IMAGE = PROJECT_DIR / "results"/"result.jpg"
MAX_LENGTH = 32

def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


class MiniLMTaskObjectScorer(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def encode_texts(self, texts: List[str], device: torch.device, max_length: int) -> torch.Tensor:
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        outputs = self.encoder(**tokens)
        emb = mean_pooling(outputs.last_hidden_state, tokens["attention_mask"])
        return emb / torch.clamp(emb.norm(dim=-1, keepdim=True), min=1e-9)

    def forward(
        self,
        task_texts: List[str],
        object_texts: List[str],
        device: torch.device,
        max_length: int,
    ) -> torch.Tensor:
        task_emb = self.encode_texts(task_texts, device, max_length)
        obj_emb = self.encode_texts(object_texts, device, max_length)
        fused = torch.cat(
            [task_emb, obj_emb, torch.abs(task_emb - obj_emb), task_emb * obj_emb],
            dim=-1,
        )
        return self.head(fused).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Complete COCO-task inference pipeline: image -> YOLO visible objects -> "
            "MiniLM task/object scoring -> selected object."
        )
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--task", required=True, help='Task prompt, for example "serve wine".')
    parser.add_argument("--yolo-weights", default=str(DEFAULT_YOLO_WEIGHTS))
    parser.add_argument("--minilm-ckpt", default=str(DEFAULT_MINILM_CKPT))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--text-mode",
        choices=["finetuned", "similarity"],
        default="finetuned",
        help=(
            "finetuned uses minilm_task_text_best.pt. "
            "similarity uses original MiniLM embeddings without the fine-tuned head."
        ),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--yolo-conf", type=float, default=0.05)
    parser.add_argument("--none-threshold", type=float, default=0.10)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--rank-by",
        choices=["minilm", "product"],
        default="minilm",
        help="Use MiniLM score alone, or MiniLM score multiplied by YOLO confidence.",
    )
    parser.add_argument("--output-image", default=str(DEFAULT_OUTPUT_IMAGE))
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--no-image-output", action="store_true")
    parser.add_argument("--print-json", action="store_true", help="Print the full detailed JSON to console.")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def load_minilm_scorer(ckpt_path: Path, model_name: str, device: torch.device) -> MiniLMTaskObjectScorer:
    model = MiniLMTaskObjectScorer(model_name)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_original_minilm(model_name: str, device: torch.device) -> MiniLMTaskObjectScorer:
    model = MiniLMTaskObjectScorer(model_name)
    model.to(device)
    model.eval()
    return model


def run_yolo_detection(
    yolo_model: YOLO,
    image_path: Path,
    conf_threshold: float,
    max_det: int,
) -> List[Dict[str, Any]]:
    results = yolo_model(str(image_path), conf=conf_threshold, max_det=max_det, verbose=False)
    result = results[0]
    if result.boxes is None:
        return []

    detections: List[Dict[str, Any]] = []
    for det_idx, box in enumerate(result.boxes):
        cls_id = int(box.cls[0].item())
        detections.append(
            {
                "detection_id": det_idx,
                "bbox_xyxy": [float(v) for v in box.xyxy[0].tolist()],
                "class_id": cls_id,
                "class_name": result.names[cls_id],
                "yolo_conf": float(box.conf[0].item()),
            }
        )
    return detections


@torch.no_grad()
def score_detections(
    task_prompt: str,
    detections: List[Dict[str, Any]],
    model: MiniLMTaskObjectScorer,
    device: torch.device,
    rank_by: str,
) -> Optional[Dict[str, Any]]:
    if not detections:
        return None

    task_texts = [f"task: {task_prompt}"] * len(detections)
    object_texts = [f"object: {det['class_name']}" for det in detections]
    logits = model(task_texts, object_texts, device, MAX_LENGTH)
    scores = torch.sigmoid(logits).cpu().tolist()

    best_idx = 0
    best_rank = -1.0
    for idx, det in enumerate(detections):
        det["minilm_score"] = float(scores[idx])
        det["product_score"] = det["minilm_score"] * det["yolo_conf"]
        det["rank_score"] = det["product_score"] if rank_by == "product" else det["minilm_score"]
        if det["rank_score"] > best_rank:
            best_idx = idx
            best_rank = det["rank_score"]

    return {
        "selected_detection": detections[best_idx],
        "all_detections": detections,
    }


@torch.no_grad()
def score_detections_similarity(
    task_prompt: str,
    detections: List[Dict[str, Any]],
    model: MiniLMTaskObjectScorer,
    device: torch.device,
    rank_by: str,
) -> Optional[Dict[str, Any]]:
    if not detections:
        return None

    task_texts = [task_prompt] * len(detections)
    object_texts = [det["class_name"] for det in detections]
    task_emb = model.encode_texts(task_texts, device, MAX_LENGTH)
    obj_emb = model.encode_texts(object_texts, device, MAX_LENGTH)
    similarities = (task_emb * obj_emb).sum(dim=-1)
    scores = ((similarities + 1.0) / 2.0).cpu().tolist()

    best_idx = 0
    best_rank = -1.0
    for idx, det in enumerate(detections):
        det["minilm_score"] = float(scores[idx])
        det["product_score"] = det["minilm_score"] * det["yolo_conf"]
        det["rank_score"] = det["product_score"] if rank_by == "product" else det["minilm_score"]
        if det["rank_score"] > best_rank:
            best_idx = idx
            best_rank = det["rank_score"]

    return {
        "selected_detection": detections[best_idx],
        "all_detections": detections,
    }


def draw_pipeline_result(
    image_path: Path,
    selected_detection: Dict[str, Any],
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    x1, y1, x2, y2 = [int(v) for v in selected_detection["bbox_xyxy"]]
    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
    label = (
        f"{selected_detection['class_name']} "
        f"M={selected_detection['minilm_score']:.2f} "
        f"Y={selected_detection['yolo_conf']:.2f}"
    )
    draw.text((x1, max(0, y1 - 16)), label, fill="red")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def validate_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    image_path = Path(args.image)
    yolo_weights = Path(args.yolo_weights)
    minilm_ckpt = Path(args.minilm_ckpt)
    validate_path(image_path, "Image")
    validate_path(yolo_weights, "YOLO weights")
    if args.text_mode == "finetuned":
        validate_path(minilm_ckpt, "MiniLM checkpoint")

    device = choose_device(args.device)
    yolo_model = YOLO(str(yolo_weights))
    if args.text_mode == "finetuned":
        minilm_model = load_minilm_scorer(minilm_ckpt, args.model_name, device)
    else:
        minilm_model = load_original_minilm(args.model_name, device)

    detections = run_yolo_detection(yolo_model, image_path, args.yolo_conf, args.max_det)
    if args.text_mode == "finetuned":
        scored = score_detections(args.task, detections, minilm_model, device, args.rank_by)
    else:
        scored = score_detections_similarity(args.task, detections, minilm_model, device, args.rank_by)

    output: Dict[str, Any] = {
        "image": str(image_path),
        "task": args.task,
        "device": str(device),
        "rank_by": args.rank_by,
        "text_mode": args.text_mode,
        "yolo_weights": str(yolo_weights),
        "minilm_checkpoint": str(minilm_ckpt) if args.text_mode == "finetuned" else None,
        "minilm_model_name": args.model_name,
        "num_detections": len(detections),
        "selected_object": None,
        "selected_detection": None,
        "all_detections": detections,
    }

    if scored is None:
        output["status"] = "no_detections"
        return output

    selected = scored["selected_detection"]
    output["selected_object"] = selected["class_name"]
    output["selected_detection"] = selected
    output["all_detections"] = scored["all_detections"]
    output["status"] = "selected" if selected["rank_score"] >= args.none_threshold else "below_threshold"

    if not args.no_image_output and output["status"] == "selected":
        output_path = Path(args.output_image)
        draw_pipeline_result(image_path, selected, output_path)
        output["output_image"] = str(output_path)

    return output


def print_summary(output: Dict[str, Any]) -> None:
    detections = output.get("all_detections", [])
    selected = output.get("selected_detection")

    print("\nDetected objects and MiniLM scores:")
    if not detections:
        print("  None")
    else:
        for det in detections:
            print(
                f"  [{det['detection_id']}] {det['class_name']} | "
                f"MiniLM={det['minilm_score']:.4f} | "
                f"Rank={det['rank_score']:.4f} | "
                f"YOLO={det['yolo_conf']:.4f}"
            )

    if selected is None:
        print("\nFinal selected object: None")
    else:
        print(
            "\nFinal selected object: "
            f"{selected['class_name']} | "
            f"MiniLM={selected['minilm_score']:.4f} | "
            f"Rank={selected['rank_score']:.4f} | "
            f"YOLO={selected['yolo_conf']:.4f}"
        )


def main() -> None:
    args = parse_args()
    output = run_pipeline(args)
    text = json.dumps(output, indent=2)
    if args.print_json:
        print(text)
    print_summary(output)

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
