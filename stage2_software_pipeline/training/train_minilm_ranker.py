import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_JSONL = os.getenv(
    "TRAIN_JSONL",
    str(PROJECT_DIR / "derived" / "corrected_coco_tasks_train.jsonl"),
)
VAL_JSONL = os.getenv(
    "VAL_JSONL",
    os.getenv("TEST_JSONL", str(PROJECT_DIR / "derived" / "corrected_coco_tasks_test.jsonl")),
)
SAVE_DIR = os.getenv("SAVE_DIR", str(PROJECT_DIR / "checkpoints" / "finetuned_text"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
GRAD_ACCUM_STEPS = int(os.getenv("GRAD_ACCUM_STEPS", "4"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "20"))
LR = float(os.getenv("LR", "2e-5"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "32"))
DROPOUT = float(os.getenv("DROPOUT", "0.1"))
FREEZE_ENCODER = os.getenv("FREEZE_ENCODER", "0") == "1"
USE_AMP = os.getenv("USE_AMP", "1") == "1"
MAX_TRAIN_GROUPS = int(os.getenv("MAX_TRAIN_GROUPS", "0"))
MAX_VAL_GROUPS = int(os.getenv("MAX_VAL_GROUPS", "0"))
RANK_LOSS_WEIGHT = float(os.getenv("RANK_LOSS_WEIGHT", "1.0"))
MARGIN = float(os.getenv("MARGIN", "0.25"))


COCO_CAT_ID_TO_NAME = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}


class CocoTaskRankingDataset(Dataset):
    """
    One sample is one task/image candidate set.

    The inference model only sees object class text, not individual boxes. If a row has
    repeated instances of the same COCO class, we collapse them and mark the class
    positive if any instance is preferred.
    """

    def __init__(self, jsonl_path: str):
        self.groups: List[Dict[str, Any]] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                labels_by_name: Dict[str, int] = {}
                for obj in entry["objects"]:
                    coco_id = obj["coco_category_id"]
                    object_name = COCO_CAT_ID_TO_NAME.get(coco_id, f"unknown_{coco_id}")
                    labels_by_name[object_name] = max(
                        labels_by_name.get(object_name, 0),
                        int(obj.get("preferred", 0)),
                    )

                if not labels_by_name:
                    continue

                object_names = sorted(labels_by_name)
                self.groups.append(
                    {
                        "task_text": f"task: {entry['task_prompt']}",
                        "object_texts": [f"object: {name}" for name in object_names],
                        "labels": [float(labels_by_name[name]) for name in object_names],
                    }
                )

        print(f"Loaded {len(self.groups)} task/object groups from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.groups[idx]


def collate_groups(batch: List[Dict[str, Any]]) -> Tuple[List[str], List[str], torch.Tensor, List[int]]:
    task_texts: List[str] = []
    object_texts: List[str] = []
    labels: List[float] = []
    group_sizes: List[int] = []

    for group in batch:
        size = len(group["object_texts"])
        task_texts.extend([group["task_text"]] * size)
        object_texts.extend(group["object_texts"])
        labels.extend(group["labels"])
        group_sizes.append(size)

    return task_texts, object_texts, torch.tensor(labels, dtype=torch.float32), group_sizes


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


class MiniLMTaskObjectScorer(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1, freeze_encoder: bool = False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

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


def ranking_loss(logits: torch.Tensor, labels: torch.Tensor, group_sizes: List[int]) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    start = 0
    for size in group_sizes:
        group_logits = logits[start : start + size]
        group_labels = labels[start : start + size]
        pos = group_logits[group_labels > 0.5]
        neg = group_logits[group_labels <= 0.5]
        if pos.numel() > 0 and neg.numel() > 0:
            losses.append(F.relu(MARGIN - pos[:, None] + neg[None, :]).mean())
        start += size

    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def compute_group_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: List[int],
) -> Tuple[int, int, int, int]:
    top1_correct = 0
    top1_total = 0
    all_negative_correct = 0
    all_negative_total = 0

    start = 0
    for size in group_sizes:
        group_logits = logits[start : start + size]
        group_labels = labels[start : start + size]
        has_positive = bool((group_labels > 0.5).any().item())

        if has_positive:
            best_idx = int(torch.argmax(group_logits).item())
            top1_correct += int(group_labels[best_idx].item() > 0.5)
            top1_total += 1
        else:
            all_negative_correct += int((torch.sigmoid(group_logits) <= 0.5).all().item())
            all_negative_total += 1

        start += size

    return top1_correct, top1_total, all_negative_correct, all_negative_total


def run_epoch(
    model: MiniLMTaskObjectScorer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    train: bool,
    epoch: int,
) -> Dict[str, float]:
    model.train(train)
    total_loss = 0.0
    total_bce = 0.0
    total_rank = 0.0
    total_objects = 0
    top1_correct = 0
    top1_total = 0
    all_neg_correct = 0
    all_neg_total = 0

    desc = f"Epoch {epoch}" if train else "Eval"
    pbar = tqdm(loader, desc=desc, leave=not train)
    if train:
        optimizer.zero_grad(set_to_none=True)

    for batch_idx, (task_texts, object_texts, labels, group_sizes) in enumerate(pbar, start=1):
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(task_texts, object_texts, device, MAX_LENGTH)
                bce = criterion(logits, labels)
                rank = ranking_loss(logits, labels, group_sizes)
                loss = bce + RANK_LOSS_WEIGHT * rank

            if train:
                scaler.scale(loss / GRAD_ACCUM_STEPS).backward()
                if batch_idx % GRAD_ACCUM_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        batch_objects = labels.numel()
        total_loss += float(loss.detach().item()) * batch_objects
        total_bce += float(bce.detach().item()) * batch_objects
        total_rank += float(rank.detach().item()) * batch_objects
        total_objects += batch_objects

        c, t, nc, nt = compute_group_metrics(logits.detach(), labels, group_sizes)
        top1_correct += c
        top1_total += t
        all_neg_correct += nc
        all_neg_total += nt

        pbar.set_postfix(
            loss=f"{total_loss / max(total_objects, 1):.4f}",
            top1=f"{100.0 * top1_correct / max(top1_total, 1):.2f}%",
        )

    if train and len(loader) % GRAD_ACCUM_STEPS != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return {
        "loss": total_loss / max(total_objects, 1),
        "bce_loss": total_bce / max(total_objects, 1),
        "rank_loss": total_rank / max(total_objects, 1),
        "top1_acc": top1_correct / max(top1_total, 1),
        "top1_total": float(top1_total),
        "all_negative_acc": all_neg_correct / max(all_neg_total, 1),
        "all_negative_total": float(all_neg_total),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA runtime:", torch.version.cuda)
    else:
        print("CUDA is not available. Training will be slow on CPU.")

    print("Train JSONL:", TRAIN_JSONL)
    print("Val JSONL:", VAL_JSONL)
    print("Save dir:", SAVE_DIR)
    print("Epochs:", NUM_EPOCHS)
    print("Batch size in groups:", BATCH_SIZE)
    print("Effective group batch size:", BATCH_SIZE * GRAD_ACCUM_STEPS)

    train_dataset = CocoTaskRankingDataset(TRAIN_JSONL)
    val_dataset = CocoTaskRankingDataset(VAL_JSONL)

    if MAX_TRAIN_GROUPS > 0:
        train_dataset.groups = train_dataset.groups[:MAX_TRAIN_GROUPS]
        print("Using train group limit:", len(train_dataset.groups))
    if MAX_VAL_GROUPS > 0:
        val_dataset.groups = val_dataset.groups[:MAX_VAL_GROUPS]
        print("Using val group limit:", len(val_dataset.groups))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_groups,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_groups,
    )

    positive_count = sum(sum(group["labels"]) for group in train_dataset.groups)
    object_count = sum(len(group["labels"]) for group in train_dataset.groups)
    negative_count = object_count - positive_count
    pos_weight = torch.tensor([negative_count / max(positive_count, 1.0)], device=device)

    print("Candidate classes:", object_count)
    print("Positive candidate classes:", int(positive_count))
    print("Negative candidate classes:", int(negative_count))
    print("pos_weight:", float(pos_weight.item()))

    model = MiniLMTaskObjectScorer(
        model_name=MODEL_NAME,
        dropout=DROPOUT,
        freeze_encoder=FREEZE_ENCODER,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    use_amp = device.type == "cuda" and USE_AMP
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_top1 = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_metrics = run_epoch(
            model, train_loader, optimizer, criterion, device, scaler, use_amp, True, epoch
        )
        val_metrics = run_epoch(
            model, val_loader, optimizer, criterion, device, scaler, use_amp, False, epoch
        )

        print(
            f"Epoch {epoch} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_top1={train_metrics['top1_acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_top1={val_metrics['top1_acc']:.4f} "
            f"val_all_negative={val_metrics['all_negative_acc']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_name": MODEL_NAME,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["top1_acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["top1_acc"],
        }

        last_path = os.path.join(SAVE_DIR, "minilm_task_text_last.pt")
        torch.save(checkpoint, last_path)

        if val_metrics["top1_acc"] > best_top1:
            best_top1 = val_metrics["top1_acc"]
            best_path = os.path.join(SAVE_DIR, "minilm_task_text_best.pt")
            torch.save(checkpoint, best_path)
            print("Saved best checkpoint:", best_path)

    print("Training complete.")


if __name__ == "__main__":
    main()
