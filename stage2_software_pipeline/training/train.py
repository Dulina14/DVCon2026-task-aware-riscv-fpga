import os
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

PROJECT_DIR = Path(__file__).resolve().parent

TRAIN_JSONL = os.getenv("TRAIN_JSONL", str(PROJECT_DIR / "derived" / "coco_tasks_train.jsonl"))
TEST_JSONL = os.getenv("TEST_JSONL", str(PROJECT_DIR / "derived" / "coco_tasks_test.jsonl"))

SAVE_DIR = os.getenv("SAVE_DIR", str(PROJECT_DIR / "checkpoints" / "finetuned_text"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
GRAD_ACCUM_STEPS = int(os.getenv("GRAD_ACCUM_STEPS", "16"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "3"))
LR = float(os.getenv("LR", "2e-5"))
WEIGHT_DECAY = 0.01
MAX_LENGTH = 32
DROPOUT = 0.1
FREEZE_ENCODER = False
USE_AMP = os.getenv("USE_AMP", "1") == "1"
MAX_TRAIN_SAMPLES = int(os.getenv("MAX_TRAIN_SAMPLES", "0"))
MAX_TEST_SAMPLES = int(os.getenv("MAX_TEST_SAMPLES", "0"))

# =========================================================
# ORIGINAL COCO CATEGORY ID -> NAME
# (matches your jsonl's coco_category_id field style)
# =========================================================

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


class CocoTasksTextDataset(Dataset):
    """
    One sample per object:
      task_text + object_name_text -> preferred label
    """
    def __init__(self, jsonl_path: str):
        self.samples: List[Dict[str, Any]] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                task_text = entry["task_prompt"]

                for obj in entry["objects"]:
                    coco_id = obj["coco_category_id"]
                    object_name = COCO_CAT_ID_TO_NAME.get(coco_id, f"unknown_{coco_id}")
                    preferred = int(obj["preferred"])

                    # simple text templates
                    object_text = f"object: {object_name}"
                    task_text_fmt = f"task: {task_text}"

                    self.samples.append(
                        {
                            "task_text": task_text_fmt,
                            "object_text": object_text,
                            "label": preferred,
                        }
                    )

        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_fn(batch):
    task_texts = [x["task_text"] for x in batch]
    object_texts = [x["object_text"] for x in batch]
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.float32)
    return task_texts, object_texts, labels


def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


class MiniLMTaskObjectScorer(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1, freeze_encoder: bool = False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        hidden_size = self.encoder.config.hidden_size  # 384 for all-MiniLM-L6-v2

        self.head = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def encode_texts(self, texts: List[str], device, max_length: int):
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        outputs = self.encoder(**tokens)
        emb = mean_pooling(outputs.last_hidden_state, tokens["attention_mask"])
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    def forward(self, task_texts: List[str], object_texts: List[str], device, max_length: int):
        task_emb = self.encode_texts(task_texts, device, max_length)
        obj_emb = self.encode_texts(object_texts, device, max_length)

        fused = torch.cat(
            [
                task_emb,
                obj_emb,
                torch.abs(task_emb - obj_emb),
                task_emb * obj_emb,
            ],
            dim=-1,
        )
        logits = self.head(fused).squeeze(-1)
        return logits


def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for task_texts, object_texts, labels in tqdm(loader, desc="Eval", leave=False):
            labels = labels.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(task_texts, object_texts, device, MAX_LENGTH)
                loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)

    return total_loss / total_count, total_correct / total_count


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, scaler, use_amp):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    optimizer.zero_grad(set_to_none=True)
    for batch_idx, (task_texts, object_texts, labels) in enumerate(pbar, start=1):
        labels = labels.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(task_texts, object_texts, device, MAX_LENGTH)
            loss = criterion(logits, labels)
            step_loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(step_loss).backward()

        if batch_idx % GRAD_ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

        if total_count > 0:
            pbar.set_postfix(
                loss=f"{total_loss / total_count:.4f}",
                acc=f"{100.0 * total_correct / total_count:.2f}%"
            )

    if len(loader) % GRAD_ACCUM_STEPS != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / total_count, total_correct / total_count


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA runtime:", torch.version.cuda)
        print("Batch size:", BATCH_SIZE)
        print("Gradient accumulation steps:", GRAD_ACCUM_STEPS)
        print("Effective batch size:", BATCH_SIZE * GRAD_ACCUM_STEPS)
        print("AMP:", USE_AMP)
    else:
        print("CUDA is not available. Install a CUDA-enabled PyTorch build to train on the laptop GPU.")

    train_dataset = CocoTasksTextDataset(TRAIN_JSONL)
    test_dataset = CocoTasksTextDataset(TEST_JSONL)
    if MAX_TRAIN_SAMPLES > 0:
        train_dataset.samples = train_dataset.samples[:MAX_TRAIN_SAMPLES]
        print("Using train sample limit:", len(train_dataset.samples))
    if MAX_TEST_SAMPLES > 0:
        test_dataset.samples = test_dataset.samples[:MAX_TEST_SAMPLES]
        print("Using test sample limit:", len(test_dataset.samples))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = MiniLMTaskObjectScorer(
        model_name=MODEL_NAME,
        dropout=DROPOUT,
        freeze_encoder=FREEZE_ENCODER,
    ).to(device)

    # class imbalance handling
    pos_count = sum(s["label"] for s in train_dataset.samples)
    neg_count = len(train_dataset.samples) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32, device=device)

    print("Positives:", pos_count)
    print("Negatives:", neg_count)
    print("pos_weight:", pos_weight.item())

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    use_amp = device.type == "cuda" and USE_AMP
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_val_acc = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, scaler, use_amp
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device, use_amp)

        print(
            f"Epoch {epoch} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        last_path = os.path.join(SAVE_DIR, "minilm_task_text_last.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_name": MODEL_NAME,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            last_path,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(SAVE_DIR, "minilm_task_text_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": MODEL_NAME,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                best_path,
            )
            print("Saved best checkpoint:", best_path)

    print("Training complete.")


if __name__ == "__main__":
    main()
