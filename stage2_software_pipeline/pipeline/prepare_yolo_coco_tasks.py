import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path, PureWindowsPath

from PIL import Image


COCO_ID_TO_YOLO_ID = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    13: 11,
    14: 12,
    15: 13,
    16: 14,
    17: 15,
    18: 16,
    19: 17,
    20: 18,
    21: 19,
    22: 20,
    23: 21,
    24: 22,
    25: 23,
    27: 24,
    28: 25,
    31: 26,
    32: 27,
    33: 28,
    34: 29,
    35: 30,
    36: 31,
    37: 32,
    38: 33,
    39: 34,
    40: 35,
    41: 36,
    42: 37,
    43: 38,
    44: 39,
    46: 40,
    47: 41,
    48: 42,
    49: 43,
    50: 44,
    51: 45,
    52: 46,
    53: 47,
    54: 48,
    55: 49,
    56: 50,
    57: 51,
    58: 52,
    59: 53,
    60: 54,
    61: 55,
    62: 56,
    63: 57,
    64: 58,
    65: 59,
    67: 60,
    70: 61,
    72: 62,
    73: 63,
    74: 64,
    75: 65,
    76: 66,
    77: 67,
    78: 68,
    79: 69,
    80: 70,
    81: 71,
    82: 72,
    84: 73,
    85: 74,
    86: 75,
    87: 76,
    88: 77,
    89: 78,
    90: 79,
}


YOLO_COCO_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert COCO-task JSONL files into one YOLO COCO-object dataset."
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument(
        "--image-root",
        required=True,
        help="Folder containing train2014 and val2014 image folders.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--image-mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="Use symlinks for speed/storage or copy images into the YOLO dataset.",
    )
    return parser.parse_args()


def source_image_path(entry, image_root):
    raw_image_path = str(entry["image_path"])
    source_name = PureWindowsPath(raw_image_path).name
    if source_name == raw_image_path:
        source_name = Path(raw_image_path).name
    split = entry.get("split", "")
    image_subdir = "train2014" if split == "train" else "val2014"
    return image_root / image_subdir / source_name


def link_or_copy_image(src, dst, mode):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def normalized_yolo_box(bbox_xywh, width, height):
    x, y, w, h = bbox_xywh
    x = max(0.0, min(float(x), float(width)))
    y = max(0.0, min(float(y), float(height)))
    w = max(0.0, min(float(w), float(width) - x))
    h = max(0.0, min(float(h), float(height) - y))
    if w <= 0.0 or h <= 0.0:
        return None
    return ((x + w / 2.0) / width, (y + h / 2.0) / height, w / width, h / height)


def collect_split(jsonl_path, split_name, image_root, out_dir, image_mode):
    labels_by_image = defaultdict(set)
    missing_images = 0
    skipped_boxes = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            src_img = source_image_path(entry, image_root)
            if not src_img.exists():
                missing_images += 1
                continue

            dst_img = out_dir / "images" / split_name / src_img.name
            link_or_copy_image(src_img, dst_img, image_mode)

            with Image.open(src_img) as image:
                width, height = image.size

            for obj in entry["objects"]:
                yolo_id = COCO_ID_TO_YOLO_ID.get(int(obj["coco_category_id"]))
                if yolo_id is None:
                    skipped_boxes += 1
                    continue
                box = normalized_yolo_box(obj["bbox_xywh"], width, height)
                if box is None:
                    skipped_boxes += 1
                    continue
                labels_by_image[dst_img.name].add(
                    (yolo_id, round(box[0], 6), round(box[1], 6), round(box[2], 6), round(box[3], 6))
                )

    labels_dir = out_dir / "labels" / split_name
    labels_dir.mkdir(parents=True, exist_ok=True)
    for image_name, labels in labels_by_image.items():
        label_path = labels_dir / f"{Path(image_name).stem}.txt"
        label_lines = [
            f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
            for cls_id, xc, yc, w, h in sorted(labels)
        ]
        label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    print(
        f"{split_name}: {len(labels_by_image)} images, "
        f"{sum(len(v) for v in labels_by_image.values())} boxes, "
        f"{missing_images} missing image rows, {skipped_boxes} skipped boxes"
    )


def write_data_yaml(out_dir):
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(YOLO_COCO_NAMES))
    yaml_text = (
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"{names}\n"
    )
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print("Wrote:", out_dir / "data.yaml")


def main():
    args = parse_args()
    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    collect_split(Path(args.train_jsonl), "train", image_root, out_dir, args.image_mode)
    collect_split(Path(args.val_jsonl), "val", image_root, out_dir, args.image_mode)
    write_data_yaml(out_dir)


if __name__ == "__main__":
    main()
