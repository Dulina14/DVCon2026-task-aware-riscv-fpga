import os, json, glob
from pathlib import Path

ROOT = Path(r"D:\DATASETS")
COCO_ROOT = ROOT / "mscoco"
COCOTASKS_ROOT = ROOT / "coco-tasks"
OUT_DIR = ROOT / "derived"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# REAL task prompts from the COCO-Tasks paper
TASK_PROMPTS = {
    1: "step on something to reach top of a shelf",
    2: "sit comfortably",
    3: "place flowers",
    4: "get potatoes out of fire",
    5: "water plant",
    6: "get lemon out of tea",
    7: "dig hole",
    8: "open bottle of beer",
    9: "open parcel",
    10: "serve wine",
    11: "pour sugar",
    12: "smear butter",
    13: "extinguish fire",
    14: "pound carpet",
}

def build_coco_image_map():
    """image_id -> (file_name, split) where split in {'train2014','val2014'}"""
    img_map = {}
    for split in ["train2014", "val2014"]:
        p = COCO_ROOT / "annotations" / f"instances_{split}.json"
        with open(p, "r") as f:
            d = json.load(f)
        for im in d["images"]:
            img_map[im["id"]] = (im["file_name"], split)
    return img_map

def write_jsonl(task_json_path, out_path, img_map, append=True):
    task_json_path = Path(task_json_path)
    task_num = int(task_json_path.stem.split("_")[1])  # task_1_train.json -> 1
    split = task_json_path.stem.split("_")[2]          # train/test

    with open(task_json_path, "r") as f:
        d = json.load(f)

    # group annotations by image_id
    by_image = {}
    for ann in d["annotations"]:
        image_id = ann["image_id"]
        by_image.setdefault(image_id, []).append(ann)

    mode = "a" if append else "w"
    with open(out_path, mode, encoding="utf-8") as out:
        for image_id, anns in by_image.items():
            if image_id not in img_map:
                # if this happens, your COCO instances jsons don't match the ids used
                continue

            file_name, coco_split = img_map[image_id]
            image_path = str(COCO_ROOT / coco_split / file_name)

            objects = []
            for ann in anns:
                objects.append({
                    "ann_id": ann["id"],
                    "bbox_xywh": ann["bbox"],
                    "coco_category_id": ann["COCO_category_id"],
                    "preferred": 1 if int(ann["category_id"]) == 1 else 0,
                    "area": ann.get("area", None),
                    "iscrowd": ann.get("iscrowd", 0),
                })

            record = {
                "image_id": image_id,
                "image_path": image_path,
                "task_id": task_num,
                "task_prompt": TASK_PROMPTS.get(task_num, f"task {task_num}"),
                "split": split,
                "objects": objects
            }
            out.write(json.dumps(record) + "\n")

def main():
    img_map = build_coco_image_map()

    train_out = OUT_DIR / "coco_tasks_train.jsonl"
    test_out  = OUT_DIR / "coco_tasks_test.jsonl"

    # Remove existing files to start fresh
    if train_out.exists(): train_out.unlink()
    if test_out.exists(): test_out.unlink()

    # Process train files - first file writes (w), rest append (a)
    train_files = sorted((COCOTASKS_ROOT / "annotations").glob("task_*_train.json"))
    for i, p in enumerate(train_files):
        append = (i > 0)
        write_jsonl(p, train_out, img_map, append=append)
        print("wrote train from", p.name)

    # Process test files - first file writes (w), rest append (a)
    test_files = sorted((COCOTASKS_ROOT / "annotations").glob("task_*_test.json"))
    for i, p in enumerate(test_files):
        append = (i > 0)
        write_jsonl(p, test_out, img_map, append=append)
        print("wrote test from", p.name)

    print("DONE")
    print("train:", train_out)
    print("test :", test_out)

if __name__ == "__main__":
    main()