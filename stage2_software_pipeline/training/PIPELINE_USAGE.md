# COCO-Task YOLO + MiniLM Pipeline

This repository now has the complete inference flow in `inference.py`:

```text
image
  -> YOLO detects visible COCO objects
  -> MiniLM scores task prompt + detected object name
  -> highest MiniLM score is selected
```

## Trained Artifacts

Default paths used by `inference.py`:

```text
yolov8x.pt
checkpoints/finetuned_text/minilm_task_text_best.pt
```

## Run One Image

For an interactive top-level runner:

```powershell
python run_pipeline_interactive.py
```

It prompts for:

```text
Enter image path:
Enter task prompt, for example "serve wine":
```

It saves the annotated image to `result.jpg` and the JSON output to `pipeline_result.json`.

```powershell
python inference.py `
  --image "D:\DATASETS\mscoco\val2014\COCO_val2014_000000472375.jpg" `
  --task "serve wine" `
  --output-image result.jpg `
  --output-json pipeline_result.json
```

`inference.py` now defaults to the YOLOv8x detector at:

```text
yolov8x.pt
```

The default YOLO confidence is `0.05` so small task-relevant objects remain available for MiniLM ranking.

The command prints JSON with:

```text
selected_object
selected_detection
all_detections
```

By default, selection uses the MiniLM score only. To combine MiniLM preference with YOLO detection confidence:

```powershell
python inference.py --image "path\to\image.jpg" --task "serve wine" --rank-by product
```

## Useful Options

```text
--yolo-conf 0.25          YOLO confidence threshold
--none-threshold 0.10     minimum selected score required for status=selected
--no-image-output         skip annotated result image
--device cpu             force CPU
--device cuda            force CUDA
```

To make detection stricter, raise YOLO confidence:

```powershell
python inference.py `
  --image "D:\DATASETS\mscoco\val2014\COCO_val2014_000000472375.jpg" `
  --task "serve wine" `
  --yolo-conf 0.25
```

## Evaluate The Full Pipeline

Run a bounded check on the test JSONL:

```powershell
python evaluate_pipeline.py `
  --jsonl derived\coco_tasks_test.jsonl `
  --yolo-conf 0.05 `
  --max-samples 100 `
  --output-json pipeline_eval_sample100.json
```

For JSONL files whose `image_path` values point at a different machine, add:

```powershell
--image-root "D:\DATASETS\mscoco"
```

## Training Inputs

MiniLM training uses:

```text
derived/coco_tasks_train.jsonl
derived/coco_tasks_test.jsonl
```

Recommended MiniLM training for object selection:

```text
colab_train_minilm.ipynb
train_minilm_ranker.py
```

The ranker trains on each task/image candidate set and optimizes validation top-1 accuracy: the highest-scored object class should be one of the preferred objects for that prompt. This matches the inference behavior better than raw binary object-label accuracy on the imbalanced labels.

YOLO training data can be created from those JSONL files with `prepare_yolo_coco_tasks.py`, then trained with Ultralytics using the generated `data.yaml`.
