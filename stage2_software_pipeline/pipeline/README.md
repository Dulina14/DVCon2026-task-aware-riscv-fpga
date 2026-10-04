# Task-Aware Object Detection Pipeline

This project implements a task-aware object selection pipeline. Given an image and a natural-language task prompt, YOLOv8x detects candidate objects and a fine-tuned MiniLM model ranks the detected object classes against the task. The final output is the most suitable object, its bounding box, and confidence/relevance scores.

## 1. Required Files

Required for single-image and CSV batch inference:

```text
task-aware-object-detection-pipeline/
  README.md
  requirements-colab.txt
  inference.py
  run_pipeline_interactive.py
  batch_inference.py
  yolov8x.pt
  checkpoints/
    finetuned_text/
      minilm_task_text_best.pt
```

Optional files for reproducing COCO-Tasks evaluation:

```text
  evaluate_pipeline.py
  batch_test_1000_pipeline.py
  prepare_yolo_coco_tasks.py
  derived/
    coco_tasks_test.jsonl
```

Optional evidence files:

```text
  results/
    result.jpg
    batch_1000_pipeline_summary.json
    batch_1000_pipeline_results.csv
    batch_1000_pipeline_results.jsonl
```

The submitted project does not include the MSCOCO image dataset. COCO-Tasks evaluation can only be rerun if the evaluator already has COCO `train2014/` and `val2014/` images locally. Normal inference works with any image supplied by the evaluator.

## 2. Included 1000-Sample Results

The `results/` folder includes the output from our balanced 1000-sample COCO-Tasks pipeline evaluation:

```text
results/batch_1000_pipeline_summary.json
results/batch_1000_pipeline_results.csv
results/batch_1000_pipeline_results.jsonl
```

Use these files to inspect our reported performance without rerunning the COCO evaluation.

```text
batch_1000_pipeline_summary.json    aggregate accuracy and per-task metrics
batch_1000_pipeline_results.csv     spreadsheet-friendly row-by-row results
batch_1000_pipeline_results.jsonl   full row-by-row output with top detections
```

The summary file is the main analysis file. Important reported values include overall accuracy, positive top-1 accuracy, negative rejection accuracy, per-task accuracy, selected object counts, and expected-object availability from YOLO detections.

## 3. Install Dependencies

```powershell
pip install torch torchvision torchaudio
pip install -r requirements-colab.txt
pip install pillow
```

The first run may download the base MiniLM model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

## 4. Single-Image Inference

### Interactive Mode

Run:

```powershell
python run_pipeline_interactive.py
```

Enter an image path and task prompt when asked:

```text
Enter image path: D:\path\to\image.jpg
Enter task prompt, for example "serve wine": serve wine
```

Outputs are written to:

```text
results/result.jpg
results/pipeline_result.json
```

### Command-Line Mode

```powershell
python inference.py `
  --image "D:\path\to\image.jpg" `
  --task "serve wine" `
  --output-image "results\result.jpg" `
  --output-json "results\pipeline_result.json"
```

Other examples:

```powershell
python inference.py --image "D:\path\to\image.jpg" --task "open parcel" --output-image "results\parcel.jpg" --output-json "results\parcel.json"
python inference.py --image "D:\path\to\image.jpg" --task "water plant" --device cpu
python inference.py --image "D:\path\to\image.jpg" --task "sit comfortably" --print-json
```

Useful options:

```text
--device auto|cpu|cuda       default: auto
--yolo-conf 0.05             YOLO detection confidence threshold
--none-threshold 0.10        minimum score required to select an object
--rank-by minilm|product     product multiplies MiniLM score by YOLO confidence
--no-image-output            skip annotated image generation
```

## 5. Batch Inference on Custom Inputs

Use `batch_inference.py` when evaluators want to test many image/task pairs without COCO labels.

Create a CSV file, for example `batch_inputs.csv`:

```csv
image_path,task_prompt
D:\images\image_001.jpg,serve wine
D:\images\image_002.jpg,open parcel
D:\images\image_003.jpg,water plant
```

Run:

```powershell
python batch_inference.py `
  --csv "batch_inputs.csv" `
  --output-dir "results\batch_inference"
```

Outputs:

```text
results/batch_inference/batch_summary.csv   one summary row per input
results/batch_inference/images/             annotated selected-object images
results/batch_inference/json/               full JSON result for each input
```

`batch_summary.csv` contains:

```text
image_path
task_prompt
status
selected_object
rank_score
minilm_score
yolo_conf
num_detections
output_image
output_json
```

This batch mode is for demonstration/inference only. It does not compute accuracy because it does not require ground-truth labels.

## 6. Understanding Inference Results

Each JSON output contains:

```text
selected_object       final selected object class, or null
selected_detection    selected object's bbox and scores
all_detections        all YOLO detections after MiniLM scoring
status                selected, below_threshold, or no_detections
```

Important score fields:

```text
bbox_xyxy       [x1, y1, x2, y2] selected bounding box
yolo_conf       YOLO object detection confidence
minilm_score    semantic relevance between task prompt and object class
rank_score      score used for final ranking
```

Status meaning:

```text
selected          an object passed the threshold and was selected
below_threshold   objects were detected, but no object was relevant enough
no_detections     YOLO did not detect any candidate object
```

## 7. COCO-Tasks Evaluation

This section is optional. It requires both:

```text
derived/coco_tasks_test.jsonl
local COCO images arranged as train2014/ and val2014/
```

The JSONL file contains task labels and image paths, but it does not contain the image files. If the paths inside the JSONL do not exist on the evaluator's machine, use `--image-root` to point to the local COCO dataset root:

```text
D:\datasets\mscoco\
  train2014\
  val2014\
```

### Quick Evaluation

Runs the pipeline on the first N JSONL rows and writes a compact JSON report:

```powershell
python evaluate_pipeline.py `
  --jsonl "derived\coco_tasks_test.jsonl" `
  --image-root "D:\datasets\mscoco" `
  --max-samples 100 `
  --output-json "results\pipeline_eval_results.json"
```

Output:

```text
results/pipeline_eval_results.json
```

### Balanced 1000-Sample Evaluation

Purpose of `batch_test_1000_pipeline.py`:

```text
Runs the complete YOLOv8x + MiniLM pipeline on a balanced sample of COCO-Tasks rows.
Samples approximately evenly across the task prompts.
Evaluates positive rows where a preferred object exists.
Evaluates negative rows where the correct behavior is to reject the scene.
Writes aggregate metrics and row-level results.
```

Run:

```powershell
python batch_test_1000_pipeline.py `
  --jsonl "derived\coco_tasks_test.jsonl" `
  --image-root "D:\datasets\mscoco" `
  --num-samples 1000 `
  --output-jsonl "results\batch_1000_pipeline_results.jsonl" `
  --output-csv "results\batch_1000_pipeline_results.csv" `
  --output-summary "results\batch_1000_pipeline_summary.json"
```

Outputs:

```text
results/batch_1000_pipeline_summary.json    aggregate metrics
results/batch_1000_pipeline_results.csv     spreadsheet-friendly row results
results/batch_1000_pipeline_results.jsonl   full row-level detections and top candidates
```

If the evaluator does not have COCO images, skip this section and refer to the included files under `results/`.

## 8. Evaluation Metrics

Important fields in `batch_1000_pipeline_summary.json`:

```text
total                                  number of evaluated rows
selected                               rows where an object was selected
below_threshold                        rows rejected by the MiniLM threshold
no_detections                          rows with no YOLO detections
positive_total                         rows where a preferred object exists
positive_top1_correct                  selected object matched a preferred object
positive_top1_accuracy                 positive_top1_correct / positive_total
positive_expected_available_by_yolo    positive rows where YOLO detected a preferred class
positive_top1_when_expected_detected   accuracy when the expected class was detected by YOLO
positive_expected_in_top5_when_detected top-5 candidate availability
negative_total                         rows where no preferred object exists
negative_rejected                      negative rows correctly rejected
negative_rejection_accuracy            negative_rejected / negative_total
overall_accuracy                       combined positive selection and negative rejection accuracy
per_task                               metrics split by task prompt
selected_object_counts                 frequency of selected object classes
```

Use `positive_top1_accuracy` to measure object selection quality. Use `negative_rejection_accuracy` to measure how well unsuitable scenes are rejected. Use `positive_expected_available_by_yolo` to separate YOLO detection recall from MiniLM ranking behavior.

## 9. Default Configuration

```text
YOLO weights: yolov8x.pt
MiniLM checkpoint: checkpoints/finetuned_text/minilm_task_text_best.pt
YOLO confidence threshold: 0.05
None/rejection threshold: 0.10
Ranking mode: minilm
Device: auto
```

The low YOLO threshold keeps small or low-confidence task-relevant objects available for MiniLM ranking.
