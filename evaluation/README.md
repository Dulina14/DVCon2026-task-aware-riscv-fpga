# evaluation/

Accuracy measurement with the **reference paper's protocol**: average precision at IoU 0.5 against the preferred-object annotations, per task, averaged over the 14 tasks (mAP@0.5), plus top-1. The suitability scores are read from the **deployed INT8 package**, so the numbers reflect what the board computes.

| File | What it is |
|---|---|
| `eval_coco_tasks.py` | The evaluator. Box sources: `gt` (ground truth), `faster` (the paper's Faster R-CNN detections), `yolo_paper`, `yolov8n` (Ultralytics), `detfile` (detections from our C program's `--detect-json`). Ranking: `product` / `minilm` / `det`. `--method pickbest` reproduces the paper's baseline as a sanity check |
| `make_image_list.py` | Writes `<id> <path>` lists of the COCO-Tasks test images for batch detection with the C program |
| `scripts/` | Resumable sweeps that run the C program (`pipeline_x86 --sim`) over image lists in chunks (`sweep.sh`, `step*.sh`, `steps.sh`, `step256.sh`), a one-image-per-task run (`sim14.sh`), and `ppm_to_png.py` |
| `EVAL_RESULTS.md` | Ranking-stage results and how they compare with the paper |
| `results/minilm_task_class_scores.json` | 14 × 80 table of MiniLM suitability scores dumped from the INT8 package (`--dump-scores`) |
| `results/eval_*.json` | Saved outputs of evaluation runs |

## Main results

| System | top-1 | mAP@0.5 |
|---|---|---|
| Sawatzky et al. 2019 — Faster R-CNN + GGNN (GPU) | 0.640 | 0.409 |
| **Ours — YOLOv8s INT8, 256 px (board build)** | **0.627** | **0.425** |
| Ours — YOLOv8n INT8, 192 px | 0.311 | 0.201 |

Ranking stage alone:

| Boxes given to the ranker | mAP@0.5 |
|---|---|
| Ground-truth boxes | **0.648** (top-1 94.7 %) |
| The paper's own Faster R-CNN detections | **0.288** — equal to the paper's classification baseline (0.288), 88 % of its full GGNN (0.326/0.327) |

**Where the errors are:** of the top-1 misses, the detector accounts for ~72 points and the ranker ~8. Resolution is the lever.

**How sure are we?** Re-implementing the paper's pick-best baseline gave 0.339 vs their published 0.386, so absolute figures carry about **±0.05**. Comparisons made with the same code on the same boxes are firmer. Tie order adds ±0.003.

> `docs/reports/ACCURACY_FINDINGS.html` is an earlier (23 Aug) analysis. Some of its figures (e.g. ground-truth mAP 0.756, 0.362 vs 0.394 on Faster R-CNN boxes) come from a different scoring set-up and were superseded by the numbers above and in [`EVAL_RESULTS.md`](EVAL_RESULTS.md) / [TECHNICAL_REPORT §13](../docs/TECHNICAL_REPORT.md#13-accuracy).

## Running it

See [`docs/10_reproduce.md` §5](../docs/10_reproduce.md#5-level-4--measure-accuracy-on-coco-tasks). The scripts have our default dataset root (`D:\DVCon\datasets`, `$HOME/mnt/...`) — pass `--root` or edit the paths.
