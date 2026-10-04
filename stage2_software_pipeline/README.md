# stage2_software_pipeline/

The Stage 2A software solution (April – May 2026): **YOLOv8x + a fine-tuned MiniLM ranker** in Python, plus the training code and an abandoned MobileCLIP experiment. Story and results: [`docs/04_stage2a_software_pipeline.md`](../docs/04_stage2a_software_pipeline.md).

```
stage2_software_pipeline/
├── pipeline/                  the submitted pipeline (cleaned copy)
│   ├── inference.py           core: YOLO → MiniLM scoring → selection → JSON + annotated image
│   ├── run_pipeline_interactive.py, batch_inference.py
│   ├── evaluate_pipeline.py, batch_test_1000_pipeline.py   (COCO-Tasks evaluation)
│   ├── prepare_yolo_coco_tasks.py   JSONL → Ultralytics dataset (+ COCO id ↔ YOLO id tables)
│   ├── requirements-colab.txt
│   └── README.md              full usage
├── results/                   the balanced 1,000-row evaluation (summary JSON + CSV) and an example output
├── training/
│   ├── train_minilm_ranker.py      final trainer (BCE + margin ranking loss, group format)
│   ├── train.py                    first trainer (object-level binary classifier)
│   ├── colab_train_minilm.ipynb    Colab T4 training notebook with the full 20-epoch log
│   ├── colab_train_yolo_coco_tasks.ipynb, colab_test_pipeline.ipynb
│   ├── PIPELINE_USAGE.md
│   └── yolo_finetune_run/          YOLOv8n fine-tune on COCO-Tasks: args, metrics, curves (mAP50 0.496)
├── dataset_tools/
│   └── make_coco_tasks_jsonl.py    COCO-Tasks → one JSON line per (image, task)
└── experiments/mobileclip_abandoned/   see README there
```

**Not included** (too large or third-party): the fine-tuned MiniLM checkpoint `minilm_task_text_best.pt` (278.6 MB), `yolov8x.pt` (137 MB, download from Ultralytics), the COCO images and the derived JSON-lines files (re-create them with `make_coco_tasks_jsonl.py`).

Quick start:

```bash
cd pipeline
pip install torch torchvision && pip install -r requirements-colab.txt
python inference.py --image photo.jpg --task "serve wine" --output-image result.jpg --output-json result.json
```

Most scripts have Windows or Google-Drive paths from our machines at the top — adjust them for yours.
