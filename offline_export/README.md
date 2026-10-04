# offline_export/

Python tools that run **once, on a PC**, to turn the trained PyTorch models into the INT8 packages and C headers the board uses. Line-by-line explanation and the legality argument for each script: [`docs/code_guides/CODE_GUIDE_TOOLCHAIN.md`](../docs/code_guides/CODE_GUIDE_TOOLCHAIN.md).

| Script | Produces | Notes |
|---|---|---|
| `dump_vocab.py` | `vocab.txt` | From the all-MiniLM-L6-v2 tokenizer; checks ids are contiguous |
| `export_minilm_int8.py` | `team43_minilm_int8.bin`, `team43_minilm_manifest.json`, `team43_minilm_golden.json` | Needs the fine-tuned checkpoint (`CKPT` path at the top). Sequence length 8. Embeds the 80 fixed class names |
| `make_layout_header.py` | `team43_minilm_layout.h` | Turns the manifest into compile-time offsets (no JSON parsing on the board) |
| `yolo_flatten.py` | (check) | Flattens YOLOv8n into 6 op types (CONV, MAXPOOL, UPSAMPLE, CONCAT, SLICE, ADD) and re-implements it in NumPy — matches PyTorch to ~1.8e-7 |
| `yolo_int8_eval.py [IMGSZ]` | (check) | INT8 simulation (letterbox, im2col, per-channel weights) vs FP32 |
| `export_yolov8n_int8.py` | `team43_yolo_int8.bin`, `team43_yolo_graph.h`, golden detections | Folds BatchNorm, quantises per output channel, packs weights into 8×8 tile order. Point `PT` at `yolov8s.pt` for the YOLOv8s package |

**Before running:** edit the absolute paths at the top of the scripts (`CKPT`, `PT`) — they point to our machines.

**Requirements:** `torch`, `ultralytics`, `transformers`, `numpy`, `Pillow`.

**The standalone rule:** these scripts do only input-independent work — no detection, no activations, no task scores for any run-time input.
