# models/

INT8 model packages read by the board program. They are produced offline by [`offline_export/`](../offline_export/README.md) and contain only **input-independent** data (weights, scales, the 80 fixed class-name embeddings, metadata).

| File | Size (bytes) | Contents | Pair with header |
|---|---|---|---|
| `team43_yolo_int8.bin` | 3,192,064 | YOLOv8n: BatchNorm folded, per-output-channel INT8 weights pre-packed in 8×8 tile order, FP32 scales and biases | `firmware/generated/yolov8n/team43_yolo_graph.h` |
| `team43_yolo_s_int8.bin` | 11,229,824 | YOLOv8s, same format | `firmware/generated/yolov8s/team43_yolo_graph.h` |
| `team43_minilm_int8.bin` | 23,088,004 | Fine-tuned all-MiniLM-L6-v2 (6 layers) + ranker head + 80 precomputed class embeddings | `firmware/generated/team43_minilm_layout.h` |
| `team43_minilm_packed.bin` | 11,211,008 | Cache of the MiniLM weights already packed in accelerator tile order (`--packed`); big start-up saving on the board | — |
| `vocab.txt` | 262,030 | WordPiece vocabulary, 30,522 tokens, one per line in id order | — |
| `team43_minilm_manifest.json` | — | Human-readable map of every tensor in the MiniLM package | — |
| `team43_minilm_golden.json` | — | Host reference scores used to validate the board (e.g. "serve wine" → wine glass 0.970027) | — |

Checksums: [`SHA256SUMS.txt`](SHA256SUMS.txt).

**A YOLO header and package must match** (n with n, s with s). The program prints `largest weight block 294912` (v8n) or `1179648` (v8s) at start-up — the quickest check.

## Licences

These files are derived works:

- `team43_yolo_int8.bin`, `team43_yolo_s_int8.bin` — from **Ultralytics YOLOv8** weights, licensed **AGPL-3.0** ([ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)).
- `team43_minilm_*`, `vocab.txt` — from **sentence-transformers/all-MiniLM-L6-v2**, licensed **Apache-2.0**, fine-tuned by Team 43 on the COCO-Tasks annotations.
