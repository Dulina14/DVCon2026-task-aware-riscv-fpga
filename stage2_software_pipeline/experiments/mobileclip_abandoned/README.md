# MobileCLIP experiment (abandoned, April 2026)

Our Stage 1 plan was to crop every object found by YOLO and compare each crop with the task text using **Apple MobileCLIP** (an image–text model). We tried it for a few days in April 2026 and then switched to a text-only MiniLM ranker. These are the scripts we added on top of a clone of [apple/ml-mobileclip](https://github.com/apple/ml-mobileclip) (which is **not** included here — clone it and put these files in its root).

| Script | What it tried |
|---|---|
| `task_and_object_selector.py` | Match free text to one of the 14 tasks with the MobileCLIP-S2 text encoder, filter objects by a hand-made task → COCO-id table, rank crops by image–text similarity |
| `train_custom.py` | Contrastive (InfoNCE) fine-tuning of MobileCLIP-S2 on whole images ↔ task prompts (CPU, 1 epoch, capped at 51 batches) |
| `train_full.py` | The same on the full set (batch 32, 3 epochs, CPU) |
| `train_object_level.py` | One sample per object crop, binary loss on the scaled cosine similarity |
| `inference.py`, `inference_object_level.py`, `inference_test.py` | Single-image scoring, per-object re-ranking, in-batch retrieval accuracy |

## Why we dropped it

- YOLO already gives a **class name** for every object. Comparing *text with text* (MiniLM) is far cheaper than running an image encoder on every crop, and much easier to fine-tune — which matters a lot on a 50 MHz CPU.
- Training ran on CPU only, so iterations were slow.

## Bugs we found later (kept as lessons)

- In `train_object_level.py` and `inference_object_level.py`, `Image.LANCOS` should be `Image.LANCZOS`. The error was swallowed by a bare `except:`, so **every crop silently became an all-zero image** — the model could not have learned anything from pixels. *Never use a bare `except` around data loading.*
- The hand-written task → COCO-id table in `task_and_object_selector.py` has wrong ids (e.g. id 57 is *carrot*, not *chair*) and lists classes that are not in COCO ("lemon", "shovel").
- `requirements.txt` in our clone had a typo (`wgclip-benchmark`).

No accuracy results from this experiment were kept.
