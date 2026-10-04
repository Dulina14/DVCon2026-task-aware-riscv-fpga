# 10 — Reproduce it

You can rebuild this project at six levels. Levels 1–4 need only a normal PC. Levels 5–6 need the **organiser's VEGA SoC kit** (from C-DAC / DVCon), which is not public, and level 6 also needs a **Genesys 2 board**.

| Level | You get | You need |
|---|---|---|
| [1](#2-level-1--run-the-board-program-on-a-pc) | The exact board program running on a PC, bit-identical answers | gcc (Linux, macOS or WSL) |
| [2](#3-level-2--the-python-pipeline-stage-2a) | The Stage 2A Python pipeline (YOLOv8x + MiniLM) | Python 3.10+, PyTorch, Ultralytics, Transformers, a MiniLM checkpoint |
| [3](#4-level-3--re-create-the-int8-model-packages) | Your own INT8 model packages and headers | Python, PyTorch, Ultralytics, the checkpoints |
| [4](#5-level-4--measure-accuracy-on-coco-tasks) | mAP@0.5 / top-1 on COCO-Tasks | Python + the COCO 2014 and COCO-Tasks datasets |
| [5](#6-level-5--simulate-and-synthesise-the-accelerator-inside-the-soc) | Full-system simulation + synthesis | Vivado (2018.1 – 2026.1 tested) + organiser kit |
| [6](#7-level-6--run-on-a-real-genesys-2-board) | The real thing | Genesys 2 board + organiser bitstream flow, VEGA Linux image, buildroot Docker image |

---

## 1. Get the repository

```bash
git clone https://github.com/Dulina14/dvcon2026-task-aware-riscv-fpga.git
cd dvcon2026-task-aware-riscv-fpga
```

The INT8 model packages in `models/` are about 49 MB in total.

## 2. Level 1 — run the board program on a PC

The board program [`firmware/src/team43_pipeline.c`](../firmware/src/team43_pipeline.c) is one C file. With `--sim`, the accelerator is replaced by `sim_tile()`, an exact software copy, so you get the **same output as the board**.

```bash
cd firmware
mkdir -p build
cp src/*.c src/*.h generated/team43_minilm_layout.h build/
cp generated/yolov8n/team43_yolo_graph.h build/          # use generated/yolov8s/ for YOLOv8s
cd build
gcc -std=gnu99 -O2 -ffp-contract=off -Wall -Wextra -o pipeline_x86 team43_pipeline.c -lm

./pipeline_x86 --sim \
  --image ../../test_data/input/demo_image.jpg --task "serve wine" \
  --imgsz 256 --rect \
  --yolo   ../../models/team43_yolo_int8.bin \
  --minilm ../../models/team43_minilm_int8.bin \
  --vocab  ../../models/vocab.txt \
  --packed ../../models/team43_minilm_packed.bin \
  --out result.ppm
```

**Check the run header.** It must contain:

```
fpcheck: 34800000 3efe65ea be54d74a   (host and board must match)
ddr    : K/X windows 2 MB each, largest weight block 294912 bytes
```

- `-ffp-contract=off` is **not optional** — without it x86 and RISC-V round differently and `fpcheck` changes.
- `largest weight block 294912` means the YOLOv8n header and model match. For YOLOv8s, use `generated/yolov8s/team43_yolo_graph.h` with `models/team43_yolo_s_int8.bin`; the header then reports `1179648`. A mismatched pair gives garbage.
- On a laptop the run takes well under a second. On the 50 MHz board the same work took ~90 s (YOLOv8n) to ~275 s (YOLOv8s, 256 px).
- Try other tasks: `--task "sit comfortably"`, `"place flowers"`, `"pour sugar"` … Only the 14 COCO-Tasks prompts are meaningful (list in [01](01_the_challenge.md#3-the-reference-paper-and-dataset)).
- View the result: `python3 ../../evaluation/scripts/ppm_to_png.py result.ppm result.png` (needs Pillow).

All options: run `./pipeline_x86` with no arguments. The important ones are in [TECHNICAL_REPORT §3.3](TECHNICAL_REPORT.md#33-run).

**Build the probes** (board-only programs; they need the accelerator device, but they compile anywhere): see [`firmware/probes/README.md`](../firmware/probes/README.md).

## 3. Level 2 — the Python pipeline (Stage 2A)

```bash
cd stage2_software_pipeline/pipeline
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch torchvision
pip install -r requirements-colab.txt                  # Pillow, transformers, tqdm, ultralytics
```

You need two weight files next to `inference.py`:

- `yolov8x.pt` — downloaded automatically by Ultralytics on first use (or from the Ultralytics releases page).
- `checkpoints/finetuned_text/minilm_task_text_best.pt` — **our fine-tuned MiniLM** (278.6 MB, not in git). Either train it yourself (below) or use the release asset if one is published with this repository.

Run:

```bash
python inference.py --image path/to/photo.jpg --task "serve wine" \
    --output-image results/result.jpg --output-json results/result.json
```

Options: `--device auto|cpu|cuda`, `--yolo-conf 0.05`, `--none-threshold 0.10`, `--rank-by minilm|product`. Batch mode over a CSV: `batch_inference.py --csv inputs.csv --output-dir results/batch`. Full usage: [`pipeline/README.md`](../stage2_software_pipeline/pipeline/README.md).

**Train the MiniLM ranker yourself:**

1. Download COCO 2014 images + annotations and the COCO-Tasks annotations (section 5).
2. Build the JSON-lines files: `python stage2_software_pipeline/dataset_tools/make_coco_tasks_jsonl.py` (edit the dataset root at the top of the script).
3. Open [`training/colab_train_minilm.ipynb`](../stage2_software_pipeline/training/colab_train_minilm.ipynb) in Google Colab (a free T4 GPU is enough), or run `training/train_minilm_ranker.py` locally. 20 epochs; the best epoch is kept.

## 4. Level 3 — re-create the INT8 model packages

The scripts are in [`offline_export/`](../offline_export/README.md). They contain absolute paths from our machines (`CKPT = …`, `PT = …`) — edit those first.

```bash
pip install torch ultralytics transformers numpy
cd offline_export

python dump_vocab.py                 # → vocab.txt (30,522 tokens)
python export_minilm_int8.py         # → team43_minilm_int8.bin, _manifest.json, _golden.json
python make_layout_header.py         # → team43_minilm_layout.h
python yolo_flatten.py               # checks the 103-op graph against PyTorch (~1e-7)
python yolo_int8_eval.py 128         # INT8 vs FP32 on test images
python export_yolov8n_int8.py        # → team43_yolo_int8.bin, team43_yolo_graph.h, golden
```

For YOLOv8s, point `PT` in `export_yolov8n_int8.py` at `yolov8s.pt` (the graph is identical; only widths change).

The pre-packed MiniLM cache `team43_minilm_packed.bin` is produced by the board program itself: `./pipeline_x86 --sim … --emit-packed team43_minilm_packed.bin`.

**What may be done offline:** only input-independent work — folding BatchNorm, quantising constant weights, packing tiles, flattening the graph, embedding the 80 fixed class names, making golden references. Nothing that depends on the run-time image or task. See [`code_guides/CODE_GUIDE_TOOLCHAIN.md`](code_guides/CODE_GUIDE_TOOLCHAIN.md).

## 5. Level 4 — measure accuracy on COCO-Tasks

1. Download **MS-COCO 2014** (`train2014/`, `val2014/`, annotations) from [cocodataset.org](https://cocodataset.org/#download).
2. Download **COCO-Tasks** annotations from [github.com/coco-tasks/dataset](https://github.com/coco-tasks/dataset) (they are stored with Git LFS — run `git lfs pull`). For the paper's Faster R-CNN detection files, see the same repository.
3. Lay them out as expected by [`evaluation/eval_coco_tasks.py`](../evaluation/eval_coco_tasks.py) (`--root`, default was our `D:\DVCon\datasets`).
4. Run:

```bash
cd evaluation
python eval_coco_tasks.py --root /data/coco-tasks-root --boxes gt                       # ranker on perfect boxes
python eval_coco_tasks.py --root /data/coco-tasks-root --boxes faster                   # paper's detections
python eval_coco_tasks.py --root /data/coco-tasks-root --boxes gt --method pickbest     # sanity check vs paper baseline
```

The suitability scores come from [`results/minilm_task_class_scores.json`](../evaluation/results/minilm_task_class_scores.json), which was dumped from the **deployed INT8 package** (`--dump-scores`), so the evaluation reflects what the board computes. To evaluate our own detector, run the C program in batch mode (`--detect-list` / `--detect-json`) and pass the result with `--boxes detfile --detfile …`. Helper scripts: [`evaluation/scripts/`](../evaluation/scripts/). Expected results: [`EVAL_RESULTS.md`](../evaluation/EVAL_RESULTS.md).

## 6. Level 5 — simulate and synthesise the accelerator inside the SoC

You need the **organiser kit** (`DVCon_2026/` with `DVCon_SoC_SRC/AS1061_SYSTEM`, `TOP/Top.vhd`, `MEMORY_IP`, `TB`, `DVCon_SoC_XDC`, `TCL`, `RUN`, `Application`). Then:

1. Replace the kit's `DVCon_SoC_SRC/ACCELERATOR_IP/` with [`hardware/accelerator_ip/`](../hardware/accelerator_ip/).
2. Copy [`hardware/stage3a_simulation/firmware/`](../hardware/stage3a_simulation/firmware/) to `Application/stage3a_test/`, together with the kit's support files (`crt.S`, `uart.c`, `stdlib.c`, `mbl.lds`, `Makefile`, `include/`), build it with the kit's Docker toolchain, and install the 64-bit MIF as the boot ROM image.
3. Copy [`hardware/stage3a_simulation/tcl/`](../hardware/stage3a_simulation/tcl/) to `TCL/` and [`run/`](../hardware/stage3a_simulation/run/) to `RUN/`.
4. Run `./RUN/run_stage3a.sh --all` (Linux) or `RUN\run_stage3a.cmd` (Windows).

Expected: analyser verdict **ACCEPTED**, UART `BCSP`, 64/64 results, accelerator ≈ 12.4 k LUTs after synthesis. Details: [`hardware/stage3a_simulation/README.md`](../hardware/stage3a_simulation/README.md) and [06](06_stage3a_soc_integration.md).

## 7. Level 6 — run on a real Genesys 2 board

You need the organiser's hardware flow: the VEGA netlist (`AS1061_SYSTEM_TOP.edn`) to build a bitstream (or their prebuilt one), `FILE_TRANSFER/send.sh` + `bbl.bin` + device tree to boot VEGA Linux over Ethernet, and the `buildroot-2004_new` Docker image for the RISC-V cross-compiler.

1. Build the bitstream with our accelerator: [`hardware/REBUILD.md`](../hardware/REBUILD.md) (Vivado 2020.2 was used; full implementation takes hours). Edit the JTAG cable serial in `FPGA_PROGRAMMING.tcl` for your board.
2. Program the FPGA, boot VEGA Linux, bring up `eth0`: the full sequence is in [07 §3](07_stage3b_remote_hardware.md#3-the-daily-workflow-on-the-board) and [`reports/stage3b_quick_start.md`](reports/stage3b_quick_start.md).
3. Cross-compile (`firmware/scripts/build.sh`) or use [`firmware/prebuilt/`](../firmware/prebuilt/) (static RISC-V ELFs).
4. Copy the program, `models/*`, `vocab.txt` and your images to `/tmp/team43` on the board, plus a `SHA256SUMS.txt`, and run `verify.sh`.
5. Run `team43_memcheck` first (healthy board ≈ 6+ MB/s copy), then a query. Compare the `fpcheck` line with your PC run — they must match.

Expected on the board: YOLOv8n 256 px aspect-fitted ≈ 87–130 s per query; YOLOv8s 256 px ≈ 274 s; outputs identical to Level 1.

## 8. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `fpcheck` differs between PC and board | one build is missing `-ffp-contract=off` |
| Garbage detections | YOLO header and `.bin` do not match (n vs s) |
| Bottom of the picture looks wrong | the image file was truncated in transfer — check hashes |
| Same result, but 4–20× slower | board DRAM came up degraded; run `team43_memcheck`, re-program |
| Results wrong only for long dot products | `Y_DONE` not cleared before START |
| Stage 3A sim boots the wrong program | the ROM IP regenerated `rom_32KB_axi.mif` from its `.coe`; check the MIF hash |
