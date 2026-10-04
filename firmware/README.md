# firmware/

The program that runs on the VEGA RISC-V CPU under Linux on the Genesys 2 board. It takes a photo and a task sentence, runs YOLO and MiniLM using our accelerator, and writes the marked photo. The same source builds on a PC, where `--sim` replaces the accelerator with an exact software copy.

```
firmware/
├── src/
│   ├── team43_pipeline.c        the whole program (~2,000 lines) — final version (29 Aug 2026)
│   ├── jpeg.h                   baseline JPEG decoder written for this project (integer IDCT)
│   └── font8x8.h                8×8 bitmap font for the label drawn on the result
├── generated/                   produced by offline_export/ — must match the model file
│   ├── yolov8n/team43_yolo_graph.h   103-op graph for team43_yolo_int8.bin   (largest weight block 294,912 B)
│   ├── yolov8s/team43_yolo_graph.h   103-op graph for team43_yolo_s_int8.bin (largest weight block 1,179,648 B)
│   └── team43_minilm_layout.h        tensor offsets inside team43_minilm_int8.bin
├── frozen_aug26/                the frozen build used for the 14-task board evidence (+ SHA256SUMS)
├── milestones/                  programs used to bring the board up step by step (A–D) — see README there
├── probes/                      small programs, each answering one hardware question — see README there
├── scripts/                     build, deploy, verify and run scripts — see README there
└── prebuilt/                    static RISC-V ELF binaries that ran in the finals
```

## Build

```bash
# PC (x86-64), accelerator emulated with --sim
cp src/* generated/team43_minilm_layout.h generated/yolov8n/team43_yolo_graph.h build/ && cd build
gcc -std=gnu99 -O2 -ffp-contract=off -Wall -Wextra -o pipeline_x86 team43_pipeline.c -lm

# Board (RISC-V, static) — inside the organisers' buildroot Docker image
riscv64-buildroot-linux-gnu-gcc -std=gnu99 -O2 -ffp-contract=off -static -Wall -Wextra \
    -o team43_pipeline team43_pipeline.c -lm -lrt
```

**Both builds must use `-ffp-contract=off`.** Check the run header: `fpcheck: 34800000 3efe65ea be54d74a` on both machines (with `--silu-table` the middle word becomes `3efe65e9`).

## Run

```bash
./team43_pipeline --image photo.jpg --task "serve wine" --imgsz 256 --rect \
  --yolo team43_yolo_int8.bin --minilm team43_minilm_int8.bin --vocab vocab.txt \
  --packed team43_minilm_packed.bin --out result.ppm
```

| Option | Default | Meaning |
|---|---|---|
| `--imgsz N` | 128 | detector canvas width, multiple of 32 |
| `--rect` | off | fit the canvas to the photo's shape (`--imgsz` = long side) — no work wasted on padding |
| `--conf F` | 0.02 | detection floor, low on purpose (every withheld box is a candidate the ranker never sees) |
| `--rank-by` | `product` | `product` = detection × suitability (the paper's rule); `minilm` = text score only |
| `--kdepth N` | 0 | K-tiles per accelerator call; 0 = whole K in one call (fastest); 1 = original path |
| `--im2col-block N` | 64 | row-tiles per im2col block (smaller write set); 0 = unblocked |
| `--packed P` | — | pre-packed MiniLM weight cache (big speed-up); `--emit-packed P` creates it |
| `--silu-table` | off | table SiLU instead of `expf` |
| `--draw-top N` | 8 | draw only the N best candidates |
| `--sim` | off | emulate the accelerator in software (no FPGA needed) |
| `--dump-scores F` | — | write MiniLM(task, class) for all 80 classes as JSON, then exit |
| `--detect-list F --detect-json OUT` | — | batch detection for accuracy evaluation |
| `--unsafe-batched-regs` | off | **produces wrong results** — kept only for timing comparison |

The defaults for the model paths are `/tmp/team43/…` (the board's deploy folder).

## Which version is which

| File | Date | Notes |
|---|---|---|
| `src/team43_pipeline.c` | 29 Aug 2026 | Final: adds blocked im2col (`--im2col-block`), bit-identical output |
| `frozen_aug26/team43_pipeline.c` | 25–26 Aug 2026 | Build used for the 14-task evidence logs; SHA-256 in `frozen_aug26/SHA256SUMS.txt` |
| `milestones/team43_pipeline_milestoneD.c` | 17 Aug 2026 | Stage 3B submission version (Milestone D) |
| `prebuilt/team43_pipeline_n` / `_s` | 26 Aug 2026 | Static RISC-V binaries built with the YOLOv8n / YOLOv8s graph headers |

> The technical report (§9.2) describes a "bias-window" optimisation (opt1c) and a probe `team43_syscost_opt1c.c`. That branch is **not** in this source; see [TECHNICAL_REPORT §21](../docs/TECHNICAL_REPORT.md#21-known-gaps-between-this-document-and-the-tree).

## Read the code

- Short tour: [`FIRMWARE_STUDY_README_original.md`](FIRMWARE_STUDY_README_original.md) (line pointers into the source)
- Deep dive: [`docs/code_guides/CODE_GUIDE_FIRMWARE.md`](../docs/code_guides/CODE_GUIDE_FIRMWARE.md)
