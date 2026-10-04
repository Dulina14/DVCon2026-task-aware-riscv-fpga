# 09 — System architecture

This page explains the final system that ran in the finals: what each part does, **where data lives**, and **how it moves from one place to another**. For every measured number and the reasoning behind it, see the [technical report](TECHNICAL_REPORT.md).

---

## 1. One query, start to finish

```
INPUT (from the command line):  photo.jpg   +   "serve wine"
                                    │                 │
        ┌───────────────────────────┼─────────────────┼──────────────────────────────┐
        │  VEGA CPU   JPEG decode → resize/letterbox  │  WordPiece tokenise (≤ 8 tokens)│
        │                           │                 │                              │
        │  ACCEL      YOLO: 63 convolutions as        │  MiniLM: 6 layers ×          │
        │             matrix multiplies (im2col)      │  (Q, K, V, O, QKᵀ, A·V,       │
        │                           │                 │   FFN1, FFN2)                │
        │  VEGA CPU   SiLU, max-pool, upsample,       │  LayerNorm, softmax, GELU,   │
        │             concat, DFL, box decode, NMS    │  mean-pool, L2-normalise     │
        │                           │                 │                              │
        │                     candidate boxes         task vector t (384 numbers)    │
        │                     + class + confidence     + 80 fixed class vectors o     │
        │                           │                 │                              │
        │  ACCEL                    │      ranker head: [t, o, |t−o|, t⊙o] → 384 → 1   │
        │                           │      (only for classes that were detected)     │
        │  VEGA CPU   score = detection confidence × suitability → pick the best box   │
        │             draw boxes + label with an 8×8 bitmap font → write result.ppm    │
        └──────────────────────────────────────────────────────────────────────────────┘
OUTPUT:  printed list (label, score, box)  +  marked image
```

## 2. Who does what

**Rule:** anything that is a large INT8 matrix multiply goes to the accelerator; everything else stays on the CPU.

| On our 8×8 accelerator | On the VEGA RISC-V CPU |
|---|---|
| All 63 YOLO convolutions (via im2col) | JPEG decode (our own decoder), resize, normalise |
| MiniLM Q, K, V and output projections | SiLU, max-pool, upsample, concat, slice, residual add |
| MiniLM QKᵀ and attention × value | DFL softmax, box decode, sigmoid, non-max suppression |
| MiniLM feed-forward layers | Tokeniser, LayerNorm, softmax, GELU |
| The ranker head (2 layers) | Mean-pool, L2 norm, ranking, drawing, writing the image |

## 3. The models, as they exist on the board

### 3.1 Detector — YOLOv8n or YOLOv8s

- BatchNorm is folded into each convolution offline; weights are **INT8, one scale per output channel** (symmetric).
- The network is flattened offline into a **list of 103 simple operations** that reference **63 convolution weight tensors** (`firmware/generated/*/team43_yolo_graph.h`). The board walks that list with a small interpreter — there is no model-specific code in the program.
- YOLOv8n and YOLOv8s have the **same 103-operation structure**; only the layer widths differ. Changing detector = changing the header + model file.
- The canvas size is a run-time option (`--imgsz`, multiple of 32). `--rect` fits the canvas to the photo's shape so no work is spent on grey padding.

### 3.2 Ranker — fine-tuned all-MiniLM-L6-v2

| | |
|---|---|
| Hidden size / layers / heads | 384 / 6 / 12 (head size 32) |
| Feed-forward width | 1,536 |
| Vocabulary | 30,522 WordPiece tokens (`models/vocab.txt`) |
| Sequence length on the board | **8** (the longest task needs 7 tokens) |

**Why the class vectors can be precomputed:** the ranker sees the task text and the *class name*, never image pixels. The 80 COCO class names never change, so their 80 vectors are computed once offline and shipped in the model package. Only the **task** vector depends on the run-time input, and it is computed on the board. The head runs only for classes that were actually detected (e.g. 4 classes instead of 80).

## 4. The accelerator

### 4.1 What one call computes

`Y = X · K + B` on one **8×8×8 tile**: X (8×8 INT8 activations) times K (8×8 INT8 weights), plus B (8×8 INT32). That is **512 multiply-adds per tile**. For longer dot products, the running result is fed back in as the next B — or, after our *K-depth collapse*, the DMA simply streams the whole length of K in one call and the array accumulates it all.

### 4.2 Parameters (`hardware/accelerator_ip/rtl/sys/config.svh`)

| Parameter | Value | Meaning |
|---|---|---|
| `R × C` | 8 × 8 | processing-element grid |
| `WX`, `WK` | 8, 8 | activation / weight width (INT8) |
| `WY` | 32 | result width (INT32) |
| `AXI_WIDTH` | 64 | memory data path |
| `AXIL_WIDTH` | 32 | register interface |
| `ADDR_WIDTH` | 64 | address width |
| `AXI_MAX_BURST_LEN` | **1** (in `top_axi_int.v`) | every AXI transfer moves exactly one 64-bit word |

### 4.3 The data path — where data lives and how it moves

```
DDR3 memory (data lives here: weights in window K, activations in window X, results in window Y)
   │  64-bit AXI4, one 8-byte word per transaction
   ▼
AXI crossbar (4 DMA engines share one memory port)
   ├── read DMA 0 ──► K stream (64 bits = 8 weights, one per column) ──► skid buffer ─┐
   ├── read DMA 1 ──► X stream (64 bits = 8 activations, one per row) ─► skid buffer ─┤
   │                                                                               ▼
   │                                  axis_sa: 8×8 array (operands skewed in, results drained out)
   │                                                                               │
   ├── read DMA 2 ──► B stream (64→256 bits = 8 INT32 biases) ───────► skid buffer ─► + (8 adders)
   │                                                                               │
   └── write DMA ◄── Y stream (256→64 bits) ◄────────────── skid buffer ◄────────────┘
```

- **One 64-bit AXI word = one step of the array.** The software packs tiles so that a word read from DDR is already in the order the array needs. The K and X width adapters therefore do nothing and disappear in synthesis.
- **Skid buffers** are 2-deep registers that let `valid` and `ready` be registered without losing data (~1,336 flip-flops in total).
- K and X are **locked together**: the array steps only when both streams have a word.
- **Inside the array** the X rows and K columns are delayed by 0…7 cycles (a "triangular" buffer) so the right numbers meet in the right cell. Each cell keeps its own sum (*output-stationary*). Fill + drain ≈ 17 cycles ≈ 340 ns at 50 MHz.
- **Bias is added outside the array** by 8 parallel 32-bit adders; a result cannot leave until its bias word has arrived.

Per single tile: 640 bytes in, 256 bytes out, 24 AXI transactions, 512 multiply-adds.

### 4.4 Register map (from the CPU's point of view)

On the board the CPU uses the Linux character device `/dev/accelerator_test` (organiser driver). Offsets are bytes; **`pread`/`pwrite` counts are in 32-bit words, not bytes**.

| Offset | Register | Offset | Register |
|---|---|---|---|
| `0x00` | START | `0x20` | X tuser |
| `0x04` | K done | `0x24` | B done |
| `0x08` | K address | `0x28` | B address |
| `0x0C` | K bytes | `0x2C` | B bytes |
| `0x10` | K tuser | `0x30` | B tuser |
| `0x14` | X done | `0x34` | **Y done** |
| `0x18` | X address | `0x38` | Y address |
| `0x1C` | X bytes | `0x3C` | Y bytes |

In the SoC address map these registers sit at **`0x2004_0000`**.

**The sequence for one call (after the operands are in DDR):**

```c
rw(REG_Y_DONE, 0);          // clear the done flag first  (a real bug if you forget)
rw(REG_START, 0);
rw(REG_K_ADDR, k_tile);     // where the weights are
rw(REG_X_ADDR, x_tile);     // where the activations are
rw(REG_START, 1);           // go
wait_done();                // poll Y done, with a limit
```

Five single-word register writes per call is the minimum — a multi-word write sets only the first register (we tested it).

### 4.5 Memory map on the board

From the device tree: Linux owns `0x8000_0000 … 0xAF00_0000`. A 16 MB region above it (`0xBF00_0000 … 0xC000_0000`) is reserved, and we place the accelerator's buffers there, where Linux will never touch them:

| Window | Address range | Size | Holds |
|---|---|---|---|
| K | `0xBF00_0000 – 0xBF20_0000` | 2 MB | packed weight tiles |
| X | `0xBF20_0000 – 0xBF40_0000` | 2 MB | packed activation tiles |
| Y | `0xBF40_0000 – 0xBF60_0000` | 2 MB | result tiles |

The windows were 1 MB until YOLOv8s arrived with a 1,179,648-byte weight block that would have silently overwritten the X window. The program now prints the largest weight block at start-up and stops if it does not fit.

### 4.6 Cost on the chip

| Resource | Accelerator | Whole SoC | Device |
|---|---|---|---|
| LUTs | **11,945 (5.86 %)** | 107,937 (52.96 %) | 203,800 |
| Flip-flops | 12,706 (3.12 %) | 64,654 (15.86 %) | 407,600 |
| Block RAM | **0** | 91.5 (20.56 %) | 445 |
| DSP | **0** | 50 (5.95 %) | 840 |

The 8×8 array is 58 % of the accelerator's LUTs; the four DMA engines 19 %; the crossbar 7 %. About 108 LUTs per multiply-add cell.

## 5. Where the time goes, and why the array is not the bottleneck

For one 8×8×8 tile:

| | Time |
|---|---|
| The array's own work | ≈ **0.5 µs** |
| One accelerator call through the driver | ≈ 55 µs on-chip, ≈ 485 µs wall-clock |
| Measured wall-clock per output tile | ≈ **1,800 µs** |

**The array finishes in half a microsecond and then waits almost two milliseconds for software.** Each register access is a Linux system call (≈ 65–85 µs). That is why our big wins came from removing calls (operands uploaded once, K-depth collapse), not from a bigger array. With the array, the GEMM part ran **5.2× faster** than the identical binary in software emulation; end to end **2.8×**, because ~47 s per query (packing, software floating point, JPEG decode) is work the array cannot touch.

Full analysis, including why we chose not to build a 16×16 array: [TECHNICAL_REPORT §11](TECHNICAL_REPORT.md#11-performance).

## 6. Keeping the PC and the board in agreement

The same C file is compiled for the board (RISC-V) and the PC (x86). On the PC, `--sim` replaces the accelerator with an exact software copy (`sim_tile`). This only works if both compilers round floating point the same way:

- RISC-V GCC **fuses** `a*b + c` into one instruction (FMA); x86 does not. Same code, different last bits.
- Fix: compile **both** with `-ffp-contract=off`.
- Proof on every run: the program prints an `fpcheck` line computed from a deliberately sensitive expression. A healthy build prints `fpcheck: 34800000 3efe65ea be54d74a` on both machines.

This is what makes every hardware number in this repository checkable on an ordinary PC.
