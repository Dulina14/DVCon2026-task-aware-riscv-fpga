# Task-Aware Object Selection on a RISC-V FPGA

**Team 43 — Silicon Syndicate** · University of Moratuwa, Sri Lanka

> **About this document.** This is the full technical write-up we prepared for
> the DVCon India 2026 finals (the last version of our working write-up). It is
> reproduced here almost unchanged. File names refer to our working tree; the map
> to this repository is in the top-level `README.md` ("Repository map"). Section
> 21 honestly lists two places where this text describes work that is not in the
> shipped source — read it before quoting §9.2.

DVCon India 2026 Design Contest · Digilent Genesys 2 · Xilinx Kintex-7 XC7K325T · VEGA AS1061 RISC-V @ 50 MHz

---

## 1. What this is

Give the board a photograph and an English instruction. It gives back the same
photograph with one object outlined — the object you would actually reach for to
do that job.

```
$ ./team43_pipeline_n --image kitchen.jpg --task "serve wine" --imgsz 256 --rect \
      --out result.ppm

--- DETECTED (74) ---
  laptop 0.799 · person 0.708 · chair 0.617 · cup 0.221 · wine glass 0.120 · ...

--- TASK-AWARE RANKING (minilm x yolo) ---
  cup          yolo 0.221  minilm 0.93393  rank 0.20655   <== SELECTED
  wine glass   yolo 0.120  minilm 0.97036  rank 0.11676
  ...
--- RESULT ---
  SELECTED : cup   (rank 0.20655)
```

This is **not** object detection. A detector answers *what is in the picture*. We
answer *which one of those things is right for this task* — and the right answer
changes when the sentence changes, on the same photograph.

**Everything runs on the board.** The image is decoded on the board, both neural
networks run on the board, and the answer is computed on the board. No host
performs any input-dependent computation at runtime.

### Headline results

| | |
|---|---|
| End-to-end latency | **87.5 s** (YOLOv8n, 256 px aspect-fitted, `--silu-table`) |
| Accelerator speed-up | **5.2× on GEMM, 2.8× end-to-end** vs the identical binary in software |
| Accuracy (YOLOv8s, 256 px) | **top-1 0.627 · mAP@0.5 0.425** |
| Reference paper (GPU Faster R-CNN) | top-1 0.640 · mAP@0.5 0.409 |
| Ranker on ground-truth boxes | **mAP@0.5 0.648** · top-1 94.7 % |
| Accelerator fabric cost | **11,945 LUTs (5.86 %), 0 DSP, 0 BRAM** |
| Timing | 50 MHz system clock, WNS **+0.556 ns**, 157,192 endpoints |
| Host ↔ board agreement | **bit-identical** output, verified by `fpcheck` on every run |

### 1.1 The competition, and where this sits in it

The **DVCon India 2026 Design Contest** ran in stages, each gating the next.

| stage | what it demanded | our outcome |
|---|---|---|
| 1–2 | Concept and accelerator design | advanced |
| **3A** | Integrate a team-designed accelerator into the organisers' VEGA AS1061 SoC; demonstrate a **successful simulation**; **synthesise** and report utilisation | passed — see §10.6 |
| **3B** | **Hardware implementation** on the physical Genesys 2 | this document |
| final | 6 teams present on stage and demonstrate at the conference | **Winner** (Bengaluru, 1 Sept 2026) |

Stage 3A was judged on evidence, not description. The organisers stated they
would download each submission and run its script themselves, and that *"if
anything is missing… we will reject that submission."* They checked three
things: that proper AXI transactions were generated, that the accelerator
actually computed, and that results reached memory — plus whether the design
would physically fit, with informal guidance to stay inside **20–25 %** of the
device.

**Twelve of the entrant teams advanced to Stage 3B.** Team 43 — Silicon
Syndicate — was one of them.

Two consequences shaped this project and are visible throughout the document.
First, the accelerator was **verified in simulation before it ever met silicon**
(§10.6), which is why the hardware bring-up in §10.2 proceeded without a single
RTL change. Second, the submission had to be **reproducible by a stranger on an
unknown machine**, which is the origin of the payload verification in §12.1, the
`fpcheck` fingerprint in §8.3, and the habit throughout of shipping a probe
alongside every claim.

---

## 2. The problem

Task-driven object detection, as posed by **Sawatzky, Souri, Grund and Gall,
CVPR 2019**, with their **COCO-Tasks** dataset: fourteen everyday tasks, each
with task-dependent ground truth over COCO images. The preferred object for
*"serve wine"* is a wine glass if one is present, and a cup if not — so the
annotation is a function of the sentence, not of the picture alone.

The fourteen prompts, exactly as trained:

| # | prompt | # | prompt |
|---|---|---|---|
| 1 | step on something to reach top of a shelf | 8 | open bottle of beer |
| 2 | sit comfortably | 9 | open parcel |
| 3 | place flowers | 10 | serve wine |
| 4 | get potatoes out of fire | 11 | pour sugar |
| 5 | water plant | 12 | smear butter |
| 6 | get lemon out of tea | 13 | extinguish fire |
| 7 | dig hole | 14 | pound carpet |

Prompts outside this set are outside the training distribution and their scores
carry no meaning. That is a property of the fine-tuned checkpoint, not of the
deployment.

Where the reference work uses a Faster R-CNN detector and a Gated Graph Neural
Network on desktop hardware, we target the same problem with a deliberately
lightweight pair — YOLOv8n/YOLOv8s for detection, a fine-tuned MiniLM ranker for
selection — quantised to INT8 and executed on an embedded FPGA platform.

---

## 3. Quick start

### 3.1 Build

The board binary is cross-compiled in the organiser's buildroot container. **Both
the board and host builds must use `-ffp-contract=off`** (see §8.3 — this is not
optional).

```bash
# board (RISC-V, static)
docker run --rm -t -v "$PWD":/workspace -w /workspace --user $(id -u):$(id -g) \
  buildroot-2004_new bash -c \
  'B=buildroot_dvcon/buildroot/output/host/bin/riscv64-buildroot-linux-gnu-gcc
   $B -std=gnu99 -O2 -ffp-contract=off -static -Wall -Wextra \
      -o team43_pipeline team43_pipeline.c -lm -lrt'

# host (x86, --sim emulates the accelerator in software)
gcc -std=gnu99 -O2 -ffp-contract=off -Wall -Wextra \
    -o pipeline_x86 team43_pipeline.c -lm
```

Required headers alongside the `.c`: `jpeg.h`, `font8x8.h`,
`team43_yolo_graph.h`, `team43_minilm_layout.h`. The YOLO graph header must match
the model blob — v8n header with `team43_yolo_int8.bin`, v8s header with
`team43_yolo_s_int8.bin`. The run header prints `largest weight block 294912`
for v8n and `1179648` for v8s, which is the quickest way to confirm the pairing.

### 3.2 Program the board

1. **Load the bitstream.** `cd .../RUN && ./run.sh`, choose option **3**. The
   launcher rewrites `MDIR_PATH` in the four TCL scripts to match the local tree,
   which removes a whole class of file-not-found failures. `FPGA_PROGRAMMING.tcl`
   selects the JTAG cable by serial number — that line must match your board.
2. **Load the processor image.** `send.sh` compiles the device tree and pushes
   `bbl.bin` + `riscv.dtb` over raw Ethernet to the board's MAC. On Windows use
   `send.ps1` / `send.bat` (§12.2).
3. **Copy the payload.** `scp team43_deploy.tar root@192.168.1.50:/tmp/` then
   untar into `/tmp/team43`.
4. **Verify the payload.** `cd /tmp/team43 && ./verify.sh` — see §12.1. Do not
   skip this.

> `/tmp` on the board is a RAM disk. Everything there is lost on power-down.
> Restoring ~30 MB takes about six minutes at the measured 94 KB/s link rate.
> Leaving the board powered between sessions avoids the problem entirely.

### 3.3 Run

```bash
cd /tmp/team43
./team43_pipeline_n \
  --image live_image.jpg \
  --task "serve wine" \
  --imgsz 256 --rect \
  --yolo   team43_yolo_int8.bin \
  --minilm team43_minilm_int8.bin \
  --vocab  vocab.txt \
  --packed team43_minilm_packed.bin \
  --out    result.ppm
```

The flags that matter:

| flag | default | meaning |
|---|---|---|
| `--imgsz N` | 128 | detector canvas width, multiple of 32 |
| `--rect` | off | fit the canvas to the photo's aspect — `--imgsz` becomes the long side, no accelerator work spent on grey padding |
| `--conf F` | 0.02 | detection floor, deliberately low (§6.2) |
| `--rank-by` | `product` | `product` = detection × affinity; `minilm` = language only |
| `--kdepth N` | 0 | K-tiles per invocation; 0 = whole K in one (§9.1). `1` restores the original path |
| `--packed P` | — | pre-packed MiniLM weight cache, large speed-up |
| `--silu-table` | off | table-interpolated SiLU instead of `expf` |
| `--draw-top N` | 8 | draw only the N best-ranked candidates |
| `--sim` | off | emulate the accelerator in software — no FPGA needed |
| `--dump-scores F` | — | write MiniLM(task, class) for all 80 classes as JSON, then exit |
| `--detect-list` / `--detect-json` | — | batch detection mode for accuracy evaluation |

Because the graph shapes come from the generated header and the canvas is a
runtime argument, **one binary and one weight package serve every resolution and
every prompt.** A different image or task needs no rebuild.

---

## 4. System architecture

### 4.1 The partition

One rule: anything expressible as a dense INT8 matrix multiply goes to the
accelerator; everything else runs on the VEGA core.

| On the 8×8 accelerator | On the VEGA RISC-V core |
|---|---|
| All 63 YOLOv8n convolutions, via im2col | JPEG decode, letterbox, normalisation |
| MiniLM Q, K, V and output projections | SiLU, max-pool, upsample, concat, slice |
| MiniLM QKᵀ and attention·value products | Residual add, DFL soft-max, box decode |
| MiniLM FFN1 and FFN2 | Non-max suppression, sigmoid |
| Trained head layers `head.0` and `head.3` | LayerNorm, attention soft-max, GELU |
| | Mean-pool, L2 norm, ranking, drawing, PPM write |

### 4.2 The runtime sequence

```
image (.jpg or .ppm) + task string, both from argv
 -> JPEG decode (baseline DCT, written for this project)      VEGA
 -> letterbox to the inference size, normalise                VEGA
 -> YOLOv8n/s, 103-op graph, 63 convolutions as im2col GEMM   ACCELERATOR
 -> SiLU, max-pool, upsample, concat, slice, residual add     VEGA
 -> DFL soft-max, box decode, sigmoid, non-max suppression    VEGA
 -> WordPiece tokenisation of the task prompt                 VEGA
 -> MiniLM: 6 encoder layers, Q/K/V/O, QKᵀ, A·V, FFN1, FFN2   ACCELERATOR
 -> LayerNorm, soft-max, GELU, residual, mean-pool, L2-norm   VEGA
 -> trained head over the detected classes (2 GEMMs)          ACCELERATOR
 -> ranking, box drawing, image write-back                    VEGA
```

---

## 5. The detector

YOLOv8n (and optionally YOLOv8s), with BatchNorm folded into the preceding
convolutions and **per-output-channel symmetric INT8** weight quantisation.

Offline, the network is flattened from its PyTorch module hierarchy into a
**linear list of 103 operations referencing 63 convolution weight tensors**. The
C2f split/concatenate structure, the SPPF max-pool chain, the nearest-neighbour
upsamples and the detection head are all expanded into explicit primitives. The
board walks that list with a small interpreter — **no model-specific control flow
is compiled into the binary.**

Every convolution becomes an im2col GEMM. YOLOv8n and YOLOv8s share a depth
multiplier of 0.33, so they have **identical graph structure** — the same 103 ops
and 63 weights — and differ only in width (max output channels 256 vs 512; blob
3,192,064 vs 11,229,824 bytes). Swapping detectors therefore needed no code
change beyond sizing two buffers from the graph instead of from a constant.

---

## 6. The language ranker

### 6.1 Architecture

`all-MiniLM-L6-v2`, fine-tuned as a task–object suitability scorer:

| | |
|---|---|
| hidden size | 384 |
| encoder layers | 6 |
| attention heads | 12 (head dim 32) |
| feed-forward width | 1536 |
| vocabulary | 30,522 |
| deployed sequence length | **8** (the longest supported prompt tokenises to 7) |

The scorer embeds `"task: <prompt>"` and `"object: <class name>"`, forms the
**fused 1536-wide vector** `[t, o, |t − o|, t ⊙ o]`, and passes it through a
two-layer trained head (1536 → 384, ReLU, → 1, sigmoid).

**One design consequence shapes the whole system.** The ranker sees the task
string and the object's *class name* — never image pixels. The eighty COCO label
embeddings therefore depend on nothing at runtime, so they are computed once
offline and shipped inside the package. Mathematically identical to computing
them on the board, at zero runtime cost. The **task** embedding depends on the
runtime prompt and *is* computed on the board, and the trained head is two GEMMs
that run on the accelerator — so the final selection genuinely depends on
accelerator output.

The head evaluates only the classes actually detected, not all eighty. That is
the `head over N detected classes, not all 80` line in the run output; for a
four-class scene it cuts the head from 92,640 invocations to 9,264.

### 6.2 Two design choices, both measured

**Multiply, don't replace.** The final score is `detection confidence ×
language affinity`, which is the reference paper's rule and the one the metric
rewards.

| ranking | mAP@0.5 | top-1 |
|---|---|---|
| product | **0.0975** | 0.198 |
| language only | 0.0705 | 0.198 |

Identical on the single marked box, 38 % better over the ranked list. A weak box
should not outrank a strong one just because its class name sounds right.

**Keep the weak boxes.** The detection floor is 0.02, not the 0.25 a detection
demo would use. Every box withheld is a candidate the ranker never sees. Measured
on 717 COCO-Tasks test images at 192 px with product ranking:

| conf | mAP@0.5 | top-1 | right answer on the list at all |
|---|---|---|---|
| 0.01 | 0.2046 | 0.3036 | 0.5714 |
| **0.02** | **0.2008** | **0.3107** | 0.5214 |
| 0.05 | 0.1914 | 0.2857 | 0.4500 |
| 0.10 | 0.1786 | 0.2643 | 0.3857 |

Accuracy still climbs below 0.02, but the shortlist stops being readable, and
product ranking already discounts a weak box by its own confidence.

---

## 7. The accelerator

### 7.1 What it computes

One 8×8×8 INT8 tile product per invocation: **Y = X · K + B**, where X is an 8×8
INT8 activation tile, K an 8×8 INT8 weight tile, and B and Y are 8×8 INT32
accumulator tiles. **512 multiply-accumulates per invocation.**

Accumulation along the contraction dimension uses the bias path: the accumulator
is seeded with the bias, and each successive tile presents the running partial
result as the bias input, so the hardware accumulates with no host addition.

### 7.2 Parameters (`config.svh`)

| parameter | value | meaning |
|---|---|---|
| `R × C` | 8 × 8 | processing-element array |
| `WK`, `WX` | 8, 8 | weight / activation width (INT8) |
| `WY` | 32 | accumulator and result width (INT32) |
| `AXI_WIDTH` | 64 | operand streaming data path |
| `AXIL_WIDTH` | 32 | control register width |
| `ADDR_WIDTH` | 64 | descriptor address width |
| `AXIS_USER_WIDTH` | 8 | stream sideband |
| `AXI_MAX_BURST_LEN` | **1** | set in `top_axi_int.v` line 93 — see §7.5 |

### 7.3 Register map

Byte offsets into `/dev/accelerator_test`. **`pread`/`pwrite` counts are in
32-bit words, not bytes** — a common trip-up.

| | | | |
|---|---|---|---|
| `0x00` START | `0x04` K done | `0x08` K addr | `0x0C` K bytes |
| `0x10` K tuser | `0x14` X done | `0x18` X addr | `0x1C` X bytes |
| `0x20` X tuser | `0x24` B done | `0x28` B addr | `0x2C` B bytes |
| `0x30` B tuser | `0x34` Y done | `0x38` Y addr | `0x3C` Y bytes |

### 7.4 The data path, end to end

```
DDR3 (MIG)
   |  64-bit AXI4, burst length 1  -> 8 bytes per transaction
   v
axi_crossbar  (4 slaves -> 1 master, M_BASE_ADDR=0, M_ADDR_WIDTH=64: no decode)
   |
   +-- MM2S_0 DMA --> ADAPTER_MM2S_K (64->64, identity, optimised away) --> SKID_K
   +-- MM2S_1 DMA --> ADAPTER_MM2S_X (64->64, identity, optimised away) --> SKID_X
   +-- MM2S_2 DMA --> ADAPTER_MM2S_A (64->256, gathers 4 beats)         --> SKID_A
   |
   v
axis_sa : tri_buffer skew -> 8x8 PE array -> diagonal drain
   |
   +-- per-row 32-bit adders:  m_y_tdata[r] = SA[r] + A[r]
   v
SKID_Y --> ADAPTER_S2MM (256->64, splits into 4 beats) --> S2MM DMA --> DDR3
```

**Internal bus widths, all derived from R and C:**

| bus | formula | width | one word is |
|---|---|---|---|
| `K_BUS_W` | C × WK | **64 bits** | 8 weight bytes, one per column |
| `X_BUS_W` | R × WX | **64 bits** | 8 activation bytes, one per row |
| `A_BUS_W` | R × WA | **256 bits** | 8 INT32 biases |
| `Y_BUS_W` | R × WY | **256 bits** | 8 INT32 results |

**One AXI beat is exactly one array step.** The AXI bus is 64 bits and so are
`K_BUS_W` and `X_BUS_W` — 8 bytes feeding the 8 columns, or the 8 rows. This is
why the software packs tiles with byte index `kk*8 + rr` for X and `kk*8 + cc`
for W: a 64-bit word read from DDR arrives already in beat order with zero
reformatting, and the K and X adapters reduce to nothing in synthesis.

**Skid buffers.** Not FIFOs — `skid_buffer.sv` has exactly two storage registers
(`m_data` and `buffer`) with an EMPTY/PARTIAL/FULL controller. **Depth 2,
always.** The job is to register both data and `ready` so `valid` never has to be
pulled combinationally; two slots is the minimum that absorbs one cycle of
back-pressure without dropping a beat.

| buffer | width | breakdown | storage |
|---|---|---|---|
| `SKID_K` | 73 bits | 64 data + 8 tuser + 1 tlast | 146 FF |
| `SKID_X` | 73 bits | 64 data + 8 tuser + 1 tlast | 146 FF |
| `SKID_A` | 265 bits | 256 data + 8 tuser + 1 tlast | 530 FF |
| `SKID_Y` | 257 bits | 256 data + 1 tlast | 514 FF |

≈ **1,336 flip-flops** of skid storage in total.

**The two input streams are locked together:**

```verilog
assign s_valid    = s_k_tvalid & s_x_tvalid;
assign s_last     = s_k_tlast  & s_x_tlast;
```

The array steps only when both K and X have a beat; if one DMA starves, the other
stalls. And `s_last` is the AND of both `tlast`s, which the DMA asserts at
descriptor end — **that is the mechanism behind the k-depth collapse in §9.1.**

**Inside the array.** `sk_reversed` flips the column order so results emerge with
C=0 first. Two `tri_buffer` instances skew X by row and K by column, creating the
diagonal wavefront. Diagonal length `D = R + C − 1 = 15`; `n_delay` pipelines
valid/last by `LM + LA + D = 17` stages to match. Fill-and-drain ≈ 17 cycles
≈ **340 ns** at 50 MHz.

```
                 K columns, skewed by tri_buffer (c cycles of delay)
                 c0    c1    c2   ...   c7
                  |     |     |          |
        r0 --->  PE ── PE ── PE ─ ... ─ PE  ──┐
                  │     │     │          │    │
X rows, r1 --->  PE ── PE ── PE ─ ... ─ PE  ──┤
skewed    │       │     │     │          │    │  diagonal
by r      :       :     :     :          :    │  drain
cycles    │       │     │     │          │    │
        r7 --->  PE ── PE ── PE ─ ... ─ PE  ──┘
                                               │
                       accumulators stay put ──┤  output-stationary:
                       operands move           │  each PE owns one Y[r][c]
                                               v
                         + bias  (8 parallel 32-bit adders, OUTSIDE the array)
                                               │
                                               v   m_y_tvalid = s_a_tvalid & m_valid
                                            SKID_Y
```

Operands enter on two edges and advance one PE per cycle; the accumulator never
moves. Because the skew delays row *r* by *r* cycles and column *c* by *c*, the
product for PE(r, c) meets its operands on exactly the right cycle — the
wavefront sweeps the array diagonally, and after `D = 15` steps every PE is busy
simultaneously.

**The bias is added outside the array**, in eight parallel 32-bit adders in
`top_axi_int`, and `m_y_tvalid = s_a_tvalid & m_valid` — a result cannot leave
until its bias beat has arrived. B is a genuinely independent read stream with
its own descriptor, which is what made the optimisation in §9.2 legal.

### 7.5 Burst length — the one number to know

`AXI_MAX_BURST_SIZE = AXI_MAX_BURST_LEN << AXI_BURST_SIZE = 1 << 3 = 8 bytes`,
so `m_axi_arlen = 0`: **every AXI transaction moves exactly one 64-bit beat.**

A 64-byte K tile is not one 8-beat burst — it is **8 separate AXI read
transactions**, each with its own address handshake through the crossbar to the
memory controller. Per single tile:

| stream | bytes | AXI transactions |
|---|---|---|
| K in | 64 | 8 |
| X in | 64 | 8 |
| A in (bias) | 256 | 4 |
| Y out | 256 | 4 |

**640 bytes in, 256 bytes out, 24 AXI transactions, for 512 MACs.**

### 7.6 Fabric cost

Vivado 2020.2, device `7k325tffg900-2`, design state Routed.

| resource | accelerator | whole SoC | available |
|---|---|---|---|
| Slice LUTs | **11,945 (5.86 %)** | 107,937 (52.96 %) | 203,800 |
| Slice registers | 12,706 (3.12 %) | 64,654 (15.86 %) | 407,600 |
| Block RAM tiles | **0** | 91.5 (20.56 %) | 445 |
| DSP slices | **0** | 50 (5.95 %) | 840 |

Where the accelerator's LUTs go:

| block | LUTs | share |
|---|---|---|
| `axis_sa` — the 8×8 array | 6,913 | 58 % |
| four DMA engines (3 read + 1 write) | 2,306 | 19 % |
| AXI crossbar | 893 | 7 % |
| `dma_controller` register file | 332 | 3 % |
| AXI-Lite interface + stream adapters | 551 | 5 % |
| top-level AXI wrapper | 242 | 2 % |

≈ **108 LUTs and 99 flip-flops per multiply-accumulate cell.**

**The two zeros matter.** Both the MAC cells and the array's internal buffering
are LUT logic and registers, so the accelerator competes with nothing else on the
device for its scarcest resources — and leaves 840 DSP slices and 79 % of the
block RAM untouched for a larger array.

### 7.7 Timing

| clock | frequency | endpoints | WNS | WHS |
|---|---|---|---|---|
| `clk_out1_clk_wiz_0` (system) | 50 MHz | 157,192 | **+0.556 ns** | +0.051 ns |
| `clk_out2_clk_wiz_0` | 200 MHz | 1,706 | +1.331 ns | — |
| `clk_ref_mmcm_400` (MIG) | 400 MHz | 14 | +1.767 ns | — |
| `clk_out1/2_clk_wiz_eth` | 125 MHz | 3,820 | +0.593 ns | — |

Both VEGA and the accelerator sit in the 50 MHz domain. **Setup is met on every
clock with no failing endpoints.** One hold path fails at −0.659 ns on the
Ethernet RGMII receive crossing (`eth_rxctl` → IDDR), inside an
organiser-supplied block; no team-authored logic is involved and the link works.
The overall report line therefore reads `Timing constraints are not met` — raise
this yourself before anyone asks.

### 7.8 Integration

The organiser top level `Top.vhd` already declared a component called
`Accelerator_Top` but did not supply it. We wrote that module with a matching
port list, so **no organiser file had to be edited.**

---

## 8. The standalone boundary

### 8.1 The rule

The host **may** convert checkpoints, quantise constant weights, emit static
metadata, build reference outputs and transfer files.

The host **must not** compute, for any particular runtime input: the detections,
the prompt embedding, the activations, the candidate scores, or the final answer.

| offline, once (input-independent) | on the board, every run (input-dependent) |
|---|---|
| BatchNorm folding into convolutions | JPEG decode and letterboxing |
| INT8 quantisation of constant weights | All activations at every layer |
| Pre-packing weights into 8×8 tile order | Activation quantisation scales |
| Flattening YOLOv8 into a 103-op list | Every detection and its confidence |
| Embedding the 80 fixed COCO label names | Tokenising and embedding the task prompt |
| Generating reference outputs for testing | All suitability scores and the selection |

Two points deserve explicit note. The **eighty label embeddings** are precomputed
because they depend only on fixed class names, never on the runtime image or
prompt — the result is mathematically identical to computing them on the board.
And **images arrive as ordinary JPEG files** and are decoded on the board by a
baseline decoder written for this project, so no image preparation whatsoever
happens off-board. A raw PPM path is retained as an alternative.

### 8.2 The JPEG decoder

The platform has none, so we wrote one: baseline DCT, integer islow IDCT, table
colour conversion. Verified against a reference decoder on sixteen COCO
photographs — mean absolute difference 0.03–0.04, maximum 3 on a 0–255 scale,
which is inverse-DCT rounding and nothing more.

### 8.3 Numerical equivalence, and the FMA trap

The same source is compiled for both the board and the host, and `--sim`
emulates the accelerator in software. That only works if both round identically.

**RISC-V contracts `a*b+c` into a single fused multiply-add; x86-64 does not.**
Same source, different rounding — and our own software reference quietly stopped
being a reference. This was invisible for a long time and was only caught when a
supposedly bit-identical optimisation changed a score in the fifth decimal.

The fix is `-ffp-contract=off` on **both** builds, plus a fingerprint printed in
every run header:

```c
volatile float a = 1.0000001f, b = 3.0000002f, c = -3.0000005f;
p1.f = a * b + c;        /* catastrophic cancellation: fused != unfused */
p2.f = silu(0.7351f);
p3.f = silu(-2.3174f);
printf("fpcheck: %08x %08x %08x   (host and board must match)\n", ...);
```

A healthy default (`expf`) build prints:

```
fpcheck: 34800000 3efe65ea be54d74a
```

**If host and board print different fpcheck lines, no comparison between them
means anything.** (With `--silu-table` the third word changes to `3efe65e9`,
because the table changes the SiLU evaluation — that is expected; what matters is
that the two sides agree with each other.)

---

## 9. Optimisation history

Every optimisation in this table was verified byte-identical or against a defined
numerical tolerance before it was accepted. The ones that failed that test were
rejected and are listed too.

| change | predicted | delivered | verification |
|---|---|---|---|
| Weight packing: walk source, scatter to tile | — | 154.7 s → 7.6 s (**20.3×**) | byte-identical on 4 real tensors |
| Bulk-copy weight tiles out of the mmap | — | 4,502 µs → negligible per tile | byte-identical |
| Operands uploaded once, inner loop = 5 register writes | — | **4.37×** | bit-identical |
| **K-depth collapse** (§9.1) | — | 713 s → 218 s at 320 px (**3.3×**) | md5-identical at 128 and 320 px |
| Blocked dequantise loop | ~29 s | **~1 s** | bit-identical, kept anyway |
| SiLU lookup table | ~50 s | **~0 s** once FMA was fixed | demoted to opt-in |
| Reciprocal quantise | — | small | bit-identical |
| Bias-window separation (§9.2) | 8–9 s | pending board A/B | `--sim` byte-identical, 4 configs |
| im2col row-tile blocking | unknown | pending board A/B | `--sim` byte-identical |
| Multi-word register write | large | **PROVEN INCORRECT** | rejected — sets only the first word |

Two of four predicted wins were off by more than an order of magnitude. That is
why every optimisation now ships with a standalone probe that measures it in
seconds instead of a 130-second query.

### 9.1 The k-depth collapse — the largest single win

**What we assumed:** one accumulation per K-tile, so a K=576 convolution meant 72
separate invocations and 72 × 8 system calls.

**What the RTL actually says:** the array ends an accumulation on `s_last`, which
the DMA asserts at the end of a descriptor — and the descriptor length is just
the BYTES register. **K per invocation is a software choice, not an RTL
constant.**

```c
int KC = (g_kdepth > 0 && g_kdepth < KT) ? g_kdepth : KT;   /* default: whole K */
for (int k0 = 0; k0 < KT; k0 += KC) {
    int kc = (KT - k0 < KC) ? KT - k0 : KC;
    set_stream_bytes((uint32_t)kc * INPUT_BYTES);
    rw(REG_Y_DONE, 0u);
    rw(REG_START,  0u);
    rw(REG_K_ADDR, ...);
    rw(REG_X_ADDR, ...);
    rw(REG_START,  1u);
    wait_done();
}
```

**320 px: 713 s → 218 s, byte-identical, md5-verified.** Proved first in a
314-line standalone probe (`team43_kdepth.c`) against a software reference at
K = 8 … 1024, before the pipeline was touched.

This is also what reversed the earlier finding that software emulation beat the
accelerator per tile: before the collapse, five system calls bought 512 MACs;
after it, the same five calls buy the entire K run — up to 36,864 MACs for a
K=576 convolution.

### 9.2 Optimisation 1 — the bias window

`REG_B_ADDR` and `REG_Y_ADDR` were both pointed at `Y_ADDR`, so the accelerator's
result destroyed its own accumulator seed and **every output tile had to
re-upload its 256-byte bias tile** — 7.2 MB of device writes per query to restate
values that take only NT distinct forms.

B is MM2S_2, a read channel with its own descriptor; Y is S2MM, the write
channel. They never had to share an address. The bias tiles now live in their own
64 KB window at `0xBF600000`, uploaded once per GEMM, with the tile loops swapped
so `n0` is outer and `REG_B_ADDR` is set once per column-tile.

The window was sized from the **whole** pipeline, not the detector — MiniLM's
feed-forward expansion is N=1536, NT=192, needing 49,152 bytes. Three
compile-time assertions now make a bad window a build error rather than a board
failure. Validated on hardware by `team43_syscost_opt1c`, which runs the real
array with the bias seeded from the new window and checks all 64 outputs against
a software golden at both ends of the window: **5/5 cases, 64/64 match.**

---

## 10. Verification

### 10.1 Four independent implementations

The ranking pipeline is implemented **four times** — in NumPy from the deployed
weights, in C emulating the accelerator on a PC, in C emulating it on the board,
and in C driving the real accelerator. **All four agree to every printed digit.**
That agreement is what makes any hardware number here trustworthy rather than
merely reported.

Detection has three further independent checks:

- The flattened 103-op graph, re-implemented in NumPy from the fused weights,
  matches PyTorch's forward pass to **1.8 × 10⁻⁷** on class scores and
  **1.8 × 10⁻⁴ pixels** on box coordinates.
- INT8 quantisation tracks FP32 closely at every resolution; the complete INT8
  pipeline costs **+0.0006** on the top score and changes no rankings.
- The C implementation matches the NumPy INT8 pipeline to ≈ **0.005** in
  confidence, the residual being single- versus double-precision accumulation.

### 10.2 Milestones

Development ran as four milestones, each built, deployed and run on the physical
board — not in simulation — before the next was attempted, each backed by a
captured terminal log.

| | milestone | result on hardware |
|---|---|---|
| A | Plumbing: one real MiniLM layer-0 query projection | 3,072 / 3,072 exact |
| B | Complete MiniLM task→object scoring over 80 classes | wine glass 0.970027 |
| C | YOLOv8n detection end to end | detections produced on board; C vs NumPy INT8 agreement ≈ 0.005 confidence (§10.1) |
| D | Full integrated pipeline | PASS |

### 10.3 Two bugs that passed every test we had

Worth stating at a verification conference, because both produced correct output
for months.

**`Y_DONE` was never cleared.** The completion flag was read but never reset, so
after the first tile it stayed high and the "poll" was really one system call of
delay — about 100 µs. Correct *by accident*. `team43_kdepth` measured the
accident failing: with the whole K streamed in one invocation the loose wait is
exact to K=256 and **wrong from K=512** (max error 127,452 at K=512, 169,893 at
K=1024). Invisible until we made invocations longer.

**The host and board disagreed in the last bits.** See §8.3. The FMA contraction
issue made our own golden reference wrong.

Neither would have been caught by comparing the system against itself.

### 10.4 The trailing START clear

The trailing `rw(REG_START, 0u)` is retained deliberately. Experiment showed
three chained tiles compute correctly without it — but only because each was
separated by ~270 µs of unrelated traffic. In the tight optimised loop the next
register write follows within 65 µs, and without the clear the done bit never
de-asserts, so the poll returns a stale completion and the **previous** tile's
result is read.

### 10.5 Rejected: batched register writes

A naive first probe suggested several registers could be set in one `pwrite`.
`team43_regstress` disproved it under a real GEMM pattern: **one `pwrite`
covering several register words sets at most the first.** Five single-word writes
per tile is the floor. This is kept in the code behind
`--unsafe-batched-regs` for timing comparison only, clearly labelled as producing
wrong results.

### 10.6 Before silicon — the Stage 3A verification

The accelerator reached hardware without an RTL change because it had already
been verified against the same SoC in simulation. That work is the reason
milestone A passed first time, and it is worth stating explicitly at a
verification conference.

**Self-checking firmware.** A bare-metal test program ran on the simulated VEGA
core, configured the accelerator over AXI-Lite, waited for completion, then
**recomputed every output in software and compared all of them**. It emitted one
character per phase on the UART — `B` booted, `C` configured, `S` started, `P`
passed. Success is the exact string `BCSP`; a failure emits `F` instead, and the
completion poll is bounded so a hang still produces a marker rather than silence.

**An independent golden model.** The expected result was generated in Python,
written separately from both the firmware and the RTL, so a shared
misunderstanding between the two could not pass unnoticed. Two entries were
additionally computed by hand as a check on the checker.

**Transaction-level waveform analysis.** A VCD analyser reconstructed every AXI
transaction from first principles — queuing AW and W independently, pairing AR
with R by transaction ID, and sampling each channel at its own `VALID && READY`
edge rather than at the address handshake. It verified:

- every K, X and bias read beat against the expected memory packing
- the raw array output against `m_y_tdata` against the S2MM write stream —
  three independent observations of the same values
- exactly the expected number of result words, all matching the golden model
- the UART line decoded from one bit and required to read exactly `BCSP`
- **22 AXI protocol properties** — `RRESP`/`BRESP` OKAY, `WSTRB`, `WLAST`,
  `RLAST`, burst type and size, ID pairing, no duplicate or unexpected
  addresses, no access outside the declared buffers

The analyser exits non-zero on any single failure, so the pass banner could not
be produced by a partially working design.

**Test-vector design.** Operands were generated from closed-form expressions
rather than shipped as data, so an evaluator could reproduce the expected answer
from three lines of arithmetic. Both operands span negative and positive values,
which exercises two's-complement handling throughout the datapath. Fault
sensitivity was measured rather than assumed: a transposed activation matrix
changes 64/64 outputs, a dropped bias 60/64, an unsigned misread 56/64.

**Cross-version validation.** The design was exercised under **three Vivado
releases** — 2018.1, 2025.2 and 2026.1 — on both Linux and Windows. Post-synthesis
utilisation agreed to within 0.2 % across a seven-year tool gap, and the block-RAM
count reconciled exactly once the out-of-context ROM synthesis was accounted for.

**One finding worth repeating.** `rom_32KB_axi.mif` is not a static file — it is
an *output product* of the ROM IP, regenerated from a COE whenever the IP is
rebuilt. Newer Vivado leaves the 2018.1-authored IP locked and the file survives;
the organisers' own 2018.1 regenerates it and silently replaces the firmware with
their default bootloader. The submission therefore carries the program in **both**
the MIF and the COE, so either path boots the right image. This was caught by a
hash check in the run script, not by a failing simulation — the design would have
appeared broken on the evaluators' machine while being entirely correct.

---

## 11. Performance

> **Read the configuration on every number.** Latency depends on model, canvas
> size, `--rect`, image, and board state. Numbers from different configurations
> are not comparable.

### 11.1 Where the time goes

YOLOv8n, 256 px aspect-fitted, `live_image.jpg`, healthy board:

| stage | seconds | share |
|---|---|---|
| Detection — YOLOv8n INT8 | 94.8 | 73 % |
| MiniLM encoder — 6 layers | 16.5 | 13 % |
| JPEG decode on the board | 14.0 | 11 % |
| **total** | **130.6** | |

Inside detection: GEMM loop 51.0 s · VEGA float 51.5 s · packing/im2col 18.0 s ·
DDR upload 14.0 s. 28,308 accelerator invocations.

**39 % of the query is software floating point.** There is no FPU on this core,
so every float operation is a library call — dequantise-and-activate costs more
than the accelerator it feeds.

### 11.2 Measured device primitives

`team43_syscost_opt1c`, mean over 200 calls, array idle:

| access | cost |
|---|---|
| single register write (1 word) | 85.6 µs |
| single register read (1 word) | 80.4 µs |
| tile write (64 words, 256 B) | 284.8 µs |
| tile read (64 words, 256 B) | 296.8 µs |
| bulk write (256 words, 1 KB) | 585.2 µs |

Cost is **per call, not per byte**. This measurement is what motivated the final
kernel design, in which all operands are uploaded once in large bursts and the
inner loop performs only register writes.

### 11.3 The three timescales

For one 8×8×8 tile:

| | |
|---|---|
| fabric time (8 beats + 17-cycle fill/drain) | ≈ **0.5 µs** |
| on-chip invocation | ≈ **55 µs** |
| measured wall clock per output tile | ≈ **1,800 µs** |

**The array does its work in half a microsecond and then waits about 1.8
milliseconds for software to come back.** Every layer of the data path is
efficient — 64-bit beats mapping one-to-one onto array steps, adapters that
vanish in synthesis, two-deep skid buffers with no wasted storage. The loss is
entirely at the driver boundary, and secondarily in the burst length of 1.

### 11.4 Accelerator versus software — the controlled A/B

Same board, same binary, same image, same task, same flags. **Only `--sim`
differs.** `nki.jpg`, "extinguish fire", 256 px aspect-fitted, `--silu-table`.

| stage | with array | software only | gain |
|---|---|---|---|
| GEMM loop | 38.5 s | 198.9 s | **5.2×** |
| Detection total | 63.2 s | 194.1 s | 3.1× |
| MiniLM encoder | 16.5 s | 46.0 s | 2.8× |
| **End to end** | **87.5 s** | **247.9 s** | **2.8×** |

Both runs selected `bottle` at rank **0.25624** with the same `fpcheck`, so the
comparison is exact rather than approximate.

**Why 5.2× becomes 2.8×.** Packing 12.2 s, software floating point 29.7 s and
JPEG decode 5.2 s are identical in both runs — the array cannot touch them. About
47 s of every query is work outside its reach. The array path additionally pays
12.4 s of DDR upload that software emulation never pays, and still wins.

### 11.5 Why 8×8 and not 16×16

A question worth being able to answer directly:

1. **The array was never the bottleneck.** A tile's arithmetic is ~0.16 µs
   against ~1,800 µs of wall clock — the multipliers are idle more than 99 % of
   the time. A bigger array does not make a driver round trip faster.
2. **A wider array helps only indirectly**, by needing four times fewer round
   trips. Real, but bounded.
3. **Amdahl caps it.** With an infinitely fast array, 87.5 s only becomes ~49 s.
4. **Fabric was not the binding constraint.** 16×16 ≈ 27,600 LUTs would take the
   design from 53 % to ~63 % and fits; 32×32 ≈ 110,000 LUTs does not. The route
   past that is the 840 unused DSP slices, not more LUTs.

The right order of attack is: cut system calls per tile → increase AXI burst
length → attack the 47 s the array cannot reach → *then* widen the array.

---

## 12. Deployment

### 12.1 Payload verification — do this first

At 94 KB/s, a silently truncated transfer is a standing risk. We shipped a tar in
which three of fourteen JPEGs were short prefixes of the real files — no EOI
marker, bottom 3–22 % of each picture missing. Nothing complains: the pipeline
prints a correct-looking `source: 640 x 513` and produces a marked image with the
bottom of the frame quietly garbage.

`team43_deploy.tar` therefore carries `SHA256SUMS.txt`, `MD5SUMS.txt` and:

```bash
cd /tmp/team43 && ./verify.sh
# PAYLOAD OK -- 19 files match
```

It falls back from `sha256sum` to `md5sum` to a size check depending on what the
board has. **Do not run the pipeline until it passes.**

### 12.2 Programming from Windows

`ether` is a Linux x86-64 ELF that opens an `AF_PACKET`/`SOCK_RAW` socket and
writes raw Ethernet frames to the MAC in `env.txt`. Windows has no equivalent
socket, so `send.ps1` drives a Linux kernel rather than replacing it:

- **`-Mode wsl`** (default) — the interface is a USB Ethernet adapter, so
  `usbipd-win` can hand it to WSL2, which then sees a real NIC with full
  raw-socket access. One-time: `wsl --install -d Ubuntu`,
  `apt install device-tree-compiler`, `winget install dorssel.usbipd-win`. Per
  plug-in: `usbipd bind` then `usbipd attach --wsl`.
- **`-Mode remote`** — `scp` the four files to a Linux machine and run there.
  Needs nothing installed; Windows 10/11 ship `ssh.exe` and `scp.exe`.

`send.bat -WhatIf` prints every command and touches nothing. The script echoes
the MAC, entrypoint and DTB address from `env.txt` before sending, because a
wrong MAC is the nasty failure: frames go out, the board ignores them, nothing
errors, and you get silence on the serial console.

### 12.3 Memory map

From `FDC.dts`:

```
memory@80000000     reg = <0x0 0x80000000 0x0 0x2F000000>   -> Linux owns 0x80000000..0xAF000000
framebuffer@20000   reg = <0x0 0xbf000000 0x0 0x1000000>    -> 0xBF000000..0xC0000000, 16 MB
```

Linux's allocator never sees anything above `0xAF000000`. The scratch windows
live in the declared 16 MB above it:

| window | range | size |
|---|---|---|
| K | `0xBF000000` … `0xBF200000` | 2 MB |
| X | `0xBF200000` … `0xBF400000` | 2 MB |
| Y | `0xBF400000` … `0xBF600000` | 2 MB (256 B used) |
| B (opt1c) | `0xBF600000` … `0xBF610000` | 64 KB |
| free | `0xBF610000` … `0xC0000000` | ~10 MB |

The windows were 1 MB each until YOLOv8s arrived with a 1,179,648-byte weight
block, which would have overrun the K window into the activation window — silent
corruption, no error. Caught before any board run; windows enlarged to 2 MB with
a startup assertion that prints the largest weight block and dies if it does not
fit.

---

## 13. Accuracy

Measured with the reference work's metric: average precision at IoU 0.5 against
the preferred-object annotations, per task, averaged over the fourteen tasks. The
suitability probabilities are read from the **deployed INT8 package**, so the
evaluation reflects what the board actually computes. Confidence is detection
score × suitability, per the reference protocol.

| system | top-1 | mAP@0.5 |
|---|---|---|
| Sawatzky et al. 2019 — Faster R-CNN (GPU) | 0.640 | 0.409 |
| **Ours — YOLOv8s INT8, 256 px** | **0.627** | **0.425** |
| Ours — YOLOv8n INT8, 192 px | 0.311 | 0.201 |

**Ranking stage in isolation:**

| protocol | mAP@0.5 |
|---|---|
| Ground-truth boxes | **0.648** |
| The paper's own Faster R-CNN detections | **0.288** |
| — their ResNet-101 classification baseline | 0.288 |
| — their full Gated Graph Neural Network | 0.327 |
| — their no-visual-input ablation | 0.237 |

On identical detections the ranker equals their ResNet-101 baseline and reaches
88 % of their full GGNN — **using a ranker that never sees a pixel.** It exceeds
their no-visual-input ablation by 0.05, most likely because that ablation encodes
the object class as a one-hot identifier whereas we encode it as language: in a
sentence-embedding space "wine glass" and "cup" are neighbours in a way that
category indices 46 and 47 are not, and that neighbourhood is exactly the
fall-back behaviour the paper describes.

### 13.1 Where the errors live

Of the 80 % top-1 miss, the **detector loses 72.1 points and the ranker only
8.1**. Given ground-truth boxes the ranker is right 94.7 % of the time.

**So the lever is resolution.** Language understanding is not the bottleneck;
finding small objects on a 256 px canvas is. That is why latency and accuracy
trade against each other in this system.

### 13.2 Confidence in these figures

The AP implementation was validated by reproducing a published baseline. The
reference paper's pick-best-class baseline, re-implemented, yields 0.339 against
their published 0.386 — close enough to establish the metric is in the right
regime, not close enough to claim exactness. **Absolute figures carry about
±0.05.** Comparisons on identical detections are firmer, since the same code
scores both sides.

Tie ordering contributes ±0.003: the ranker gives every instance of a class the
same score, so AP depends slightly on tie order. Across four randomised orderings
the ground-truth result varied 0.6464–0.6493; the conservative mean 0.648 is
quoted rather than the 0.658 from natural annotation order.

---

## 14. Limitations

**Latency.** A query is seconds, not milliseconds. This is a functional
demonstration of an edge pipeline, not a real-time one. The cause is measured and
understood (§11), and the driver is outside the permitted scope to modify.

**No instance-level discrimination.** The ranker scores a task against a *class
name*, so every detected instance of a class receives the same suitability score
and ties break by detection confidence. The reference paper's Figure 3 turns on
exactly the cases this cannot address — the handier chair, one bed over another,
the real toilet rather than its reflection. Resolving them needs per-instance
visual features, which this architecture deliberately excludes in order to keep
the label embeddings precomputable.

**Detection recall at reduced resolution.** At 128 px many preferred objects are
not detected at all, capping end-to-end AP regardless of ranking quality. Of
sixteen COCO photographs tested, only six produced any detection at 128 px.

**Task coverage.** Fourteen prompts, no others.

**Accelerator utilisation.** The interface dominates, so the array is idle most
of each invocation.

**The honest failure case.** Task 9, "open parcel": the pipeline picks a fork,
ground truth says knife. It scores 0.000 on both detectors. We kept it in the
evidence set rather than replacing the image.

### 14.1 Board state — the open item

The same binary on the same board has run at 130 s and at 4–20× slower, depending
on how the FPGA came up. Per-element rates, which are independent of image and
model:

| pass | healthy | degraded |
|---|---|---|
| amax scan | 1.00 µs/elem | 7.64 |
| quantise convert | 3.37 µs/elem | 17.83 |
| dequant + silu | 14.65 µs/elem | 67.12 |
| JPEG IDCT | 115 µs/block | 1,989 |
| YCbCr → RGB | 2.6 µs/px | 52.6 |

Three things make this diagnosable rather than mysterious:

- **It is selective.** In one degraded state `memcpy` was healthy at 7.84 MB/s
  while a strided scan collapsed to 0.13 M elem/s — 46× down. Linear streaming
  survives; scattered access does not. That is a cache and DRAM-latency
  signature, not a bandwidth one.
- **The accelerator is not implicated.** GEMM cost held at 1.79–1.80 ms per
  output tile across healthy runs of both models.
- **The outputs never change.** Bit-identical images, same selection, same rank
  to five decimals, same `fpcheck`. **It is a timing fault, not a logic fault.**

`team43_memcheck` measures it in five seconds and should be step one of every
deployment. Note the honest scope: what is measured is variation
**programming-to-programming on one board**, plus a 3.7× swing between two probe
runs minutes apart in one session — not a comparison of two physical boards.

---

## 15. Future work

In priority order, which is also the order the measurements justify:

1. **Cut system calls per tile.** One is already removed (§9.2).
2. **Increase AXI burst length** past 1 (§7.5) — the single RTL constant with the
   most leverage on the streaming side.
3. **Batched tile commands** — a descriptor list letting many tiles issue per
   driver transaction would amortise the call cost directly.
4. **Attack the 47 s the array cannot reach** — im2col, quantisation,
   activations, and the absent FPU.
5. **Stabilise the VEGA cache and memory-latency path**, and bound working sets
   in software so a degraded cache costs less.
6. **Then** widen the array — 16×16 in LUTs, or map the MACs onto the 840 unused
   DSP slices.
7. **Raise the detector resolution**, which is worth more accuracy than any
   remaining software saving (§13.1).
8. **Evaluate over the complete 14-task validation set** at 320 and 640 px, to
   separate detector recall from ranking quality.

---

## 16. File inventory

> **Four companion guides walk through the code in this inventory** — what each
> file does, why it is shaped the way it is, and which lines you would break by
> tidying them.
>
> | guide | covers |
> |---|---|
> | `CODE_GUIDE_RTL.md` | every RTL module, and the six things in it that look like cruft and are not |
> | `CODE_GUIDE_FIRMWARE.md` | `team43_pipeline.c` stage by stage — the deep dive behind `firmware_study/README.md` |
> | `CODE_GUIDE_PROBES.md` | the six hardware probes, including the one that gave the wrong answer |
> | `CODE_GUIDE_TOOLCHAIN.md` | the host export + evaluation scripts, and the legality argument for each |

### Firmware

| file | what |
|---|---|
| `team43_pipeline.c` | the entire firmware, ~1,955 lines |
| `jpeg.h` | baseline JPEG decoder, integer islow IDCT |
| `font8x8.h` | 8×8 bitmap font for the result label |
| `team43_yolo_graph.h` | detector as a flat op list (export tool output) |
| `team43_minilm_layout.h` | MiniLM tensor offsets in the package |

### Model packages

| file | bytes |
|---|---|
| `team43_yolo_int8.bin` (v8n) | 3,192,064 |
| `team43_yolo_s_int8.bin` (v8s) | 11,229,824 |
| `team43_minilm_int8.bin` | 23,088,004 |
| `team43_minilm_packed.bin` | 11,211,008 |
| `vocab.txt` | 262,030 |

### Probes — each answers one hardware question

> `CODE_GUIDE_PROBES.md` documents all six: the question each was written to
> answer, the method, the discipline that keeps them from fooling themselves, and
> the one that did anyway.

| probe | question |
|---|---|
| `team43_kdepth.c` | can one invocation accumulate all of K? **Yes — 3.3×** |
| `team43_regstress.c` | can several registers be set in one write? **No** |
| `team43_regbatch.c` | the first, inadequate version of the above |
| `team43_memcheck.c` | is this board's memory healthy? 4 MB working set, no device needed |
| `team43_im2col.c` | does blocking the im2col write set help? hashes every variant |
| `team43_syscost_opt1c.c` | what does each device access cost, and does the B window work? **⚠ not present in this tree — see §21** |

### RTL

> **`CODE_GUIDE_RTL.md` walks through every module below** — what each does, why
> it is shaped the way it is, and the six things in it that look like cruft and
> are not. `firmware_study/README.md` does the same for the firmware.

| path | what |
|---|---|
| `rtl/sa/axis_sa.sv` | the 8×8 array, tri_buffer skew, diagonal drain |
| `rtl/sa/pe.sv`, `mac.sv` | processing element and MAC cell |
| `rtl/sa/tri_buffer.sv`, `n_delay.sv` | operand skew and control pipelining |
| `rtl/sys/dma_controller.sv` | register file, descriptor generation |
| `rtl/sys/skid_buffer.sv` | 2-deep valid/ready decoupler |
| `rtl/sys/top_axi_int.v` | AXI integration, adapters, bias adders |
| `rtl/sys/config.svh` | R, C, widths, `AXI_MAX_BURST_LEN` |
| `rtl/ext/` | crossbar, AXI-Lite, DMA engines, stream adapters |
| `Accelerator_Top.v` | the module `Top.vhd` declared but did not supply |

### Scripts

`build_fast.sh`, `run_n_opt1c.sh`, `verify.sh`, `send.ps1` / `send.bat`,
`revidence` / `revidence_s256` (14-task evidence sweeps), `rcold_s256` (timed
cold start), `r128` / `r192` / `r320` / `rrank`.

---

## 17. Five things that cost days to learn

Everything below is commented in place in the source. They are the parts that
look like cruft and are not.

1. **The k0 loop does not have to exist.** The array ends an accumulation when
   the DMA asserts last, and the transfer length is just a register — so K per
   invocation is a software choice. 713 s → 218 s.
2. **Multi-word register writes do not work.** One `pwrite` covering several
   register words sets at most the first. Five single-word writes per tile is the
   floor.
3. **`Y_DONE` must be cleared before every START.** It used to be only read, so
   the "poll" was really one syscall of delay — correct by accident, and that
   accident fails once K is large.
4. **Both builds must use `-ffp-contract=off`.** Otherwise the board fuses
   `a*b+c` and the host does not, and your golden reference is quietly wrong.
5. **Cost is memory accesses, not arithmetic.** A tile's arithmetic is ~0.16 µs
   against ~1,800 µs of wall clock per **output tile** (≈485 µs per accelerator
   *invocation*, of which the array itself is ~0.5 µs — §11.3). This is why
   widening the array was analysed and rejected, and why every optimisation that
   helped removed *accesses* rather than *operations*.

---

## 18. Reproducing this

Reproduction requires three organiser-held files not included in the package:
the buildroot toolchain, the `buildroot-2004_new` Docker image, and — for
regenerating the bitstream rather than loading the supplied one — the VEGA core
netlist `AS1061_SYSTEM_TOP.edn`.

`FILE_MANIFEST.md` records every file with size, purpose and origin.
`SHA256SUMS.txt` provides checksums. `QUICK_START.md` gives build, deployment and
execution commands. `STATUS.md` records every outstanding item.

---

---

## 19. Requirement compliance

The contest brief for the hardware stage set specific conditions. Each is mapped
here to the evidence that satisfies it, so a reader can check rather than trust.

| # | requirement | status | evidence |
|---|---|---|---|
| 1 | Image supplied at runtime | ✅ | `--image`, decoded on board (§8.2) |
| 2 | Text task supplied at runtime | ✅ | `--task`, tokenised on board (§4.2) |
| 3 | Same binary handles new inputs without rebuild | ✅ | graph shapes from header, canvas is an argument (§3.3) |
| 4 | Detector executes on the board | ✅ | 63 convolutions, 28,308 invocations (§11.1) |
| 5 | Task scoring executes on the board | ✅ | 6 encoder layers + trained head (§4.1) |
| 6 | Accelerator performs real inference consumed by the answer | ✅ | `head.0`/`head.3` run on the array; the selection is their output (§6.1) |
| 7 | Processor performs unsupported operations | ✅ | partition table (§4.1) |
| 8 | Candidate objects produced | ✅ | 74 detections in the §1 example |
| 9 | Best task-suited object selected | ✅ | ranked output (§1) |
| 10 | Label, score and bounding box printed | ✅ | run output; box drawn into `result.ppm` |
| 11 | Annotated result image | ✅ | `--out` |
| 12 | Matches an independent reference | ✅ | four implementations agree (§10.1) |
| 13 | Repeatable across runs | ✅ | bit-identical outputs, same `fpcheck` |
| 14 | Source, binaries, hashes, logs preserved | ⚠ | see §20.1 |
| 15 | Report states what ran where | ✅ | §4.1, §8.1 |
| — | Host performs no runtime inference | ✅ | boundary defined and audited (§8.1) |
| — | Fits the device | ✅ | 5.86 % accelerator, 52.96 % SoC (§7.6) |

The one caveat worth stating plainly: **the report claims no result that was not
measured on the physical board**, and where something is predicted rather than
measured — the two pending A/B rows in §9 — it is labelled as pending.

---

## 20. Provenance

### 20.1 Documents referenced but not present

§18 names `FILE_MANIFEST.md`, `QUICK_START.md`, `STATUS.md` and
`SHA256SUMS.txt`. **Only `app/TEAM43_FROZEN/yolov8n/SHA256SUMS.txt` exists in
this tree.** Either generate the other three before release, or remove the
references — a reviewer who follows a dead pointer discounts everything around
it. This is the sole item preventing compliance row 14 from being a clean pass.

### 20.2 Tool and model versions

| component | version |
|---|---|
| Vivado (implementation of record) | 2020.2 |
| Vivado (cross-checked) | 2018.1, 2025.2, 2026.1 |
| Board toolchain | `riscv64-buildroot-linux-gnu-gcc`, buildroot-2004 container |
| Board kernel | Linux 4.18.0, `rv64imafdc`, sv39 |
| Board RAM | 684,864 kB; `/tmp` a 334 MB RAM disk |
| Host export | PyTorch 2.10.0, transformers 5.5.4 |
| Detector | Ultralytics YOLOv8n / YOLOv8s, BatchNorm folded, per-channel INT8 |
| Ranker | `all-MiniLM-L6-v2` fine-tuned, epoch 18, val_acc **0.9603** |
| Dataset | COCO-Tasks (Sawatzky et al., CVPR 2019), 14 tasks |

### 20.3 Reading the numbers

Three distinctions are load-bearing and easy to conflate:

- **Post-synthesis vs post-route.** The accelerator measures 12,404 LUTs after
  synthesis and **11,945 after routing** (§7.6). The routed figure is the real
  one; optimisation during implementation accounts for the difference.
- **Command-to-DONE vs wall clock.** The array finishes a tile in ≈ 0.5 µs of
  fabric time but ≈ 1,800 µs of wall clock elapses per output tile (§11.3). Only
  the latter is user-visible, and the gap is the driver boundary, not the design.
- **512 MACs per invocation**, not 64. One invocation is a full 8×8×8 tile
  product. Dividing by 64 overstates the workload eightfold.

### 20.4 Honest summary of the contribution

What is genuinely new here is not the detector or the language model — both are
published architectures — but the demonstration that **task-conditioned object
selection runs end to end on a 50 MHz soft RISC-V core with a 12 K-LUT
accelerator and no DSP or block RAM**, at accuracy competitive with the GPU
reference on the same benchmark (§13), with every claim backed by a measurement
and every failure kept in the evidence set rather than removed from it.

---

## 21. Known gaps between this document and the tree

Found while writing the four code guides. Neither is a fault in running code; both
are places where this README describes something the package does not contain, and
both should be resolved before the document is read as a record of what shipped.

### 21.1 `team43_syscost_opt1c.c` is cited but not present

It is named in §8 (the per-access syscall cost table, *"mean over 200 calls, array
idle"*), in §9.2 (*"5/5 cases, 64/64 match"* — the only evidence offered for the
bias-window correctness claim) and in §16's probe table; `run_n_opt1c.sh` is
likewise in §16's scripts line. Neither file exists anywhere under `final_phase` —
not in `./`, `app/`, `app/TEAM43_FROZEN/`, `eval/`, `eval_s/`,
`firmware_study/probes/` or `firmware_study/src/`.

Preferred resolution: recover the probe and place it in `firmware_study/probes/`
beside the other five. Otherwise mark it explicitly as not retained at each of the
three citation sites, so a reader who goes looking is not left thinking the package
is incomplete.

### 21.2 The bias-window optimisation (opt1c) is documented but not in the source

§9.2 describes it in the present tense — *"The bias tiles now live in their own
64 KB window at `0xBF600000`, uploaded once per GEMM, with the tile loops swapped so
`n0` is outer and `REG_B_ADDR` is set once per column-tile"* — and §14's memory map
lists a `B (opt1c)` row at `0xBF600000 … 0xBF610000`.

Every copy of `team43_pipeline.c` in this tree still has, in `gemm_run`:

```c
rw(REG_B_ADDR, (uint32_t)Y_ADDR); rw(REG_B_BYTES, OUTPUT_BYTES);
...
for (int m0 = 0; m0 < MT; m0++)
    for (int n0 = 0; n0 < NT; n0++) {
        ... seed par from bias ...
        if (!g_sim) dwr(Y_ADDR, par, OUTPUT_WORDS);
```

B still points at `Y_ADDR`, `m0` is still the outer loop, and the bias tile is still
re-uploaded per output tile. `0xBF600000` appears nowhere outside this README. The
shipped firmware is therefore the **pre-optimisation** version, and §9.2, §14 and
§15's *"One is already removed (§9.2)"* currently overstate it.

21.1 and 21.2 are almost certainly the same gap — an optimisation branch and its
probe that live somewhere other than this tree. Recovering that branch resolves
both.

---

*Reference: A. Sawatzky, Y. Souri, C. Grund, J. Gall. "What Object Should I Use?
— Task Driven Object Detection." CVPR 2019.*
