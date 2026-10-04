# 03 — Stage 1: the proposal (February – March 2026)

**Goal of the stage:** understand the problem and the platform, then submit a written proposal in the organisers' template (Abstract, Introduction, Background Research, Goals, Design Process, Results, Conclusion, References).

**Outcome:** proposal shortlisted on **1 April 2026** → Stage 2.

Files from this stage in the repository:

| File | What it is |
|---|---|
| [`reports/stage1_background_research.md`](reports/stage1_background_research.md) | Our background-research section: VEGA AS1061, Genesys 2, COCO-Tasks, YOLOv8n, MobileCLIP |
| [`reports/stage1_systolic_array_design_note.pdf`](reports/stage1_systolic_array_design_note.pdf) | Our 9-page design note on the matrix-multiply accelerator (full systolic array vs tiled design, Cannon's algorithm) |

---

## 1. Understanding the platform first

Before proposing anything, we read the C-DAC VEGA AS1061 datasheet, the Genesys 2 reference manual and the Xilinx 7-series overview, and wrote down the numbers that limit a design:

- The CPU is a **single, in-order RISC-V core at 50 MHz**. That clock is about 1/60 of a typical 3 GHz laptop processor, and the core does one thing at a time, so running a neural network purely in software would take a very long time.
- The FPGA (Kintex-7 XC7K325T) has 840 DSP slices and ~16 Mbit of block RAM, but the VEGA SoC already uses about half of the logic.
- There is 1 GB of DDR3 — enough for model weights, but every byte the accelerator reads has to come over the AXI bus.

**Conclusion:** the CPU should *orchestrate*, and the heavy maths should go to a custom accelerator in the FPGA fabric.

## 2. Profiling the models

We profiled our candidate models in PyTorch (`torch.profiler`):

- **YOLOv8n** — 3.2 M parameters, 8.7 GFLOPs at 640 px, CSPDarknet backbone with C2f blocks, PAN-FPN neck, SPPF, anchor-free head.
- **MobileCLIP** (Apple) — an efficient image–text model (S0/S1/S2 variants).

In both, **convolutions and matrix multiplies take most of the time**. A convolution can be rewritten as a matrix multiply (a trick called *im2col*: copy the image patches into the rows of a matrix). So one well-designed **matrix-multiply accelerator** can speed up almost everything.

## 3. The first pipeline idea

```
photo ──► YOLOv8n ──► crop every detected object
                          │
task text ──► MobileCLIP text encoder
                          │
                          ▼
         compare each crop's image embedding with the text embedding
                          │
                          ▼
           pick the most similar crop  →  best object, box, score
```

Inputs: an RGB image + a task sentence. Outputs: the best object, its box, a similarity score. Constraints: Genesys 2, VEGA AS1061, the accelerator must fit in the remaining FPGA resources, low latency.

> In Stage 2 we changed both models (YOLOv8n → YOLOv8x for the software stage, MobileCLIP → MiniLM). In the hardware stages we went back to YOLOv8n/YOLOv8s because of the 50 MHz CPU. See [04](04_stage2a_software_pipeline.md) and [07](07_stage3b_remote_hardware.md).

## 4. The accelerator study

Our design note compares two ways to build a matrix multiplier in hardware:

| Approach | Processing elements (PEs) | Latency for N×N | Comment |
|---|---|---|---|
| **Full systolic array** | N² | O(N) | Fastest; uses more fabric |
| **Tiled (blocked)** | N²/B² | O(N·B²) | Re-uses a small block of hardware many times; slower |

We chose the **systolic-array** direction because the main aim was performance.

**How a systolic array works (plain words).** Picture an 8×8 grid of small calculators (PEs). Every clock tick, each PE takes a number from its left neighbour and one from its top neighbour, multiplies them, adds the product to its own running total, and passes the two numbers on (right and down). The inputs are fed in with a small delay per row and per column (a *skew*), so the right pairs of numbers meet in the right PE at the right time. After the data has flowed through, each PE holds one element of the result matrix.

The note also studied **Cannon's algorithm**, where the grid wraps around like a torus and blocks of A and B are first rotated and then shifted step by step. In the end (Stage 2B) we used an output-stationary systolic array with skewed inputs, adapted from an existing open-source design, because it came with ready AXI-stream and DMA interfaces — see [05](05_stage2b_accelerator.md).

## 5. References we used in Stage 1

- C-DAC, *VEGA Processors* and *VEGA AS1061 datasheet* — [vegaprocessors.in](https://www.vegaprocessors.in/)
- A. Waterman, K. Asanović (eds.), *The RISC-V Instruction Set Manual*, Vol. I (Unprivileged) and Vol. II (Privileged)
- Digilent, *Genesys 2 Reference Manual*
- AMD/Xilinx, *7 Series FPGAs Overview* (DS180); *UltraScale DSP48E2 slice* (UG579)
- J. Sawatzky et al., *What Object Should I Use? — Task Driven Object Detection*, CVPR 2019
- M. Yaseen, *What is YOLOv8: An In-Depth Exploration of the Internal Features of the Next-Generation Object Detector*, [arXiv:2408.15857](https://arxiv.org/abs/2408.15857)
- H.-C. Lu, L.-Y. Su, S.-H. Huang, *Highly Fault-Tolerant Systolic-Array-Based Matrix Multiplication*, Electronics 13(9):1780, 2024 — [doi:10.3390/electronics13091780](https://doi.org/10.3390/electronics13091780)
- Z. Li, S. Chen, *Design and Implementation of an FPGA-Based Tiled Matrix Multiplication Accelerator for Transformer Self-Attention on the Xilinx KV260 SoM*, [arXiv:2503.16731](https://arxiv.org/abs/2503.16731)
- Wikipedia, [*Systolic array*](https://en.wikipedia.org/wiki/Systolic_array)

All references for the whole project are collected in [12 — References](12_references.md).

## 6. Tips if you are writing a Stage 1 proposal

- **Profile before you propose.** Real numbers ("convolutions are X % of the time") make the proposal believable and point you to what to accelerate.
- **Write down the platform limits** (clock, memory, free fabric) on day one. They decide which models are even possible.
- **Keep the first idea flexible.** Ours changed twice. What matters in Stage 1 is a sound direction and a clear plan.
- **Mind the template.** The organisers asked for the template format to be followed strictly.
