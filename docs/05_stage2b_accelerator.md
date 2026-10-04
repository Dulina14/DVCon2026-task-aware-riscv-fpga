# 05 — Stage 2B: the accelerator on its own (9 – 30 June 2026)

**Goal of the stage (from the organisers):** *"Develop accelerator and interface for the software solution developed."* Deliverables: a working design, integration readiness, an **AXI interface**, a **testbench**, a makefile (Vivado/Questa), a README and an implementation report.

**Outcome:** qualified for Stage 3A on **9 July 2026**.

Our Stage 2B notes are in [`docs/stage2b/`](stage2b/README.md).

---

## 1. Choosing a starting point

Writing a correct AXI DMA from nothing is a project of its own. So we started from an open-source, MIT-licensed design that already had the three things we needed:

> **[`axis-systolic-array`](https://github.com/abarajithan11/axis-systolic-array) by Abarajithan G** — *"a lightweight, highly parameterizable systolic array in SystemVerilog, integrated with an Ibex SoC via AXI DMAs, a custom DMA controller, and corresponding C firmware."* It in turn uses Alex Forencich's MIT-licensed [`verilog-axi`](https://github.com/alexforencich/verilog-axi) / [`verilog-axis`](https://github.com/alexforencich/verilog-axis) DMA engines, crossbar and adapters.

What we did with it in this stage:

- studied it until we could explain every module (our reading guide is [`code_guides/CODE_GUIDE_RTL.md`](code_guides/CODE_GUIDE_RTL.md));
- set it up as a stand-alone Vivado project (`axis-systolic-array-new`) with a CPU-model testbench;
- wrote a speed-up model, a block-diagram specification and a waveform guide;
- kept the array size and bus widths as parameters, so they could be changed later for the VEGA SoC (64-bit AXI, 12-bit IDs — done in Stage 3A).

We list this origin clearly because good engineering re-uses good open-source IP — and because credit matters. See [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## 2. What the accelerator computes

```
Y = X · K + B
X : R × K_len   INT8   (activations)
K : K_len × C   INT8   (weights)
B : R × C       INT32  (bias, or the running partial sum)
Y : R × C       INT32  (result)
```

`R` and `C` (rows and columns of the processing-element grid) are parameters in `config.svh`. In Stage 2B we simulated **R = C = 2**; in Stage 3A we moved to **R = C = 8**.

## 3. The blocks inside `top_axi_int`

| Block | Module | Job |
|---|---|---|
| `AXIL2RAM` | `alex_axilite_ram` | Turns AXI-Lite register writes from the CPU into a simple register-file port |
| `CONTROLLER` | `dma_controller` | Holds the 16 registers; turns ADDR/BYTES into DMA descriptors; reports DONE |
| `mm2s_0/1/2` | `alex_axi_dma_rd` | Three read DMAs: fetch **K**, **X** and **B** from memory as streams |
| `s2mm` | `alex_axi_dma_wr` | One write DMA: stores **Y** back to memory |
| `ADAPTER_*` | `alex_axis_adapter_any` | Change stream width (e.g. bias 64→256 bits, result 256→64 bits) |
| `SKID_*` | `skid_buffer` | 2-deep pipeline registers on each stream so `valid` never depends on `ready` combinationally |
| `SA` | `axis_sa` | The systolic array: `X·K` |
| (glue) | in `top_axi_int` | 8 parallel 32-bit adders for `+ B`; the K and X streams are locked together (`s_valid = k_valid & x_valid`) |
| `CROSSBAR` | `axi_crossbar` | Merges the four DMA masters into the one AXI master port |

The *design connectivity spec* ([`stage2b/DESIGN_CONNECTIVITY.md`](stage2b/DESIGN_CONNECTIVITY.md)) lists every connection with its bus width so the block diagram can be drawn exactly.

## 4. The standalone testbench

```
tb_top_axi_int
 ├─ CPU model: AXI-Lite master tasks (axil_write / axil_read)
 ├─ DUT: top_axi_int  (the accelerator)
 └─ MEM: axi_ram_model (behavioural AXI4 RAM standing in for DDR)
```

1. Matrices are written straight into the RAM model (no bus traffic).
2. The CPU model programs the registers over AXI-Lite, writes START, and polls DONE.
3. The accelerator's DMAs read K, X, B and write Y on their own.
4. The testbench compares `Y_dut` with `Y_ref`.

Confirmed run in Vivado XSim (2×2, 10 ns clock):

```
[145 ns]  CPU: programming accelerator over AXI-Lite
[585 ns]  CPU: writing START
[1105 ns] S2MM_DONE asserted after 12 polls
          Y_dut = [29 42 ; 73 90] == Y_ref   -> SELF-CHECK PASS
```

How to read every transaction in that waveform, step by step, is in [`stage2b/WAVEFORM_GUIDE.md`](stage2b/WAVEFORM_GUIDE.md).

> The Stage 2B testbench source and Vivado project are not part of this repository (they lived in a separate working folder). The upstream repository contains equivalent testbenches (`tb/top_axi_tb.sv`) and a Python golden model, and our Stage 3A verification (in [`hardware/stage3a_simulation/`](../hardware/stage3a_simulation/)) is far stronger.

## 5. How much faster? — our first speed-up model

From [`stage2b/ACCELERATION.md`](stage2b/ACCELERATION.md):

```
Speed-up  ≈  (R × C)  ×  (CPU cycles per multiply-add)
```

| Array | PEs | Speed-up for a 64×64×64 matrix multiply vs a CPU doing 1 MAC/cycle |
|---|---|---|
| 2×2 | 4 | 3.7× |
| 8×8 | 64 | 50.6× |
| 16×16 | 256 | 169× |

We also noted honestly that at 2×2 there was **no real end-to-end speed-up**: START→DONE took 52 cycles for only 8 multiply-adds, because setting registers and moving data dominated.

**Looking back:** this model was right about the array and wrong about the system. On the real board (Stage 3B) the array finished a tile in about 0.5 µs, but each call took about 1.8 ms of wall-clock time, because every register access went through a Linux driver system call. The lesson — *measure the interface, not just the arithmetic* — is the main theme of [11 — Lessons learned](11_lessons_learned.md).

## 6. Tips for this kind of stage

- **Re-use proven AXI infrastructure** (Forencich's libraries are widely used) and spend your time on understanding and verification.
- **Write the waveform guide while the design is fresh.** It forced us to understand the `VALID && READY` handshake on every channel, which paid off in Stage 3A.
- **Keep the array size a parameter.** Going from 2×2 to 8×8 in Stage 3A was a one-line change in `config.svh`, plus re-checking widths.
