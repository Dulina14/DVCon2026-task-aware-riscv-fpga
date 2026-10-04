# Stage 3A — Technical Reference (8×8 Accelerator)

Detailed reference for the submitted design: configuration, packing, register map
and scalability. For how to run it see [`README.md`](README.md); for the full
narrative see `Document/DVCon2026_Stage3A_Report.pdf`.

**Status: simulation PASS, synthesis PASS — both verified and reproducible.**

---

## 1. Physical array vs. logical matrix

`R` and `C` set the **physical PE grid** — the output tile the hardware produces
per pass — not a maximum matrix size. An 8×8 array computes any GEMM by tiling;
the grid sets **throughput** (64 MACs/cycle), not capability. This test uses a
single 8×8×8 pass so one job maps exactly onto one physical tile, which makes
every intermediate value directly observable.

## 2. Configuration (`ACCELERATOR_IP/rtl/sys/config.svh`)

| Param | Value | Derived |
|---|---|---|
| R | **8** | `X_BUS_W = R×WX = 64 b` |
| C | **8** | `K_BUS_W = C×WK = 64 b` |
| WX, WK | 8 (signed int8) | `OUT_BUS_W = R×WY = 256 b` |
| WY | 32 (signed int32) | `A_BUS_W = R×32 = 256 b` |
| AXI_WIDTH | 64 | `Y_BUS_W = 256 b`, `Y_KEEP_W = 32` |
| AXIL_WIDTH | 32 | |
| ADDR_WIDTH | 64 | |

Register base is **`0x0000000020040000`** (parameter in `top_axi_int.v:27`).
The `AXIL_BASE_ADDR 32'hB0000000` macro in `config.svh` is unused — do not wire
it up.

## 3. RTL scalability

The array is fully generate-based (`axis_sa.sv`: `genvar r,c,d`, `D = R+C-1`) and
every bus width is derived from the parameters, so the datapath contains no
hard-coded dimensions, fixed indices or fixed beat counts. Changing `R`/`C`
requires no hand-editing of the datapath.

Four stream adapters bridge the 64-bit AXI memory interface and the internal
array buses:

| Adapter | Carries | Conversion |
|---|---|---|
| `ADAPTER_MM2S_K` | weights K | 64 → 64 (1:1) |
| `ADAPTER_MM2S_X` | activations X | 64 → 64 (1:1) |
| `ADAPTER_MM2S_A` | bias | 64 → 256 (**1:4 upsize**) |
| `ADAPTER_S2MM` | results Y | 256 → 64 (**4:1 downsize**) |

`alex_axis_adapter_any` converts via an LCM intermediate width, so both
directions are supported. `Y_KEEP_W = Y_BUS_W/8` scales automatically (32 here).
`ADAPTER_MM2S_A` omits an explicit `S_KEEP_WIDTH`; the module default of
`S_DATA_WIDTH/8 = 8` is correct — harmless, but noted.

Both the upsize and downsize paths are verified beat-by-beat in simulation rather
than assumed correct (see §10).

## 4. Test matrices and golden result

```
X[r][k] = ((3r + 2k) % 11) - 5     signed int8
K[k][c] = ((5k + 3c) %  9) - 4     signed int8
B[r][c] = 2r - c                   signed int32
Y = X·K + B                        signed int32
```

Golden Y (row-major display), produced independently by
`Application/stage3a_test/golden_8x8.py`:

```
[ 27, -28,  16,  24, -31,  13,  21, -34]
[ 18, -13, -17,  15, -16, -20,  12, -19]
[ -2,  24,   5,  -5,  21,   2,  -8,  18]
[-22,  -5,  -6, -25,  -8,  -9, -28, -11]
[  2,  10,  27,  -1,   7,  24,  -4,   4]
[ 26,  -8,  -6,  23, -11,  -9,  20, -14]
[ 39,  29, -17,  36,  26, -20,  33,  23]
[ -3, -22,  49,  -6, -25,  46,  -9, -28]
```

Two entries hand-verified independently: `Y[0][0] = 27`, `Y[7][7] = -28`. ✔

## 5. Memory packing

| Data | Layout | Address formula | Bytes | 64-bit beats |
|---|---|---|---|---|
| K | **row-major** | `0x20000 + k*8 + c` | 64 | 8 |
| X | **column-major** | `0x20100 + k*8 + r` | 64 | 8 |
| Bias | **column-major** | `0x20200 + c*32 + r*4` | 256 | 32 |
| Y | **column-major** | `0x20300 + c*32 + r*4` | 256 | 32 |

Beat *k* of K is row *k*; beat *k* of X is column *k*; beat *c* of bias/Y is
column *c*. Every read beat was decoded from the waveform and checked against
this packing.

**Signed packing:** every int8 is written through `(unsigned char)(signed char)`
so negatives are correct two's complement. Every comparison casts to `int`, not
unsigned.

## 6. SRAM map (no DDR)

| Buffer | Range | Size |
|---|---|---|
| K | `0x20000–0x2003F` | 64 B |
| X | `0x20100–0x2013F` | 64 B |
| Bias | `0x20200–0x202FF` | 256 B |
| Y | `0x20300–0x203FF` | 256 B |
| SIG | `0x20400` | 4 B |

Non-overlap and separation from `.data`/`.bss`/stack (`0x37000`) are enforced by
`_Static_assert` in the firmware — a build failure is impossible to miss.

## 7. Register map / DMA lengths (base `0x20040000`)

| Offset | Register | Value |
|---|---|---|
| 0x08 / 0x0C / 0x10 | MM2S_0 ADDR/BYTES/TUSER (**K**) | `0x20000` / **64** / 0 |
| 0x18 / 0x1C / 0x20 | MM2S_1 ADDR/BYTES/TUSER (**X**) | `0x20100` / **64** / 0 |
| 0x28 / 0x2C / 0x30 | MM2S_2 ADDR/BYTES/TUSER (**bias**) | `0x20200` / **256** / 0 |
| 0x38 / 0x3C | S2MM ADDR/BYTES (**Y**) | `0x20300` / **256** |
| 0x00 | START | 1 |
| 0x34 | S2MM_DONE (bit 0) | poll |

Expected 64-bit beat counts: K 8, X 8, bias 32, Y 32.

## 8. UART

`init_uart(0x1b)`, 8N1, ~115200 baud. Measured bit period **8.640 µs**. Markers:
`B` boot, `C` configured, `S` started, `P` pass, `F` fail. **Success = `BCSP`.**
No `rx_uart()`, no long prints. The completion poll is bounded, so a failure
still emits `F` inside the capture window.

Signatures: `0xC0DE0008` pass, `0xBADD0008` timeout/mismatch, at `0x20400`.

## 9. Integrity

| Artifact | SHA-256 |
|---|---|
| `stage3a_main.c` | `19297dada04f01b280099d749f713b1ec7f8d68b1d5f1ce796e44e03532cd377` |
| `golden_8x8.py` | `0f5a91541821431019fb2429334c760a469e84f182241239fc950e5c43585c44` |
| **64-bit MIF = installed ROM** | `1c9328faf2695e50b34adfad760e4db736f8cec8cec3f25e4468f93132aba03b` |

The simulation script verifies the canonical and installed MIF hashes match
before it runs, so a stale ROM cannot silently produce a green result.

Build verification: `.text.init`/`_start` @ `0x00010000`; binary **10,480 B** <
32 KB; one strong `main`; no `rx_uart`; 64-bit MIF line width = 64; the 32-bit
MIF is **not** installed.

> **VCD hashes are not fixed.** A VCD embeds a `$date` line, so every re-run
> produces a different hash even when the simulation is bit-identical. The hash
> recorded inside `stage3a_8x8_vcd_analysis.md` always refers to the VCD beside
> it. The stable integrity anchor is the MIF above.

## 10. Simulation — RESULT: PASS

Run from `RUN/run.sh` → 1 → 1 (Linux) or `RUN\run_stage3a.cmd` (Windows). Both
dispatch to `TCL/DVCon_SIM_stage3a.tcl`, which builds a clean wave config with
the divider groups **before** running — so all signals log from t=0 — verifies the
MIF hashes, runs **1500 µs**, then runs `analyze_vcd_8x8.py`, which exits non-zero
on any deviation. Wall-clock **24–30 minutes**.

Measured timeline: `B` 29.87 µs · config writes 466.45–471.07 µs · START 473.55 µs ·
first Y write 476.59 µs · `cfg_S2MM_DONE` 479.23 µs · `C` 487.79 µs · `S` 574.19 µs ·
`P` **1109.87 µs**.

Analyzer verdict: **ACCEPTED (exit 0)** — UART exactly `BCSP`, 64/64 outputs match
golden, K/X/bias read data all match, raw / `m_y_tdata` / S2MM streams agree, no
DDR access, 22/22 protocol checks OK.

Outputs: `GEN_BIT_OUT/stage3a_8x8_accelerator.vcd`,
`stage3a_8x8_accelerator.wcfg` (zoom preset 460–500 µs),
`stage3a_8x8_vcd_analysis.md`.

## 11. Synthesis — RESULT: PASS

Run from `RUN/run.sh` → 2, dispatching to `TCL/DVCon_SYN_stage3a.tcl` (project-run
flow matching the organizer's `DVCon_SYN.tcl`). Wall-clock **13–20 minutes**. The
script asserts 0 errors, 0 black boxes, `u_Accelerator_Top` present and **64
distinct PE instances** — if the array had been optimised away the datapath would
be disconnected and the run invalid.

| Scope | LUT | FF | BRAM | DSP |
|---|---|---|---|---|
| **Accelerator** | **12,404 (6.09 %)** | **12,711 (3.12 %)** | 0 | 0 |
| **SoC total** | **95,749 (46.98 %)** | 50,086 (12.29 %) | 81 (18.20 %) | 50 (5.95 %) |

Within the accelerator: PE array 7,168 LUT (58 %), DMA controller 332 LUT,
AXI-Lite bridge 242 LUT. The 50 DSPs at SoC level belong to the VEGA FPU;
INT8×INT8 products map to LUT logic.

Reports: `GEN_BIT_OUT/stage3a_8x8_utilization_{total,hier,accelerator}.rpt` and
`stage3a_8x8_timing_post_synth_ESTIMATE.rpt` (**estimate only** — no
implementation, no place-and-route, no timing closure).

Synthesis peaks near 4.5 GB and the project-run flow holds a second Vivado
process. On a machine with under 12 GB the script switches to a lighter
in-process flow automatically; see the report's Appendix A.

## 12. Known limitation — shared 64-bit operand bus

One row of K and one column of X are each 64 bits, but they share **one** 64-bit
AXI master through the crossbar. A fully-fed array wants ~16 B/cycle of operands
while the bus supplies 8 B/cycle, so sustained utilisation is bandwidth-limited.
**Full 64-MAC/cycle utilisation is not claimed** and has not been measured;
functional correctness and sustained throughput are separate claims.

## 13. Acceptable warnings

Two unbound `ddr3_model` instances (`test_bench.vhd:212,234`) — DDR is unused in
Stage 3A — the crossbar `qos`/`user` width warnings, and eight `set_clock_groups`
critical warnings from the organizer's `AS1061_SYSTEM_XDC.xdc` (lines 317–318),
which appear identically on runs that complete successfully.
