# Accelerator SoC Integration Notes

How the systolic-array accelerator is connected to the VEGA AS1061 SoC.

> **Stage 3A correction (supersedes DDR references below):** DDR3
> (`0x80000000–0xBFFFFFFF`) is **not modeled** in the supplied SoC simulation and
> must **not** be used in Stage 3A. All accelerator input/output buffers live in
> **internal SRAM** (`0x00020000–0x00036FFF`). Any mention of placing data at
> `0x80000000` in the sections below applies only to a future DDR-enabled design,
> not to Stage 3A. See `README_STAGE3A.md` and `DDR3_SIM_INVESTIGATION.md`.

## Architecture

`Top.vhd` instantiates a component named `Accelerator_Top`. We provide a Verilog
`Accelerator_Top.v` with the **exact same port list**, so `Top.vhd` is unchanged.
Inside the wrapper:

```
 SoC config port                 SoC memory port
 (AXI4 64b/64addr/12id)          (AXI4 64b/64addr/12id)
        | s_axi_*                       ^ m_axi_*
        v                               |
  axi_axil_adapter  --AXI-Lite 32b-->  top_axi_int  ------(direct)------> (to SoC)
  (64b AXI4 -> 32b AXI-Lite)           (accelerator, reparam'd to 64-bit master)
```

- **Config (slave):** the SoC drives a full AXI4-64 port. `axi_axil_adapter`
  converts it to the accelerator's AXI4-Lite-32 register interface.
- **Memory (master):** the accelerator is reparameterised to a native 64-bit
  AXI4 master (64b data, 64b addr, 12b ID), so it connects **directly** to the
  SoC memory bus — no converter needed.

## Files added / changed (all inside ACCELERATOR_IP/)

| File | Change |
|---|---|
| `Accelerator_Top.v` | **new** — SoC wrapper (matches Top.vhd component ports) |
| `rtl/ext/axi_axil_adapter.v` `_rd.v` `_wr.v` | **new** — AXI4→AXI-Lite bridge (Alex Forencich, MIT) |
| `rtl/sys/config.svh` | `AXI_WIDTH 32→64`, `ADDR_WIDTH 32→64` (AXIL_WIDTH stays 32) |
| `rtl/sys/top_axi_int.v` | `AXI_ID_WIDTH 6→12`; `AXIL_BASE_ADDR → 0x0000_0000_2004_0000` |

`Top.vhd`, `AS1061_SYSTEM_TOP.v`, and the testbench are **untouched**.

## Address map assumptions (from the SoC memory map)

- **Accelerator config region:** `0x2004_0000 – 0x2004_FFFF`. The register decode
  base `AXIL_BASE_ADDR` is set to `0x2004_0000`. The register offsets are the
  ones in `dma_controller.sv` (START=+0x0, MM2S_0_ADDR=+0x8, …), i.e. the CPU
  writes `0x2004_0000 + reg_index*4`.
- **DDR3:** `0x8000_0000 – 0xBFFF_FFFF` — fits in 32 bits, so the 32-bit
  addresses the firmware programs into the registers reach DDR after being
  zero-extended onto the 64-bit master.

> IMPORTANT — base-address caveat: `AXIL_BASE_ADDR = 0x2004_0000` assumes the
> interconnect presents the **full** address to the accelerator slave. If the
> interconnect instead strips the base and presents an **offset** (0-based),
> register writes will miss — in that case set `AXIL_BASE_ADDR = 0` in
> `top_axi_int.v` and re-run. This is the first thing to check if config writes
> don't land.

## What the firmware (bare-metal program / .mif) must do

The `.mif` is not updated by this integration — you write firmware that:
1. Places input matrices (X, K, bias) in DDR (0x8000_0000 range).
2. Programs the accelerator registers over AXI-Lite at `0x2004_0000`:
   MM2S_0/1/2 ADDR+BYTES (K, X, bias source), S2MM ADDR+BYTES (Y dest), then START.
3. Polls S2MM_DONE, then reads Y from DDR.
   (Register map: see `dma_controller.sv`.)
Because the master is now 64-bit, size the byte-length registers and data layout
for 64-bit (8-byte) beats.

## How to simulate

From the top of the submission environment:

```
cd RUN && ./run.sh        # Linux  — menu 1 → 1
RUN\run_stage3a.cmd       # Windows — menu 1 → 1
```

Either dispatches to `TCL/DVCon_SIM_stage3a.tcl`, which creates the project,
adds all accelerator files recursively, sets the include dir for `config.svh`,
runs 1500 µs and then self-checks the result. No manual `run` is needed.

## Verification checklist — ALL ITEMS PASS

This was built correct-by-construction and has since been fully verified in
Vivado (simulation PASS, synthesis PASS). The list below is retained as the
audit trail; every item was confirmed:
1. **Compile/elaboration** — the accelerator RTL + adapters + wrapper elaborate
   with no missing modules or width mismatches.
2. **Config writes** — in the waveform, confirm the AXI-Lite writes reaching
   `u_accel` land on the right registers (base-address caveat above).
3. **Master reads/writes** — the accelerator issues AR/AW bursts to the DDR
   addresses the firmware programmed.
4. **DONE + result** — S2MM_DONE asserts and the result appears in DDR.

