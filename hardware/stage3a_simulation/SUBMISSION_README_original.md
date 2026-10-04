# DVCon India 2026 — Design Contest, Stage 3A

**Accelerator:** 8×8 output-stationary systolic array (INT8 × INT8 → INT32, `Y = X·K + bias`)
**Host SoC:** VEGA AS1061 (RV64IMAFDC, 6-stage), unmodified
**Target part:** `xc7k325tffg900-2`

---

## 1. How to run (this is the entry point)

Everything is driven from `RUN/`. No Docker, no toolchain install and no network
access is needed to reproduce the results below — the firmware is already compiled
and installed into the boot ROM image.

### Linux

```bash
cd RUN
./run.sh
```

### Windows

```bat
REM Command Prompt
RUN\run_stage3a.cmd

REM or PowerShell
RUN\run_stage3a.ps1
```

Either Windows entry point works — `run_stage3a.cmd` is a thin wrapper that
launches `run_stage3a.ps1`, so use whichever shell you prefer.

Both present the same menu:

| Menu | What it does |
|---|---|
| **1 → 1** Vivado (XSim) simulation | Builds a fresh project, runs **1500 µs**, writes the VCD, then runs a strict VCD analyzer that PASSes or FAILs the run. **24–30 min.** |
| **1 → 2** QuestaSim simulation | Same DUT compiled under QuestaSim via `TCL/QuestaSim_SIM.do`, using `RUN/rom_32KB_axi.mif`. |
| **2** Synthesis | Project-run RTL synthesis; writes three utilization reports into `GEN_BIT_OUT/`. **13–20 min.** |

The **Vivado GUI is the default** — the waveform stays open at the end of
simulation, and the synthesized design stays open at the end of synthesis.

For a headless / CI run:

```bash
./run_stage3a.sh --batch          # Linux
RUN\run_stage3a.cmd all           # Windows (batch)
```

> **Note on `RUN/run.sh`** — the organizer's original is preserved unchanged as
> `RUN/run.sh.organizer_original`. It was updated because it rewrote a hard-coded
> path (`D:/DVCon/DVCon_2026/...`) that exists on no other machine, and because
> `TCL/DVCon_SIM.tcl:55` has `#run $SIM_RUN_TIME us` commented out, so the original
> flow opened the GUI and never actually simulated. The same applies to
> `TCL/QuestaSim_SIM.do` (original kept as `.organizer_original`): its file glob was
> non-recursive and missed all RTL under `rtl/ext`, `rtl/sa` and `rtl/sys`.

### Expected simulation result

```
STAGE 3A 8X8 SIMULATION PASS
```

The run is only declared PASS when **all** of the following hold:

- UART output is **exactly** `BCSP` — B boot, C configured, S started, **P pass**
- exactly **64** result words are written and all 64 match the golden reference
- K / X / bias read data match the expected packing at every beat
- the raw PE output, `m_y_tdata` and the S2MM write stream all agree
- no access falls in the DDR3 range (DDR is not available in Stage 3A)
- AXI protocol checks (handshake, ID pairing, beat counts) all pass

Any deviation makes the analyzer exit non-zero.

---

## 2. Verified results

### Simulation (Vivado XSim, 1500 µs)

| Event | Time |
|---|---|
| `B` — boot marker | 29.87 µs |
| `C` — accelerator configured | 487.79 µs |
| `S` — START written | 574.19 µs |
| `cfg_S2MM_DONE` asserted | 479.23 µs |
| `P` — all 64 outputs verified | 1109.87 µs |

64/64 outputs match the independent golden model
(`Application/stage3a_test/golden_8x8.py`). Pass signature `0xC0DE0008` at `0x20400`.

### Synthesis (`xc7k325tffg900-2`)

| | LUT | FF | BRAM | DSP |
|---|---|---|---|---|
| **Accelerator** (`u_Accelerator_Top`) | 12,404 (6.09 %) | 12,711 (3.12 %) | 0 | 0 |
| **Whole SoC** (`Top`) | 95,749 (46.98 %) | 50,086 (12.29 %) | 81 (18.20 %) | 50 (5.95 %) |

0 errors, **0 black boxes**, **64 distinct PE instances** confirmed present in the
netlist (the array was not optimised away). The accelerator sits comfortably inside
the 20–25 % guidance.

Reports: `GEN_BIT_OUT/stage3a_8x8_utilization_{total,hier,accelerator}.rpt`
Timing: `GEN_BIT_OUT/stage3a_8x8_timing_post_synth_ESTIMATE.rpt` — post-synthesis
**estimate only**, not timing closure (no implementation was run, per Stage 3A scope).

> DSPs are 0 by design: the PEs are INT8×INT8 multipliers, which Vivado maps to LUT
> logic rather than DSP48E1 slices. The 50 DSPs in the SoC total belong to the VEGA
> core's FPU.

---

## 3. Integration

```
Top.vhd (organizer, UNMODIFIED)
  └── u_Accelerator_Top  ── DVCon_SoC_SRC/ACCELERATOR_IP/Accelerator_Top.v
        ├── axi_axil_adapter        64-bit AXI4  →  32-bit AXI-Lite (register access)
        └── top_axi_int             AXI4 master (DMA) + AXI-Lite slave (config)
              ├── dma_controller    3× MM2S (K, X, bias) + 1× S2MM (Y)
              └── axis_sa           8×8 output-stationary PE array
```

`Top.vhd` is **untouched** — `Accelerator_Top.v` was written to match the component
declaration that already exists there.

| Region | Range | Use |
|---|---|---|
| Boot ROM | `0x00010000 – 0x00017FFF` | firmware |
| SRAM | `0x00020000 – 0x0003FFFF` | K, X, bias, Y buffers |
| **Accelerator regs** | `0x20040000` | AXI-Lite control |
| UART | `0x10001000` | markers |
| DDR3 | `0x80000000 – 0xBFFFFFFF` | **not used in Stage 3A** |

Register map, memory packing (K row-major, X/bias/Y column-major) and DMA byte
counts are documented in `README_STAGE3A_8X8.md` §5–§7.

---

## 4. Rebuilding the firmware (optional — not needed to reproduce)

The installed boot ROM already contains the 8×8 test program.

```
Application/stage3a_test/stage3a_main.c   SHA-256 19297dad…532cd377
DVCon_SoC_SRC/.../rom_32KB_axi.mif        SHA-256 1c9328fa…32aba03b
RUN/rom_32KB_axi.mif                      SHA-256 1c9328fa…32aba03b   (QuestaSim path)
```

The simulation script re-verifies that the canonical and installed MIF hashes match
before it runs, so a mismatched or stale ROM cannot silently produce a green run.

To rebuild, you need the RISC-V toolchain (`riscv64-unknown-elf-gcc` 12.2.0):

```bash
./run_stage3a.sh --rebuild-firmware       # Linux, uses the dvcon_2:latest image
```

---

## 5. Repository map

| Path | Contents |
|---|---|
| `RUN/` | entry points — `run.sh` (Linux), `run_stage3a.cmd` / `.ps1` (Windows), boot ROM MIF |
| `TCL/` | `DVCon_SIM_stage3a.tcl`, `DVCon_SYN_stage3a.tcl`, `QuestaSim_SIM.do` |
| `DVCon_SoC_SRC/ACCELERATOR_IP/` | the submitted accelerator RTL |
| `Application/stage3a_test/` | firmware, golden model, VCD analyzer |
| `GEN_BIT_OUT/` | VCD, WCFG, utilization + timing reports, analysis write-up |

## 6. Further reading

| Document | Covers |
|---|---|
| `../Document/DVCon2026_Stage3A_Report.pdf` | the full report — architecture, datapath, results, resource analysis, and why this primitive |
| `../Readme/README_INSTRUCTIONS.txt` | condensed run instructions for the evaluation team |
| `README_STAGE3A_8X8.md` | 8×8 configuration, packing, register map, scalability |
| `STAGE3A_WAVEFORM_GUIDE.md` | how to read the waveform — which signals, which time windows |
| `GEN_BIT_OUT/stage3a_8x8_vcd_analysis.md` | full transaction-level decode of the passing run |

## 7. Known limitations (stated up front)

- **Shared 64-bit operand bus.** X and K are each 64 bits wide at 8×8 but share one
  64-bit AXI master through the crossbar. Sustained utilisation is bandwidth-limited;
  full 64-MAC/cycle throughput is **not** claimed and has not been measured.
- **No DDR3.** DDR is out of scope for Stage 3A; the two unbound `ddr3_model`
  instances in `test_bench.vhd:212,234` are expected warnings.
- **Timing is an estimate.** No implementation, no bitstream, no place-and-route.
