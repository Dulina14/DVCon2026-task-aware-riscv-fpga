# hardware/stage3a_simulation/

Our Stage 3A deliverable (8×8 accelerator inside the VEGA SoC, full-system simulation + synthesis). These are **only the files Team 43 wrote**; they plug into the organiser's `DVCon_2026` environment, which is not public.

| Path | What it is |
|---|---|
| [`firmware/stage3a_main.c`](firmware/stage3a_main.c) | Bare-metal self-checking test program for the VEGA boot ROM. Prints `B` `C` `S` `P` (or `F`) on the UART; writes signature `0xC0DE0008` (pass) / `0xBADD0008` (fail) at `0x20400` |
| [`firmware/golden_8x8.py`](firmware/golden_8x8.py) + [`golden_8x8.txt`](firmware/golden_8x8.txt) | Independent Python golden model: X, K, B, Y, packed memory images, expected AXI beats |
| [`firmware/analyze_vcd_8x8.py`](firmware/analyze_vcd_8x8.py) | Strict VCD analyser: rebuilds every AXI transaction, decodes the UART, ~32 acceptance checks incl. 22 AXI protocol properties, exits non-zero on any failure |
| [`tcl/DVCon_SIM_stage3a.tcl`](tcl/DVCon_SIM_stage3a.tcl) | Simulation flow (GUI or batch): builds the wave layout first, checks the ROM-image SHA-256, runs 1,500 µs, calls the analyser |
| [`tcl/DVCon_SYN_stage3a.tcl`](tcl/DVCon_SYN_stage3a.tcl) | Synthesis flow: checks for 0 black boxes, `u_Accelerator_Top` present, 64 PE instances; RAM-aware |
| [`run/run_stage3a.sh`](run/run_stage3a.sh), [`.ps1`](run/run_stage3a.ps1), [`.cmd`](run/run_stage3a.cmd) | One-command launchers for Linux and Windows: `--simulation`, `--synthesis`, `--all`, `--check` |
| [`results/`](results/) | Our run: `stage3a_8x8_vcd_analysis.md` (ACCEPTED), the VCD and wave config, utilisation reports, post-synthesis timing *estimate*, manifests, full logs |
| [`README_STAGE3A_8X8.md`](README_STAGE3A_8X8.md) | Technical reference: configuration, packing, SRAM map, register map, integrity hashes, results |
| [`STAGE3A_WAVEFORM_GUIDE.md`](STAGE3A_WAVEFORM_GUIDE.md) | How to check the waveform by hand, event by event |
| [`SUBMISSION_README_original.md`](SUBMISSION_README_original.md) | The README that shipped with our submission (paths refer to the organiser environment) |

## How it fits into the organiser environment

```
DVCon_2026/                      (organiser kit — not in this repo)
├── DVCon_SoC_SRC/ACCELERATOR_IP/   ← replace with  hardware/accelerator_ip/
├── Application/stage3a_test/       ← firmware/ here + organiser crt.S, uart.c, stdlib.c, mbl.lds, Makefile, include/
├── TCL/                            ← add tcl/ here (organiser scripts stay untouched)
├── RUN/                            ← add run/ here
└── GEN_BIT_OUT/                    ← results appear here
```

The test program is built with the organiser's bare-metal toolchain (`riscv64-unknown-elf-gcc` 12.2.0, `-march=rv64ima -mabi=lp64`) in their Docker image, and its **64-bit** MIF is installed as `rom_32KB_axi.mif`. Ship the program in both the `.mif` and the ROM IP's `.coe` — older Vivado versions regenerate the `.mif` from the `.coe` (see [docs/06 §6](../../docs/06_stage3a_soc_integration.md#6-the-trap-that-almost-cost-us-the-stage)).

## Result summary

- Simulation (Vivado 2025.2 XSim, 5 Aug 2026): **ACCEPTED** — UART `BCSP`, 64/64 results match, 22/22 protocol checks, no DDR access. `P` at 1,109.87 µs.
- Synthesis (Vivado 2025.2): accelerator **12,404 LUTs (6.09 %)**, 12,711 FFs, 0 BRAM, 0 DSP.

Logs have had the local machine name and user path replaced with placeholders.
