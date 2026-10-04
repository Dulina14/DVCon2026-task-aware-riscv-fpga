# Stage 3A — Submission Cleanup Checklist

Judging rule that drives everything below: *"The organizers shall be able to
reproduce the simulation, generated waveforms, synthesis results, and reports
directly from the submitted project without requiring any additional files,
modifications, or configuration changes."*

So the test is: **unzip → run the flow → it works.** Anything needed for that
stays; anything that breaks it must be fixed; anything else is optional clutter.

---

## 🔴 TWO BLOCKERS — FIX BEFORE SUBMITTING

### BLOCKER 1 — the ROM image file is MISSING
`DVCon_SoC_SRC/MEMORY_IP/rom_32KB_axi/rom_32KB_axi.mif` **does not exist.**
Only these are present:
- `rom_32KB_axi.mif.stage2_backup` (your backup of the old one)
- `accelarator_test_64.mif` (organizer file)

The ROM IP loads `C_INIT_FILE_NAME = rom_32KB_axi.mif` (from `rom_32KB_axi.xci`).
Without it the CPU has no program → the organizers' run produces nothing.
(It presumably got renamed to `.stage2_backup` instead of copied.)

**FIX (Windows cmd):**
```
cd D:\HDL\verilator\DVCon_2026\DVCon_SoC_SRC\MEMORY_IP\rom_32KB_axi
copy ..\..\..\Application\stage3a_test\build\accelarator_test.hex.mif rom_32KB_axi.mif
```
**VERIFY** (must both be `8b5fdddb7fb5fc245345103e178424ca`):
```
certutil -hashfile rom_32KB_axi.mif MD5
certutil -hashfile ..\..\..\Application\stage3a_test\build\accelarator_test.hex.mif MD5
```
Then re-run the simulation once from scratch to confirm it still passes.

### BLOCKER 2 — organizer app folder now has TWO `main()`
`Application/dummy_firmware_accelarator_64_64_matrix_20000/` contains both
`main.c` (organizer) and `stage2_main.c` (ours, added Jul 15). Its `Makefile`
line 20 uses `$(wildcard *.c)`, so **`make` there fails with duplicate `main`.**
If a judge builds the organizer sample, it breaks — and it is *their* folder.

**FIX:** delete our stray file from the organizer folder (it is preserved inside
`stage3a_test` history / not needed):
```
del D:\HDL\verilator\DVCon_2026\Application\dummy_firmware_accelarator_64_64_matrix_20000\stage2_main.c
```

---

## Answers to your questions

### Application/ — the extra folders

| Item | What it is | Keep? |
|---|---|---|
| **`stage3a_test/`** | **Your final Stage 3A firmware.** `stage3a_main.c` = one `main()`, UART markers `B/C/S/P`, SRAM-only buffers, no `rx_uart`. Plus the supporting files copied from the organizer sample (`crt.S`, `uart.c`, `stdlib.c`, `xmodem*.c`, `mbl.lds`, `Makefile`, `include/`, `build.sh`) so it builds standalone. | ✅ **KEEP — this is deliverable "Bare-metal application source code"** |
| **`systolic_accel_test/`** | A scratch copy I made on Jul 15 while developing. Its `main.c` is just the **organizer's original example** (matrix_a @0x20000, registers @0x20040010…). Adds nothing; duplicates the organizer sample. | ❌ **DELETE** (dead weight, and a second `main()` folder invites confusion) |
| **`dummy_firmware_.../stage2_main.c`** | Our intermediate 2×2 smoke test, wrongly left **inside the organizer's folder**. Superseded by `stage3a_main.c`. Causes BLOCKER 2. | ❌ **DELETE** (see fix above) |
| `dummy_firmware_accelarator_64_64_matrix_20000/` (rest) | Organizer's original example. | ✅ keep, unmodified |

### Why `stage3a_test` had no `build.sh` / empty `build/`
`build.sh` is the organizer's convenience wrapper (`make` → objdump → objcopy →
bin2hex → hex2mif → hex2mem). When I created `stage3a_test` I copied the source
files and drove those steps manually, so `build.sh` wasn't copied and `build/`
was empty until a build was run. **Both are now fixed** — `build.sh` is present
and `build/` is populated (verified: ELF, `.bin`, `.dump`, `.hex`, `.hex.mif`,
`.hex.mem`, `.mem`).

### What the `build/` files are

| File | Meaning | Keep? |
|---|---|---|
| `accelarator_test` | the **ELF** executable (linked program) | ✅ |
| `.bin` | raw binary (ELF stripped to pure bytes), 9,264 B | ✅ |
| `.dump` | disassembly (`objdump -D`) — human-readable machine code | ✅ |
| `.hex` | 64-bit-per-line hex of the binary | ✅ |
| **`.hex.mif`** | **64-bit MIF = the ROM initialization image.** This is the one the ROM IP loads. | ✅ **critical** |
| `.hex.mem` / `.mem` | `$readmemh`-style memory image (alternate format) | ✅ (small) |
| `_32.hex` / `_32.hex.mif` | 32-bit-wide variants — **not** used (ROM is 64-bit) | optional |
| `*.o` | intermediate object files from compilation | ❌ can delete |

Keep the build outputs: judges asked for the program image, and it proves your
firmware matches the ROM.

### GEN_BIT_OUT/ — your generated evidence

| File | What it is | Keep? |
|---|---|---|
| `stage3a_accelerator.vcd` | **The waveform proving the design works** (1.83 MB, 207 signals, reset→config→SRAM read→compute→write-back→DONE, UART `BCSP`) | ✅ **required** — deliverable *"Simulation waveform files demonstrating successful execution"* |
| `stage3a_accelerator.wcfg` | Vivado waveform layout with the 10 labelled dividers — lets a judge open the waves already organized | ✅ keep (strongly recommended) |
| `stage3a_utilization_total.rpt` | Whole-design utilization (LUT 88,010 = 43.18%, BRAM 81, DSP 50) | ✅ **required** — *"Synthesis reports … resource utilization"* |
| `stage3a_utilization_accelerator.rpt` | **Your accelerator only**: LUT 5,214 (2.56%), FF 5,326 (1.31%), BRAM 0, DSP 0 | ✅ **required** (shows you're well under the 20–25% guidance) |
| `stage3a_utilization_hier.rpt` | Hierarchical breakdown | ✅ keep |
| `DVCon_SoC_Simulation_Waveform.wcfg`, `bootloader_DVCon_BF00.coe` | organizer originals | ✅ keep unmodified |

### RUN/
Contains the organizer's **`run.sh`** — the menu-driven entry point
(1 Simulation → 1 Vivado / 2 QuestaSim; 2 Implementation; 3 Programming). It
`sed`s the project path into `TCL/DVCon_SIM.tcl` and `DVCon_SYN.tcl` then calls
Vivado. **It is the official flow the judges will use, so keep it untouched.**
It also holds a stray `rom_32KB_axi.mif` copy plus `vivado.log/.jou/.Xil/dfx_runtime.txt`
run residue (deletable).

### TCL/

| File | What it is | Keep? |
|---|---|---|
| `DVCon_SIM.tcl`, `DVCon_SYN.tcl`, `QuestaSim_SIM.do` | **organizer originals** | ✅ keep unmodified |
| `DVCon_SIM_win.tcl` | your Windows sim helper (auto path, recursive accel sources, `config.svh` include dir, `top_axi_int.v` as SystemVerilog) | ✅ **keep** — deliverable *"Simulation scripts"* |
| `DVCon_SYN_win.tcl` | your Windows synthesis helper + utilization reports | ✅ **keep** — *"Synthesis scripts"* |
| `dump_vcd_stage3a.tcl` | generates the required VCD | ✅ **keep** |
| `make_wcfg_stage3a.tcl` | generates the divider-organized `.wcfg` | ✅ keep |
| `dump_vcd.tcl` | older debug version, superseded by `dump_vcd_stage3a.tcl` | ❌ delete (or keep; harmless) |
| `dfx_runtime.txt` | Vivado run residue, not a source file | ❌ delete |

### Project root

| File | What it is | Keep? |
|---|---|---|
| `README_STAGE3A.md` | your documentation (build/sim/synth commands, memory map, register map, UART, expected results, rerun procedure) | ✅ **keep** — deliverable *"Documentation"* |
| `acc_dump2.vcd` | an **old debug VCD** from the Stage-2 experiments (5 KB) — superseded | ❌ delete |
| `vivado.log`, `vivado.jou` | Vivado run logs from a root-launched run | ⚠ optional — keep the *successful* sim/synth logs (they live under `VIVADO_PROJECT/.../xsim/`), delete these stale root ones |
| `dfx_runtime.txt` | Vivado residue | ❌ delete |
| `DDR3_SIM_INVESTIGATION.md` (if present) | your DDR analysis — useful supporting documentation | ✅ optional keep |

---

## Final cleanup commands (Windows cmd)

```bat
cd /d D:\HDL\verilator\DVCon_2026

:: --- BLOCKER 1: restore the ROM image ---
copy Application\stage3a_test\build\accelarator_test.hex.mif ^
     DVCon_SoC_SRC\MEMORY_IP\rom_32KB_axi\rom_32KB_axi.mif

:: --- BLOCKER 2: remove duplicate main from the organizer folder ---
del Application\dummy_firmware_accelarator_64_64_matrix_20000\stage2_main.c

:: --- remove scratch/duplicate work ---
rmdir /s /q Application\systolic_accel_test

:: --- remove stray run residue ---
del acc_dump2.vcd dfx_runtime.txt vivado.log vivado.jou
del TCL\dfx_runtime.txt
del TCL\dump_vcd.tcl
del Application\stage3a_test\build\*.o
```

### Then re-verify (mandatory — never submit unverified)
1. `vivado -source D:/HDL/verilator/DVCon_2026/TCL/DVCon_SIM_win.tcl`
2. In the Tcl console: `source .../TCL/dump_vcd_stage3a.tcl` → confirm it prints
   `logging ~207 objects` and writes the VCD.
3. Confirm UART still decodes `B C S P` and `m_axi` writes `0x20300`.
4. Optionally re-run `DVCon_SYN_win.tcl`.

---

## What the final submission tree should contain

```
DVCon_2026/
├── Application/
│   ├── dummy_firmware_accelarator_64_64_matrix_20000/   (organizer, main.c only)
│   ├── stage3a_test/                <-- YOUR firmware + build/ outputs
│   ├── Util/, toolchain-bare/, .docker_scripts/, sim_app_compile_and_mif.sh
├── DVCon_SoC_SRC/
│   ├── ACCELERATOR_IP/              <-- YOUR accelerator RTL (+ INTEGRATION_NOTES.md)
│   ├── AS1061_SYSTEM/, TOP/, TB/, MEMORY_IP/  (organizer; ROM MIF = your image)
├── DVCon_SoC_XDC/                   (organizer constraints)
├── GEN_BIT_OUT/
│   ├── stage3a_accelerator.vcd      <-- waveform evidence
│   ├── stage3a_accelerator.wcfg
│   ├── stage3a_utilization_*.rpt    <-- synthesis reports
├── RUN/run.sh                       (organizer)
├── TCL/  (organizer 3 + your 4 helper scripts)
├── VIVADO_PROJECT/                  (generated; see note)
├── README_STAGE3A.md
└── SUBMISSION_CHECKLIST.md
```

**`VIVADO_PROJECT/`**: you have many timestamped projects, each large. Keep **at
most one** successful sim project (and one synth) as evidence, or delete them all
— the judges regenerate projects from the scripts. Deleting them massively shrinks
the ZIP. If you keep one, keep the run whose logs match your submitted VCD.
