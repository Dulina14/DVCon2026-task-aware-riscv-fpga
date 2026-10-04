# REBUILD — regenerating the bitstream from source

> **Note for this public repository.** This file is reproduced from our Stage 3B
> submission package. The organiser-owned files it mentions (`run.sh`, `tcl/*.tcl`,
> `xdc/*.xdc`, `memory_ip/*`, `tb/*`, `Top.vhd`, the VEGA netlist and the
> bitstream, which contains the VEGA core) are **not** redistributed here. You get
> them from the contest organisers (C-DAC). Paths such as `../rtl/ACCELERATOR_IP/`
> map to `hardware/accelerator_ip/` in this repository. See `hardware/README.md`.

This directory holds everything needed to re-run synthesis and implementation
and reproduce `accelerator/bitstream/DVCon_2026.bit`, **except** the organiser's
own SoC sources, which are listed under "Not included" below.

The flow is entirely Vivado batch TCL. There is no GUI step.

---

## 1. What Team 43 actually added to the SoC

The organiser's `Top.vhd` declares and instantiates a component named
`Accelerator_Top`. Team 43 supplies that entity. Nothing in `Top.vhd`,
`AS1061_SYSTEM_TOP`, or the testbench was modified.

| File | Role |
|---|---|
| `../rtl/ACCELERATOR_IP/Accelerator_Top.v` | **The integration wrapper.** Port list matches the VHDL component in `Top.vhd` exactly, so `Top.vhd` needs no edit. |
| `../rtl/ACCELERATOR_IP/INTEGRATION_NOTES.md` | How the wrapper attaches to the SoC: bus widths, address map, register base, verification checklist. |
| `../rtl/ACCELERATOR_IP/rtl/sa/` | The 8x8 systolic array itself — `pe.sv`, `mac.sv`, `axis_sa.sv`, `tri_buffer.sv`, `n_delay.sv`. |
| `../rtl/ACCELERATOR_IP/rtl/sys/` | `top_axi_int.v` (accelerator top, AXI master + AXI-Lite slave), `dma_controller.sv` (register file and the three MM2S / one S2MM channels), `config.svh`, `skid_buffer.sv`. |
| `../rtl/ACCELERATOR_IP/rtl/ext/` | Third-party AXI infrastructure (Alex Forencich, MIT licence), including the `axi_axil_adapter` that converts the SoC's 64-bit AXI4 config port to the accelerator's 32-bit AXI-Lite register interface. |

Inside the wrapper:

```
 SoC config port                    SoC memory port
 (AXI4 64b data / 64b addr / 12b id)      ^ m_axi_*
        | s_axi_*                         |
        v                                 |
  axi_axil_adapter --AXI-Lite 32b--> top_axi_int --(direct, native 64b master)-->
```

The accelerator is reparameterised to a native 64-bit AXI4 master, so its memory
port connects to the SoC bus with no converter. Only the config port needs the
adapter.

**Register base address.** The decode base is set in `top_axi_int.v`:

```verilog
AXIL_BASE_ADDR = 64'h0000000020040000
```

so the CPU reaches register *n* at `0x2004_0000 + n*4`. Note that
`rtl/sys/config.svh` still carries an unused `AXIL_BASE_ADDR 32'hB0000000` from
the standalone simulation environment — the parameter in `top_axi_int.v` is the
one that takes effect. Do not "fix" `config.svh` to match; the shipped bitstream
was built with the files exactly as they are here.

---

## 2. Placing these files

The TCL scripts use absolute paths rooted at:

```tcl
set MDIR_PATH "/home/Team43/Desktop/DVCon_2026"
```

Edit that one line in each script if your project root differs. The scripts then
expect the organiser's standard layout:

```
$MDIR_PATH/
  DVCon_SoC_SRC/
    AS1061_SYSTEM/AS1061_SYSTEM_TOP.edn      <- organiser, NOT included (see 5)
    TOP/Top.vhd                              <- ../rtl/SOC_TOP/Top.vhd
    MEMORY_IP/rom_32KB_axi/rom_32KB_axi.xci  <- memory_ip/ here
    ACCELERATOR_IP/                          <- ../rtl/ACCELERATOR_IP/ here (all of it)
  DVCon_SoC_XDC/                             <- xdc/ here
  GEN_BIT_OUT/                               <- output; also holds the bootloader .coe
  TCL/                                       <- tcl/ here
  RUN/                                       <- run.sh here; run the launcher from this directory
  VIVADO_PROJECT/                            <- created by the scripts
```

Copy `../rtl/ACCELERATOR_IP/` over `DVCon_SoC_SRC/ACCELERATOR_IP/` wholesale.
`DVCon_SYN.tcl` and `DVCon_IMP.tcl` add accelerator sources by recursive `find`,
so every `.v`, `.sv`, `.vhd` and `.xci` under that directory is picked up
automatically — including `Accelerator_Top.v`. `config.svh` is not added
directly; it is pulled in by `` `include ``.

`memory_ip/rom_32KB_axi.xci` references
`../../../GEN_BIT_OUT/bootloader_50MHZ_10001000_MAC_90.coe`, which is why that
file is included here. Put it in `GEN_BIT_OUT/`, not in `memory_ip/`.

---

## 3. Running the flow

**Use the organiser launcher, `run.sh`.** It is shipped here as
`accelerator/build/run.sh` and belongs in `$MDIR_PATH/RUN/`. Run it from inside
that directory:

```bash
cd /home/Team43/Desktop/DVCon_2026/RUN
./run.sh
```

It presents a menu:

```
1) Simulation                -> 1 Vivado, 2 QuestaSim
2) FPGA Implementation       -> 1 synthesis only, 2 full build + bitstream
3) FPGA Programming          -> loads DVCon_2026.bit onto the board
4) Exit
```

**Prefer the launcher over calling Vivado directly.** Before showing the menu it
rewrites the `set MDIR_PATH "..."` line in all four TCL scripts to the parent of
its own directory, so the absolute paths inside those scripts become correct on
whatever machine you are on. Calling Vivado by hand skips that step and the
scripts will look for files where they were on our lab PC. The launcher also
clears stale `*.log`, `*.jou`, `.Xil` and simulation artefacts from `RUN/`
before starting.

The direct equivalents, if you are not using the launcher and have fixed
`MDIR_PATH` yourself:

| Menu choice | Equivalent command |
|---|---|
| 2 → 1, synthesis only | `vivado -mode batch -source $MDIR_PATH/TCL/DVCon_SYN.tcl` |
| 2 → 2, full build **(this produced the shipped bitstream)** | `vivado -mode batch -source $MDIR_PATH/TCL/DVCon_IMP.tcl` |
| 3, program the board | `vivado -mode batch -source $MDIR_PATH/TCL/FPGA_PROGRAMMING.tcl` |
| 1 → 1, Vivado simulation | `vivado -source $MDIR_PATH/TCL/DVCon_SIM.tcl` |
| 1 → 2, QuestaSim | `vsim -do $MDIR_PATH/TCL/QuestaSim_SIM.do` |

It writes, all under `$MDIR_PATH/GEN_BIT_OUT/`:

- `DVCon_2026.bit`
- `report_timing_summary.rpt`
- `report_hierarchical_utilization_summary.rpt`
- `report_utilization_summary.rpt`

All four of those are shipped in `../bitstream/` and `../reports/`, so a rebuild
can be compared against them directly. Expect the accelerator hierarchy to
report as:

```
u_Accelerator_Top   Accelerator_Top   11945 LUT (5.86%)   12706 FF   0 BRAM   0 DSP
  u_accel           top_axi_int       11703 LUT (5.74%)   12254 FF
```

Both scripts hard-code `-jobs 40`. Lower it if your machine has fewer cores.

Simulation is menu option 1. Both simulators drive `tb/test_bench.vhd` together
with `tb/glbl.v`.

---

## 4. Programming the board

Menu option **3** of `run.sh`. One edit is still required regardless of how you
invoke it: the script targets a specific Digilent cable by serial number —

```tcl
current_hw_target [get_hw_targets */xilinx_tcf/Digilent/200300B18D0EB]
```

Change that serial to match the board in front of you, or delete the two
`*_hw_targets` lines and let Vivado pick the only attached target.

---

## 5. Not included in this package

| File | Size | Why |
|---|---|---|
| `DVCon_SoC_SRC/AS1061_SYSTEM/AS1061_SYSTEM_TOP.edn` | 175 MB | The organiser's VEGA AS1061 RISC-V core netlist. `add_files` in both TCL scripts reads it and **synthesis will not run without it.** Omitted for size; the organisers hold the authoritative copy. |
| `DVCon_SoC_SRC/AS1061_SYSTEM/AS1061_SYSTEM_TOP.v` | 50 MB | Simulation view of the same core. Not read by the synthesis or implementation scripts. |
| `MEMORY_IP/rom_32KB_axi/hdl/`, `sim/`, `simulation/`, `synth/`, `*_sim_netlist.*`, `*.dcp` | ~16 MB | Vivado regenerates all of these from the `.xci`. Only the `.xci`, its `.coe`/`.mif` data files and its `_ooc.xdc` are shipped. |

Everything else the two TCL scripts touch is present here.

---

## 6. Licence and ownership

`run.sh`, `tcl/*.tcl`, `xdc/*.xdc`, `memory_ip/*`, `tb/*` and `../rtl/SOC_TOP/Top.vhd` are
**C-DAC** files, Copyright (c) 2026 C-DAC, reproduced unmodified so that this
package is reproducible on its own (they are **not** included in the public
repository). `rtl/ext/*` is Alex Forencich's AXI infrastructure under the MIT
licence. `rtl/sa/*` and `rtl/sys/*` are adapted from Abarajithan G's
`axis-systolic-array` (MIT licence) and configured/integrated by Team 43.
`Accelerator_Top.v` and `INTEGRATION_NOTES.md` are Team 43's work.
