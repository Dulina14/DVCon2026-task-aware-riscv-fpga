# hardware/

The accelerator RTL, its integration into the C-DAC VEGA SoC, the Stage 3A simulation and synthesis flow, and the post-route reports of the board build.

```
hardware/
├── accelerator_ip/                 ← drop-in replacement for the SoC kit's DVCon_SoC_SRC/ACCELERATOR_IP/
│   ├── Accelerator_Top.v           ← Team 43: wrapper matching the component declared in Top.vhd
│   ├── INTEGRATION_NOTES.md        ← Team 43: how it attaches to the SoC (widths, address map, checklist)
│   └── rtl/
│       ├── sa/    axis_sa.sv, pe.sv, mac.sv, tri_buffer.sv, n_delay.sv     ← systolic array
│       ├── sys/   top_axi_int.v, dma_controller.sv, skid_buffer.sv, config.svh ← DMA + AXI wrapper
│       ├── ext/   AXI DMA, crossbar, AXI-Lite, adapters                    ← Alex Forencich (MIT)
│       └── LICENSE-*.txt
├── stage3a_simulation/             ← Stage 3A: self-checking firmware, golden model, VCD analyser, TCL, results
├── reports/                        ← post-route Vivado 2020.2 reports from the Stage 3B board build
└── REBUILD.md                      ← how the bitstream was rebuilt in the organiser environment
```

## Provenance — please read

| Files | Origin | Team 43 changes |
|---|---|---|
| `rtl/sa/*`, `rtl/sys/top_axi_int.v`, `rtl/sys/dma_controller.sv`, `rtl/sys/skid_buffer.sv` | **Adapted from** [Abarajithan G — `axis-systolic-array`](https://github.com/abarajithan11/axis-systolic-array), MIT licence ([LICENSE](accelerator_ip/rtl/LICENSE-axis-systolic-array.txt)) | `AXI_ID_WIDTH` 6→12, `AXIL_BASE_ADDR` → `0x2004_0000`, `s2mm_tag` tied off, Vivado-portable ID assignments |
| `rtl/sys/config.svh` | Generated-config style of the same project | R = C = 8, WX = WK = 8, WY = 32, AXI 64-bit, address 64-bit |
| `rtl/ext/*` | [Alex Forencich — `verilog-axi`](https://github.com/alexforencich/verilog-axi) / [`verilog-axis`](https://github.com/alexforencich/verilog-axis), MIT (headers kept in every file; `alex_` prefix added by the upstream bundler) | none |
| `Accelerator_Top.v`, `INTEGRATION_NOTES.md`, everything in `stage3a_simulation/` | **Team 43** | — |

**Not included** (organiser / C-DAC property — get them from the contest organisers): the VEGA AS1061 netlist, `Top.vhd`, the SoC testbench, XDC constraints, ROM IP, bootloader, the organiser TCL/RUN scripts, the bitstream, the bare-metal toolchain and Docker images.

## Key facts

| | |
|---|---|
| Array | 8 × 8 output-stationary systolic array, INT8 × INT8 → INT32, `Y = X·K + B` per 8×8×8 tile (512 MACs) |
| Interfaces | AXI4 64-bit slave (config, bridged to 32-bit AXI-Lite) + AXI4 64-bit master (memory), 12-bit IDs, 50 MHz |
| Register base | `0x2004_0000` (set in `top_axi_int.v`; the `0xB000_0000` in `config.svh` is unused) |
| Post-synthesis (Stage 3A, Vivado 2025.2) | 12,404 LUTs (6.09 %), 12,711 FFs, 0 BRAM, 0 DSP |
| Post-route (Stage 3B, Vivado 2020.2) | **11,945 LUTs (5.86 %)**, 12,706 FFs, 0 BRAM, 0 DSP; WNS +0.556 ns at 50 MHz |

## Where to read more

- Module-by-module walkthrough: [`docs/code_guides/CODE_GUIDE_RTL.md`](../docs/code_guides/CODE_GUIDE_RTL.md)
- Data path, register map, memory map: [`docs/09_system_architecture.md`](../docs/09_system_architecture.md)
- Stage 3A story: [`docs/06_stage3a_soc_integration.md`](../docs/06_stage3a_soc_integration.md)
- Re-running simulation/synthesis: [`docs/10_reproduce.md` §6](../docs/10_reproduce.md#6-level-5--simulate-and-synthesise-the-accelerator-inside-the-soc)
