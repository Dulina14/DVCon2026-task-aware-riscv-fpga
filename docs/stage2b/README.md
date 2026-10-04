# docs/stage2b/

Notes from Stage 2B (June 2026), when the accelerator was simulated on its own (2×2 array, 32-bit AXI, register base `0xB000_0000`, Vivado XSim). Story: [`docs/05_stage2b_accelerator.md`](../05_stage2b_accelerator.md).

| File | What it is |
|---|---|
| [`ACCELERATION.md`](ACCELERATION.md) / [`acceleration.tex`](acceleration.tex) | Speed-up methodology: measured 2×2 point (52 cycles START→DONE), scaling table for 2×2 … 32×32, mapping to a YOLOv8x convolution |
| [`DESIGN_CONNECTIVITY.md`](DESIGN_CONNECTIVITY.md) | Every block and connection inside `top_axi_int` and in the testbench, with bus widths — the spec we drew our block diagrams from |
| [`WAVEFORM_GUIDE.md`](WAVEFORM_GUIDE.md) / [`waveform.tex`](waveform.tex) | How to read every AXI-Lite, AXI and AXI-Stream transaction in the Stage 2B waveform; self-check passes at 1105 ns |

The testbench and Vivado project referred to in these notes (`tb_top_axi_int.sv`, `axi_ram_model`, `axis-systolic-array-new.sim`) are not part of this repository.
