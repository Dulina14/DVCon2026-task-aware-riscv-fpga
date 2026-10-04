# Third-party notices

This repository contains work by Team 43 (MIT licence, see `LICENSE`) together with third-party material listed below. Each item keeps its own licence.

## Included in this repository

| Component | Where | Origin | Licence |
|---|---|---|---|
| Systolic array core | `hardware/accelerator_ip/rtl/sa/*` | Abarajithan G, [axis-systolic-array](https://github.com/abarajithan11/axis-systolic-array) — adapted | MIT — `hardware/accelerator_ip/rtl/LICENSE-axis-systolic-array.txt` |
| DMA controller, AXI wrapper, skid buffer | `hardware/accelerator_ip/rtl/sys/top_axi_int.v`, `dma_controller.sv`, `skid_buffer.sv` | Abarajithan G, [axis-systolic-array](https://github.com/abarajithan11/axis-systolic-array) — adapted (12-bit IDs, `0x2004_0000` base, `s2mm_tag` tie-off, portable ID assignments) | MIT — same file |
| AXI DMA engines, AXI crossbar, AXI register slices, AXI-Lite RAM/RD/WR, AXI→AXI-Lite adapter, AXI-Stream adapter, arbiter, priority encoder | `hardware/accelerator_ip/rtl/ext/*` | Alex Forencich, [verilog-axi](https://github.com/alexforencich/verilog-axi) and [verilog-axis](https://github.com/alexforencich/verilog-axis) | MIT — header in each file; `hardware/accelerator_ip/rtl/LICENSE-verilog-axi.txt` |
| YOLOv8n / YOLOv8s INT8 packages | `models/team43_yolo_int8.bin`, `models/team43_yolo_s_int8.bin` | Derived from [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) weights | AGPL-3.0 |
| MiniLM INT8 package, vocabulary | `models/team43_minilm_*`, `models/vocab.txt` | Derived from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), fine-tuned by Team 43 | Apache-2.0 |
| Demo photo and output images | `test_data/input/`, `results/images/`, `stage2_software_pipeline/results/result.jpg` | MS-COCO 2014 images (and our marked versions) | [COCO terms of use](https://cocodataset.org/#termsofuse); images under their Flickr licences |
| Event photographs | `docs/images/journey/` | Team 43 / DVCon India 2026 attendees | Shared by the team for this repository |

The Python code in `stage2_software_pipeline/`, `offline_export/` and `evaluation/` uses (but does not include) PyTorch (BSD-3-Clause), Hugging Face Transformers (Apache-2.0), Ultralytics (AGPL-3.0), NumPy and Pillow. The scripts in `stage2_software_pipeline/experiments/mobileclip_abandoned/` were written to run inside a clone of [apple/ml-mobileclip](https://github.com/apple/ml-mobileclip), which is not included.

## Not included — organiser / C-DAC property

The following were provided by the DVCon India 2026 Design Contest organisers and **C-DAC** for contest use and are **not** redistributed here. Contact the organisers for access.

- VEGA AS1061 processor netlist (`AS1061_SYSTEM_TOP.edn` / `.v`)
- SoC top level `Top.vhd`, testbench, XDC constraints, MIG and ROM IP, bootloader `.coe`
- Organiser scripts: `run.sh`, `DVCon_SIM.tcl`, `DVCon_SYN.tcl`, `DVCon_IMP.tcl`, `FPGA_PROGRAMMING.tcl`, `QuestaSim_SIM.do`
- The Genesys 2 bitstream (it contains the VEGA core)
- Bare-metal toolchain, Docker images (`dvcon_2`, `buildroot-2004_new`), VEGA Linux image, device tree, `send.sh`, the accelerator Linux driver
- Bare-metal support files (`crt.S`, `uart.c`, `stdlib.c`, `xmodem*.c`, `mbl.lds`, `Makefile`, `include/*`)
- Contest documents: problem statement, report templates, slides, Rapid Guide, VEGA datasheet

## Not included — datasets

- MS-COCO 2014 images and annotations — https://cocodataset.org
- COCO-Tasks annotations and detection files — https://github.com/coco-tasks/dataset
