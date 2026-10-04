# 12 — References and resources

Everything we read, used or built on, grouped by topic.

---

## 1. The problem

| Resource | Link |
|---|---|
| J. Sawatzky, Y. Souri, C. Grund, J. Gall, *What Object Should I Use? — Task Driven Object Detection*, CVPR 2019 | [arXiv:1904.03000](https://arxiv.org/abs/1904.03000) · [doi:10.48550/arXiv.1904.03000](https://doi.org/10.48550/arXiv.1904.03000) |
| COCO-Tasks project page | [coco-tasks.github.io](https://coco-tasks.github.io/) |
| COCO-Tasks dataset (annotations, image lists, detection files) | [github.com/coco-tasks/dataset](https://github.com/coco-tasks/dataset) |
| Reference implementation of the paper | [github.com/yassersouri/task-driven-object-detection](https://github.com/yassersouri/task-driven-object-detection) |
| MS-COCO 2014 dataset | [cocodataset.org](https://cocodataset.org/#download) · [COCO API](https://github.com/cocodataset/cocoapi) |

## 2. Models

| Resource | Link |
|---|---|
| Ultralytics YOLOv8 (n / s / x) | [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) · [docs.ultralytics.com](https://docs.ultralytics.com/) |
| M. Yaseen, *What is YOLOv8: An In-Depth Exploration of the Internal Features of the Next-Generation Object Detector*, 2024 | [arXiv:2408.15857](https://arxiv.org/abs/2408.15857) |
| all-MiniLM-L6-v2 (sentence-transformers) | [huggingface.co/sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Apple MobileCLIP / MobileCLIP2 (tried in Stage 2A, dropped) | [github.com/apple/ml-mobileclip](https://github.com/apple/ml-mobileclip) |

## 3. Hardware platform

| Resource | Link |
|---|---|
| C-DAC VEGA processors | [vegaprocessors.in](https://www.vegaprocessors.in/) · [VEGA AS1061](https://www.vegaprocessors.in/vega-as1061.php) |
| Digilent Genesys 2 | [product page](https://digilent.com/shop/genesys-2-amd-kintex-7-fpga-development-board/) · [reference manual](https://digilent.com/reference/programmable-logic/genesys-2/reference-manual) |
| AMD/Xilinx 7 Series FPGAs overview (DS180) | [docs.amd.com](https://docs.amd.com/v/u/en-US/ds180_7Series_Overview) |
| 7 Series DSP48E1 slice (UG479) | [docs.amd.com](https://docs.amd.com/v/u/en-US/ug479_7Series_DSP48E1) |
| UltraScale DSP48E2 slice (UG579) | [docs.amd.com](https://docs.amd.com/v/u/en-US/ug579-ultrascale-dsp) |
| 7 Series Memory Interface Solutions — MIG (UG586) | [docs.amd.com](https://docs.amd.com/v/u/1.5-English/ug586_7Series_MIS) |
| AXI Interconnect (PG059) | [docs.amd.com](https://docs.amd.com/r/en-US/pg059-axi-interconnect) |
| Block Memory Generator (PG058) | [docs.amd.com](https://docs.amd.com/v/u/en-US/pg058-blk-mem-gen) |
| AXI UART16550 (PG143) | [docs.amd.com](https://docs.amd.com/r/en-US/pg143-axi-uart16550) |
| Arm AMBA AXI and ACE Protocol Specification (AXI4, AXI-Lite, AXI-Stream) | [developer.arm.com](https://developer.arm.com/documentation/ihi0022/latest) |
| The RISC-V Instruction Set Manual, Vol. I and II | [riscv.org/specifications](https://riscv.org/specifications/) |

## 4. Open-source IP we built on

| Resource | Licence | Used for |
|---|---|---|
| Abarajithan G, [`axis-systolic-array`](https://github.com/abarajithan11/axis-systolic-array) | MIT | Systolic array core (`rtl/sa`), DMA controller and AXI wrapper (`rtl/sys`) — adapted and integrated |
| G. Abarajithan, Z. Ma, F. Restuccia, R. Kastner, *FireBridge: Cycle-Accurate Hardware+Firmware Co-Verification for Modern Accelerators* | — | Paper describing the verification flow of the above repository |
| Alex Forencich, [`verilog-axi`](https://github.com/alexforencich/verilog-axi) | MIT | AXI DMA engines, crossbar, AXI-Lite, AXI4→AXI-Lite adapter |
| Alex Forencich, [`verilog-axis`](https://github.com/alexforencich/verilog-axis) | MIT | AXI-Stream width adapters |

## 5. Systolic arrays and accelerators

| Resource | Link |
|---|---|
| H. T. Kung, *Why Systolic Architectures?*, IEEE Computer, 1982 | [doi:10.1109/MC.1982.1653825](https://doi.org/10.1109/MC.1982.1653825) |
| H.-C. Lu, L.-Y. Su, S.-H. Huang, *Highly Fault-Tolerant Systolic-Array-Based Matrix Multiplication*, Electronics 2024 | [doi:10.3390/electronics13091780](https://doi.org/10.3390/electronics13091780) |
| Z. Li, S. Chen, *FPGA-Based Tiled Matrix Multiplication Accelerator for Transformer Self-Attention on the Xilinx KV260 SoM*, 2025 | [arXiv:2503.16731](https://arxiv.org/abs/2503.16731) |
| Wikipedia, *Systolic array* | [link](https://en.wikipedia.org/wiki/Systolic_array) |

## 6. Tools we used

| Tool | Version / notes |
|---|---|
| AMD Vivado | 2025.2 (Stage 3A, Windows), 2020.2 (Stage 3B bitstream, lab PC, CentOS 7); design also checked with 2018.1 and 2026.1 |
| Vivado Simulator (XSim) | Stage 2B and 3A simulation (QuestaSim was also allowed) |
| RISC-V GCC | `riscv64-unknown-elf-gcc` 12.2.0 (bare-metal, Stage 3A); `riscv64-buildroot-linux-gnu-gcc` in the organisers' `buildroot-2004_new` Docker image (Linux, Stage 3B) |
| Docker | to run the organisers' toolchains |
| VEGA Linux | Linux 4.18.0, `rv64imafdc`, BusyBox user space |
| AnyDesk | remote desktop to the C-DAC lab PCs in Stage 3B |
| minicom | serial console to the board |
| Python 3, PyTorch 2.x, Hugging Face Transformers, Ultralytics, NumPy, Pillow | training, export, evaluation |
| Google Colab (Tesla T4) | MiniLM and YOLO training |
| Git, VS Code, LaTeX/Overleaf | code and reports |
| EasyChair | contest submissions |

## 7. Contest material (not redistributed)

These came from the organisers and are **not** in this repository. Ask DVCon India / C-DAC for access:

- *DVCon India 2026 Design Contest — Problem Statement* (Stage 1)
- Stage 1 report template
- C-DAC *VEGA Microprocessor (RISC-V) based Design & Verification Challenge* slides and the *Rapid Guide — Hardware and Bare-metal Application Simulation*
- *VEGA AS1061 datasheet* (DS-VEGA_AS1061 V1.0)
- The DVCon_2026 SoC environment (VEGA netlist, `Top.vhd`, testbench, constraints, ROM IP, TCL/RUN scripts, bare-metal toolchain and Docker image)
- Stage 3B lab environment (Linux image, `send.sh`, buildroot Docker image, accelerator driver)
- Workshop and briefing recordings (Zoom)

## 8. Recommended further reading

Not part of our original reading list, but the papers behind the techniques used here — a good next step if you want to go deeper:

| Resource | Link |
|---|---|
| W. Wang et al., *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*, NeurIPS 2020 | [arXiv:2002.10957](https://arxiv.org/abs/2002.10957) |
| N. Reimers, I. Gurevych, *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*, EMNLP 2019 | [arXiv:1908.10084](https://arxiv.org/abs/1908.10084) |
| B. Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*, CVPR 2018 | [arXiv:1712.05877](https://arxiv.org/abs/1712.05877) |
| N. P. Jouppi et al., *In-Datacenter Performance Analysis of a Tensor Processing Unit*, ISCA 2017 | [arXiv:1704.04760](https://arxiv.org/abs/1704.04760) |
| K. Chellapilla, S. Puri, P. Simard, *High Performance Convolutional Neural Networks for Document Processing* (the im2col idea), 2006 | [HAL inria-00112631](https://inria.hal.science/inria-00112631) |
