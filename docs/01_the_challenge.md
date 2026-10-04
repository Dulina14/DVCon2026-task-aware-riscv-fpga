# 01 — The challenge

This page explains what the contest asked for, the research problem behind it, and the hardware platform every team had to use. If you are new to the topic, read this first.

---

## 1. DVCon India and its Design Contest

**DVCon** (Design and Verification Conference and Exhibition) is a conference series about chip design and verification, sponsored by the **Accellera Systems Initiative**. **DVCon India 2026** was the 11th edition. It ran from **1 to 3 September 2026** at the Radisson Blu Hotel, Marathahalli, **Bengaluru**, India, with the theme *"Architecture to Analysis"*.

Each year DVCon India runs a **Design Contest** for university students (teams of two or three). In 2026 the contest was run with **C-DAC** (Centre for Development of Advanced Computing, Thiruvananthapuram) as the technical partner. C-DAC develops the **VEGA** family of Indian RISC-V processors under the Digital India RISC-V (DIR-V) programme. On the conference floor the contest was presented as *"C-DAC's VEGA Microprocessor (RISC-V) based Design & Verification Challenge"*.

The contest was announced in a problem-statement webinar on **23 February 2026**. More than 650 teams from India and Sri Lanka took part. Teams were removed stage by stage until **12 teams** reached the hardware stage and **6 finalists** were invited to Bengaluru.

---

## 2. The problem: task-driven object detection

### 2.1 Why a normal detector is not enough

A normal object detector (for example YOLO) finds *every* object in a picture and names it: "person, cup, chair, wine glass…". It does not know what you want to *do*. If your task is *"serve wine"*, you do not want all objects — you want the one object that is best for that job.

The contest problem statement put it this way:

> *"Conventional object detection techniques are task-agnostic – they detect all objects in a scene and lack the capability to adapt their detection behavior dynamically based on the task context… there is a need for a task-driven object detection framework that selectively detects and prioritizes objects relevant to a given task description or goal."*

### 2.2 What teams had to build

From the problem statement:

1. Design a **task-aware object selection** framework, using the **COCO dataset** and the **14 tasks** of the reference work.
2. Build an **edge-compatible** detection pipeline that uses **lightweight language models** to take the task (a text prompt) into account.
3. **Deploy it on an edge device**: the **VEGA processor** (developed by C-DAC) on a **Digilent Genesys 2 FPGA board**, together with **your own accelerator**.
4. **Input** at run time: one image + one of the 14 tasks.
5. **Output**: the most appropriate object in the image for that task.

### 2.3 How solutions were judged

The evaluation criteria listed in the problem statement:

1. Novelty of the proposed detection pipeline
2. Accuracy and relevance of the detection for the given tasks
3. Inference latency (how long one answer takes)
4. CPU and FPGA utilisation (how much of the chip you use)
5. Power / energy consumption

---

## 3. The reference paper and dataset

**Paper:** J. Sawatzky, Y. Souri, C. Grund, J. Gall, *"What Object Should I Use? — Task Driven Object Detection"*, CVPR 2019, University of Bonn. [arXiv:1904.03000](https://arxiv.org/abs/1904.03000) · [project page](https://coco-tasks.github.io/) · [code](https://github.com/yassersouri/task-driven-object-detection) · [dataset](https://github.com/coco-tasks/dataset)

The paper created the **COCO-Tasks** dataset. It re-uses photos from MS-COCO 2014 (about 40,000 images) and, for each of 14 tasks, marks which objects in the photo are **preferred** for that task. The key idea: the label depends on the sentence, not only on the picture. For "serve wine", a wine glass is preferred if one is present; if not, a cup becomes the preferred object.

The 14 tasks (we used them exactly as written below):

| # | Task | # | Task |
|---|---|---|---|
| 1 | step on something to reach top of a shelf | 8 | open bottle of beer |
| 2 | sit comfortably | 9 | open parcel |
| 3 | place flowers | 10 | serve wine |
| 4 | get potatoes out of fire | 11 | pour sugar |
| 5 | water plant | 12 | smear butter |
| 6 | get lemon out of tea | 13 | extinguish fire |
| 7 | dig hole | 14 | pound carpet |

The paper's method uses a Faster R-CNN detector plus a **Gated Graph Neural Network** on a desktop GPU. Its main metric is **mAP@0.5**: average precision of the preferred-object boxes at 50 % box overlap, averaged over the 14 tasks. Its best result is 0.742 with perfect (ground-truth) boxes and 0.326 with Faster R-CNN boxes.

> **mAP@0.5 in plain words.** For each task, rank every candidate box by score. A box counts as correct if it overlaps a preferred object by at least 50 % (IoU ≥ 0.5). Average precision summarises how high the correct boxes are in that ranking. mAP is the mean over the 14 tasks. **Top-1** simply asks: is the single best box correct?

---

## 4. The platform every team had to use

### 4.1 The board — Digilent Genesys 2

| Item | Value |
|---|---|
| FPGA | AMD/Xilinx Kintex-7 **XC7K325T-2FFG900** (Vivado part `xc7k325tffg900-2`) |
| Logic | 203,800 LUTs, 407,600 flip-flops, 445 block-RAM tiles (36 Kb each), 840 DSP slices |
| Memory | 1 GB DDR3 |
| I/O used by us | USB-UART (console), Gigabit Ethernet (file transfer and Linux boot), USB-JTAG (programming) |
| Links | [product page](https://digilent.com/shop/genesys-2-amd-kintex-7-fpga-development-board/) · [reference manual](https://digilent.com/reference/programmable-logic/genesys-2/reference-manual) |

<p align="center"><img src="images/journey/photo_6.jpg" width="420" alt="A Digilent Genesys 2 board with UART, Ethernet and JTAG cables connected"/><br/><em>A Genesys 2 board as we used it in the finals: power, UART, Ethernet and JTAG cables.</em></p>

### 4.2 The processor — C-DAC VEGA AS1061

| Item | Value (from C-DAC's datasheet and contest slides) |
|---|---|
| ISA | 64-bit RISC-V (RV64G; the contest Linux reports `rv64imafdc`) |
| Pipeline | 6-stage, in-order, single core, with branch prediction (BTB, BHT, RAS) |
| Other | MMU, machine + supervisor modes, PLIC interrupt controller, JTAG debug |
| Clock in the contest SoC | **50 MHz** |
| Runs | Linux (used in Stage 3B and the finals), or bare-metal programs (used in Stage 3A) |
| Link | [vegaprocessors.in — AS1061](https://www.vegaprocessors.in/vega-as1061.php) |

> **Important practical point.** In the contest Linux build, floating-point maths was done by software library calls, so every `float` operation was slow. About 39 % of our run time was software floating point. This shaped many design decisions (see [docs/11](11_lessons_learned.md)).

### 4.3 The SoC provided by the organisers

The organisers supplied a complete **VEGA SoC** (as a netlist + VHDL top level + constraints + scripts). It contained:

- the AS1061 core,
- an **AXI4 interconnect** (64-bit address, 64-bit data, 12-bit ID, 50 MHz),
- 32 KB boot ROM and 128 KB internal RAM,
- a DDR3 memory controller (Xilinx MIG, 1 GB),
- UART (16550), Gigabit Ethernet (RGMII), PLIC,
- and an **empty accelerator slot**: one AXI4 slave port (for the CPU to write registers) and one AXI4 master port (for the accelerator to read and write memory), in the 50 MHz clock domain.

An example accelerator was provided; each team replaced it with its own. The SoC top level (`Top.vhd`) already declared a component called `Accelerator_Top` — teams had to supply that module.

The organisers' resource figure for the SoC alone was roughly 55 % of LUTs, 20 % of registers, 15 % of block RAM and 10 % of DSPs, so the accelerator had to fit in what was left. Informally, teams were advised to keep the accelerator within about 20–25 % of the device.

### 4.4 Rules that mattered

For Stage 3A the organisers required (from their *Evaluation & Selection Criteria*):

- Use **only** the provided design files, scripts and project structure. The example accelerator may be replaced.
- Simulation with **AMD Vivado Simulator (XSim) or QuestaSim only**; synthesis with **AMD Vivado only**, targeting the Genesys 2.
- The submission must be **fully reproducible** by the organisers on their own machines without any extra files or edits — RTL, accelerator source, bare-metal application, constraints, scripts, waveforms and Vivado reports.
- Submissions with missing or non-reproducible files may be rejected.

For the hardware stages, the **standalone rule** applied: the host PC may convert and quantise models offline, but it must **not** compute anything that depends on the run-time image or task. Our interpretation and audit of this rule is in [TECHNICAL_REPORT.md §8](TECHNICAL_REPORT.md#8-the-standalone-boundary).

---

## 5. Where to go next

- How the contest unfolded for us, stage by stage: [02 — Journey timeline](02_journey_timeline.md)
- Our final system design: [09 — System architecture](09_system_architecture.md)
