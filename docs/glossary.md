# Glossary — plain-language meanings

Terms are grouped by topic. Where a term names a place where data lives, that is said explicitly.

## Hardware and chips

| Term | Meaning |
|---|---|
| **FPGA** | *Field-Programmable Gate Array.* A chip made of thousands of small logic blocks whose wiring you set by loading a file. You can turn it into almost any digital circuit. |
| **Bitstream** (`.bit`) | The file that configures an FPGA. Our bitstream contains the VEGA SoC plus our accelerator. The FPGA forgets it when power is removed. |
| **LUT** | *Look-Up Table.* The basic logic block of an FPGA. Our accelerator uses 11,945 of the 203,800 available. |
| **FF / register** | *Flip-flop.* A 1-bit storage element updated on each clock tick. |
| **BRAM** | *Block RAM.* Small on-chip memories inside the FPGA (36 Kb each). Our accelerator uses none. |
| **DSP slice** | A hard multiply-add block inside the FPGA. Our accelerator uses none — its multipliers are built from LUTs. |
| **RTL** | *Register-Transfer Level.* Hardware described in a language such as Verilog/SystemVerilog/VHDL. |
| **Synthesis** | The tool step that turns RTL into LUTs and flip-flops. |
| **Implementation / place-and-route** | The step after synthesis that places the logic on the chip and wires it, then checks timing. Produces the bitstream. |
| **WNS** | *Worst Negative Slack.* How much spare time the slowest path has inside one clock period. Positive = timing is met. Ours: +0.556 ns at 50 MHz. |
| **Netlist** | A circuit already turned into gates. C-DAC gave the VEGA core as a netlist, so its source is not visible. |
| **SoC** | *System on Chip.* A CPU together with buses, memory controllers and peripherals in one design. |
| **RISC-V** | An open instruction-set architecture (the "language" a CPU understands). VEGA is a RISC-V CPU. |
| **VEGA AS1061** | C-DAC's 64-bit, single-core RISC-V processor used in the contest, running at 50 MHz. |
| **Genesys 2** | The Digilent FPGA board (Kintex-7 XC7K325T, 1 GB DDR3) used in the contest. |
| **DDR3** | The board's main memory (1 GB, outside the FPGA). Model weights, activations and results **live here** while the program runs. |
| **SRAM (internal RAM)** | 128 KB of memory inside the SoC. In Stage 3A simulation, all our test buffers **lived here** because DDR was not modelled. |
| **Boot ROM / MIF / COE** | The memory the CPU starts from. A `.mif`/`.coe` file is the content loaded into it. In Stage 3A our test program **lived here**. |
| **JTAG** | A cable/port used to load the bitstream into the FPGA. |
| **UART** | A simple serial port. We read the board's console through it (with `minicom`). |

## Buses and data movement

| Term | Meaning |
|---|---|
| **AXI4** | Arm's standard on-chip bus for memory-mapped reads and writes with bursts. Our accelerator reads DDR and writes results **from/to DDR through the SoC's AXI interconnect**. |
| **AXI-Lite** | A simple version of AXI for single register reads/writes. The CPU sets our accelerator's registers through it. |
| **AXI-Stream (AXIS)** | A one-way stream of data words with `valid`/`ready` handshakes and a `last` flag; no addresses. Inside our accelerator, data flows **from the DMA engines to the array** as streams. |
| **Handshake (`VALID`/`READY`)** | A transfer happens only on a clock edge where the sender says "valid" and the receiver says "ready". |
| **Burst length** | How many words one AXI transaction moves. Ours is 1 (one 64-bit word per transaction) — the main remaining inefficiency. |
| **DMA** | *Direct Memory Access.* A small engine that copies data **between memory and the accelerator** without the CPU doing it. We have 3 read DMAs (K, X, B) and 1 write DMA (Y). |
| **MM2S / S2MM** | Memory-to-stream (read DMA) / stream-to-memory (write DMA). |
| **Crossbar** | A switch that lets several bus masters share one port. Ours merges the four DMAs into the accelerator's single memory port. |
| **Skid buffer** | A 2-entry register stage on a stream that keeps the handshake timing clean without losing data. |
| **Register map** | The list of addresses the CPU writes to control the accelerator (START, addresses, byte counts, DONE flags). |
| **Character device `/dev/accelerator_test`** | The Linux file through which our program talks to the accelerator. Each `pread`/`pwrite` is one system call. |
| **System call** | A request from a program to the Linux kernel. On this board each one cost 60–90 µs — far more than the accelerator's actual compute. |

## Accelerator

| Term | Meaning |
|---|---|
| **Systolic array** | A grid of multiply-add cells where numbers move one cell per clock tick, so many multiplications happen at once. Ours is 8×8 = 64 cells. |
| **PE** | *Processing Element.* One cell of the array. |
| **MAC** | *Multiply-ACcumulate:* `sum = sum + a × b`. One tile = 512 MACs. |
| **Output-stationary** | Each cell keeps its own running sum; the inputs move, the sums stay put. |
| **Skew / `tri_buffer`** | Delaying row *r* by *r* cycles and column *c* by *c* cycles so the right numbers meet in the right cell. |
| **Tile** | An 8×8 block of a matrix — the unit of work for one accelerator call. |
| **GEMM** | *General Matrix Multiply*, `Y = X·K + B`. Almost all neural-network work can be expressed as GEMMs. |
| **im2col** | Copying image patches into the rows of a matrix so that a convolution becomes a GEMM. |
| **K-depth collapse** | Our optimisation: stream the whole dot-product length (K) into one accelerator call instead of 8 values per call. 3.3× faster. |
| **INT8 / INT32** | 8-bit and 32-bit whole numbers. Weights and activations are INT8; sums are INT32. |
| **Quantisation** | Converting floating-point numbers to INT8 with a scale factor. "Per-channel" = one scale per output channel. |

## Machine learning

| Term | Meaning |
|---|---|
| **Object detection** | Finding objects in an image and drawing a box with a class name and confidence around each. |
| **YOLOv8n / s / x** | Ultralytics' YOLO version 8 detectors: *nano*, *small*, *extra-large*. Bigger = more accurate but slower. |
| **NMS** | *Non-Maximum Suppression.* Removes duplicate boxes that cover the same object. |
| **MiniLM** | A small transformer language model. We fine-tuned *all-MiniLM-L6-v2* to score how well an object suits a task. |
| **Embedding** | A list of numbers (here 384) that represents the meaning of a piece of text. |
| **WordPiece / tokeniser** | Splits text into sub-word tokens from a fixed vocabulary (30,522 tokens). Runs on the board. |
| **Fine-tuning** | Further training a pre-trained model on your own data. |
| **Top-1** | Is the single highest-ranked answer correct? |
| **mAP@0.5** | Mean Average Precision with boxes counted correct at ≥ 50 % overlap (IoU). The reference paper's metric. |
| **IoU** | *Intersection over Union*: overlap area ÷ combined area of two boxes. |
| **COCO / COCO-Tasks** | COCO: a large dataset of everyday photos with 80 object classes. COCO-Tasks: the reference paper's labels saying which objects are preferred for each of 14 tasks. |

## Project-specific

| Term | Meaning |
|---|---|
| **`--sim`** | Run the board program with the accelerator replaced by an exact software copy. Works on any PC. |
| **`fpcheck`** | A fingerprint line printed by every run. If the PC and the board print the same line, their floating-point maths agrees. |
| **`-ffp-contract=off`** | Compiler flag that stops GCC from fusing `a*b+c` into one instruction, so x86 and RISC-V round identically. |
| **`BCSP`** | The UART text our Stage 3A test program prints: Booted, Configured, Started, Passed. |
| **Golden model / golden reference** | An independent calculation of the correct answer, used to check the hardware. |
| **VCD** | *Value Change Dump*, the waveform file from a simulation. Our analyser reads it to check every bus transfer. |
| **Probe** | One of our small test programs that answers one hardware question. |
| **`/tmp` RAM disk** | On the board, `/tmp` lives in memory. Files copied there **are lost** when the board is powered off. |
| **AnyDesk** | Remote-desktop software we used to control the C-DAC lab PCs in Stage 3B. |
| **Lab PC / development PC** | The CentOS machine in the C-DAC lab, connected to the board, where we built and deployed. |
