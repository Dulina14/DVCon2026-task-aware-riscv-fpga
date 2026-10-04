DVCon India 2026 — Stage 3B

**Task-Aware Object Selection**

*A Standalone Edge Pipeline on the VEGA RISC-V SoC with an 8×8 INT8 FPGA Accelerator*

**Team 43 — Silicon Syndicate**

Digilent Genesys 2 · Kintex-7 XC7K325T · VEGA AS1061

**Stage 3B Implementation Report**

18 August 2026

# Contents

**1 Executive Summary**

> Headline results

**2 Project Objective and Final Standalone Pipeline**

> The fourteen supported tasks

**3 System Architecture**

> 3.1 YOLOv8n detection
> 
> 3.2 MiniLM task-aware selection
> 
> 3.3 VEGA and FPGA partitioning

**4 The 8×8 INT8 Accelerator**

> 4.1 RTL organisation
> 
> 4.2 Fabric cost
> 
> 4.3 Clocking and timing closure
> 
> 4.4 The invocation sequence

**5 Offline Preparation and the Standalone Boundary**

**6 Hardware Validation Results**

> 6.1 Numerical validation

**7 Stage 3B Submission Summary**

> 7.1 Location of the project folder
> 
> 7.2 Work done and what has been achieved
> 
> 7.3 Completion status
> 
> 7.4 Adherence to the original solution required
> 
> 7.5 Implementation details and utilisation
> 
> 7.6 Path forward if a demonstration is required at the conference

**8 Performance Analysis**

> 8.1 Measured driver primitives
> 
> 8.2 Where the time goes
> 
> 8.3 The interface cost, quantified
> 
> 8.4 Memory behaviour of the VEGA core

**9 Accuracy Evaluation**

> 9.1 Confidence in these figures

**10 Current Limitations**

> 10.1 Latency
> 
> 10.2 No instance-level discrimination
> 
> 10.3 Detection recall at reduced resolution
> 
> 10.4 Task coverage
> 
> 10.5 Accelerator utilisation
> 
> 10.6 Known open items

**11 Why YOLOv8n Was Selected for This Stage**

**12 Future Path Toward YOLOv8x**

**13 Deliverable Package**

# 1 Executive Summary

This report documents a complete, standalone, task-aware object-selection pipeline running on the Digilent Genesys 2 board. The system accepts an ordinary image and a natural-language task prompt, detects the objects present, decides which of those objects best serves the stated task, and writes back the same image with the selected object marked. Every arithmetic step of both neural networks executes on the board: the general-matrix-multiply work runs on a custom 8×8 INT8 FPGA accelerator, and the remaining operators run on the VEGA AS1061 RISC-V soft processor.

The problem addressed is task-driven object detection as defined by Sawatzky, Souri, Grund and Gall (CVPR 2019), together with their COCO-Tasks dataset of fourteen everyday tasks. Where the reference work uses a Faster R-CNN detector and a Gated Graph Neural Network on desktop hardware, this project targets the same problem with a deliberately lightweight pair of models — YOLOv8n for detection and a fine-tuned MiniLM ranker for task-aware selection — quantized to INT8 and executed on an embedded FPGA platform.

All four development milestones have been demonstrated on hardware. The end-to-end pipeline consumes 517,888 accelerator invocations and completes in 538.8 seconds per query, producing output numerically identical to an independent host-side reference. On the ground-truth-box protocol of the reference paper the ranking stage achieves 0.648 mAP@0.5, and on the paper’s own Faster R-CNN detections it achieves 0.288 mAP@0.5 — equal to their ResNet-101 classification baseline and 88 % of their full Gated Graph Neural Network result, using a ranker that never sees a pixel.

## Headline results

|                                          |                                     |                                       |
| ---------------------------------------- | ----------------------------------- | ------------------------------------- |
| **Item**                                 | **Result**                          | **Evidence**                          |
| **End-to-end pipeline on hardware**      | **PASS, 538.8 s per query**         | **Milestone D board log**             |
| Accelerator invocations per query        | 517,888                             | program counter output                |
| Numerical agreement with host reference  | exact to 6 decimal places           | four independent implementations      |
| INT8 cost versus FP32 reference          | \+0.0006 on the top score           | MiniLM golden comparison              |
| Accelerator fabric cost                  | 11,945 LUTs (5.86 %), 0 DSP, 0 BRAM | Vivado hierarchical utilization       |
| System clock                             | 50 MHz, timing met (WNS +0.556 ns)  | Vivado timing summary                 |
| mAP@0.5, ground-truth boxes              | 0.648                               | COCO-Tasks test split, 12,588 rows    |
| mAP@0.5, paper’s Faster R-CNN detections | 0.288                               | same detections as the reference work |

# 2 Project Objective and Final Standalone Pipeline

The objective of Stage 3B was to demonstrate a genuinely standalone application on the Genesys 2 platform: one deployed executable and its model files, handling different image and task pairs at runtime without recompilation, with no host computation of any input-dependent quantity.

The final deliverable is a single RISC-V executable, team43\_pipeline, which performs the following sequence entirely on the board:

image (.jpg or .ppm) + task string, both from argv

\-\> JPEG decode (baseline DCT, written for this project) VEGA

\-\> letterbox to the inference size, normalise VEGA

\-\> YOLOv8n, 103-op graph, 63 convolutions as im2col GEMM ACCELERATOR

\-\> SiLU, max-pool, upsample, concat, slice, residual add VEGA

\-\> DFL soft-max, box decode, sigmoid, non-max suppression VEGA

\-\> WordPiece tokenisation of the task prompt VEGA

\-\> MiniLM: 6 encoder layers, Q/K/V/O, QK^T, A.V, FFN1, FFN2 ACCELERATOR

\-\> LayerNorm, soft-max, GELU, residual, mean-pool, L2-norm VEGA

\-\> trained head over the detected classes (2 GEMMs) ACCELERATOR

\-\> ranking, box drawing, image write-back VEGA

A representative invocation and its measured hardware result:

/tmp/team43/team43\_pipeline --image /tmp/team43/demo\_serve\_wine\_glasses.ppm \\

\--task "serve wine" --imgsz 128 --out /tmp/team43/result.ppm

\--- DETECTED (4) ---

wine glass 0.668 | wine glass 0.642 | wine glass 0.505 | dining table 0.369

\--- TASK-AWARE RANKING (minilm) ---

wine glass yolo 0.668 minilm 0.97006 rank 0.97006 \<== SELECTED

dining table yolo 0.369 minilm 0.00117 rank 0.00117

SELECTED : wine glass

total : 517,888 invocations, 538.8 s

The result is a strong illustration of the reference paper’s central argument. Its Figure 1 contrasts two scenes: where a wine glass is present it is preferred over other drinking vessels, and where no wine glass exists a cup is chosen instead. Both cases were reproduced on the board — the image above selects the wine glass, and a second test image containing only a cup selects the cup at 0.934.

## The fourteen supported tasks

The MiniLM ranker was fine-tuned on the fourteen COCO-Tasks prompts and only those. Prompt strings outside this set are outside the training distribution and their output carries no meaning. The exact strings are:

|        |                                           |        |                     |
| ------ | ----------------------------------------- | ------ | ------------------- |
| **\#** | **Task prompt**                           | **\#** | **Task prompt**     |
| 1      | step on something to reach top of a shelf | 8      | open bottle of beer |
| 2      | sit comfortably                           | 9      | open parcel         |
| 3      | place flowers                             | 10     | serve wine          |
| 4      | get potatoes out of fire                  | 11     | pour sugar          |
| 5      | water plant                               | 12     | smear butter        |
| 6      | get lemon out of tea                      | 13     | extinguish fire     |
| 7      | dig hole                                  | 14     | pound carpet        |

Measured behaviour on the deployed INT8 package is sound across all fourteen. Representative top-ranked classes: serve wine → wine glass 0.970 then cup 0.934; sit comfortably → couch 0.985, chair 0.981; place flowers → vase 0.961; smear butter → knife 0.953; open parcel → knife 0.971, scissors 0.967; dig hole → baseball bat and tennis racket, reflecting the dataset’s preference for long rigid implements.

# 3 System Architecture

## 3.1 YOLOv8n detection

Detection uses YOLOv8n with BatchNorm folded into the preceding convolutions and per-output-channel symmetric INT8 quantization of the weights. The network was flattened offline from its PyTorch module hierarchy into a linear list of 103 operations referencing 63 convolution weight tensors, with the C2f split/concatenate structure, the SPPF max-pool chain, the nearest-neighbour upsamples and the detection head all expanded into explicit primitive operations. The board executes this list with a small interpreter; no model-specific control flow is compiled into the binary.

Every convolution is lowered to an im2col general matrix multiply and executed on the accelerator. The image is letterboxed to a runtime-selectable inference size — 128, 192 or 256 pixels — which changes only the tensor shapes, not the operation list, so a single binary and a single weight package serve all resolutions.

|                    |                          |                             |                                |
| ------------------ | ------------------------ | --------------------------- | ------------------------------ |
| **Inference size** | **Multiply-accumulates** | **Accelerator invocations** | **Estimated accelerator time** |
| 128 × 128          | 0.175 G                  | 342,160                     | ≈ 122 s                        |
| 192 × 192          | 0.394 G                  | 785,432                     | ≈ 280 s                        |
| 256 × 256          | 0.700 G                  | 1,368,640                   | ≈ 487 s                        |

## 3.2 MiniLM task-aware selection

Selection uses all-MiniLM-L6-v2 fine-tuned as a task–object suitability scorer: hidden size 384, six encoder layers, twelve attention heads, feed-forward width 1536, vocabulary 30,522, with the deployed sequence length reduced to eight tokens because the longest supported prompt tokenises to seven. The scoring function embeds the strings "task: \<prompt\>" and "object: \<class name\>", forms the fused vector \[t, o, |t − o|, t ⊙ o\] of width 1536, and passes it through a two-layer trained head to a sigmoid suitability score.

One consequence of this design shapes the whole system. The ranker consumes the task string and the object’s class name; it never sees image pixels. The eighty COCO label embeddings are therefore independent of any runtime input and are computed once offline and shipped inside the model package — mathematically identical to computing them on the board, at no runtime cost. The task embedding depends on the runtime prompt and is computed on the board, and the trained head is two matrix multiplies that run on the accelerator, so the final selection genuinely depends on accelerator output.

In the integrated pipeline the head evaluates only the classes actually detected rather than all eighty. For a scene with four distinct detected classes this reduces the head from 92,640 invocations to 9,264, and the complete MiniLM stage from 259,104 invocations to 175,728.

## 3.3 VEGA and FPGA partitioning

The partition follows a simple rule: anything expressible as a dense INT8 matrix multiply goes to the accelerator; everything else runs on the VEGA core.

|                                          |                                               |
| ---------------------------------------- | --------------------------------------------- |
| **Executed on the 8×8 accelerator**      | **Executed on the VEGA RISC-V core**          |
| All 63 YOLOv8n convolutions (via im2col) | JPEG decode, letterbox, normalisation         |
| MiniLM Q, K, V and output projections    | SiLU, max-pool, upsample, concatenate, slice  |
| MiniLM QKᵀ and attention·value products  | Residual addition, DFL soft-max, box decode   |
| MiniLM FFN1 and FFN2                     | Non-max suppression, sigmoid                  |
| Trained head layers head.0 and head.3    | LayerNorm, attention soft-max, GELU           |
|                                          | Mean-pool, L2 normalisation, ranking, drawing |

The device interface is the organiser-supplied character device /dev/accelerator\_test. Its pread and pwrite counts are expressed in 32-bit words rather than bytes. Control is through a small register file — start, done, and address, byte-count and stream-identifier registers for the weight, activation, bias and result streams — with four one-megabyte DDR scratch windows for operands and results.

# 4 The 8×8 INT8 Accelerator

The accelerator computes one 8×8×8 INT8 tile product per invocation: Y = X · K + B, where X is an 8×8 INT8 activation tile in column-major order, K is an 8×8 INT8 weight tile in row-major order, and B and Y are 8×8 INT32 accumulator tiles in column-major order. Each invocation performs 512 multiply-accumulate operations.

Accumulation along the contraction dimension uses the bias path. The accumulator is seeded with the broadcast bias, and for each successive tile along that dimension the running partial result is presented as the bias input, so the hardware accumulates without the host performing any addition.

## 4.1 RTL organisation

The design is an 8×8 systolic array of processing elements. Its RTL is organised into three groups: the array itself (processing element, multiply-accumulate cell, triangular input buffer, delay line and the AXI-Stream array wrapper), the system layer (DMA controller, skid buffers, the AXI top-level integration and the configuration header), and a set of external AXI infrastructure modules providing the crossbar, AXI-Lite register interface and read/write DMA engines.

The configuration header fixes the parameters the software was written against:

|                  |              |                                            |
| ---------------- | ------------ | ------------------------------------------ |
| **Parameter**    | **Value**    | **Meaning**                                |
| R × C            | 8 × 8        | processing-element array dimensions        |
| WK, WX           | 8, 8         | weight and activation operand width (INT8) |
| WY               | 32           | accumulator and result width (INT32)       |
| AXI\_WIDTH       | 64           | data-path width for operand streaming      |
| AXIL\_WIDTH      | 32           | control register width                     |
| AXIL\_BASE\_ADDR | 0xB000\_0000 | control register file base                 |

The control interface exposes one start register, three memory-to-stream DMA channels — used here for the weight tile, the activation tile and the bias/accumulator tile, each with address, byte-count, stream-identifier and done registers — and one stream-to-memory channel for the result. This is exactly the register layout the application software drives.

## 4.2 Fabric cost

From the Vivado hierarchical utilization report for the implemented design:

|                 |                 |                   |               |
| --------------- | --------------- | ----------------- | ------------- |
| **Resource**    | **Accelerator** | **Whole SoC**     | **Available** |
| Slice LUTs      | 11,945 (5.86 %) | 107,937 (52.96 %) | 203,800       |
| Slice registers | 12,706 (3.12 %) | 64,654 (15.86 %)  | 407,600       |
| Block RAM tiles | 0               | 91.5 (20.56 %)    | 445           |
| DSP slices      | 0               | 50 (5.95 %)       | 840           |

The accelerator occupies under six per cent of the fabric and — notably — uses no DSP slices and no block RAM at all. Both the multiply-accumulate cells and the array’s internal buffering are realised in LUT logic and registers. The AXI wrapper is essentially free, accounting for only 242 of the 11,945 LUTs; the array and its DMA machinery account for the remainder.

Measured against this, one invocation of 512 multiply-accumulates occupies roughly 2,750 cycles. For a 64-cell array this is far more than the arithmetic requires, which places the limit on operand delivery rather than on the multipliers: each invocation streams 640 bytes in and 256 bytes out across the AXI interconnect for 512 multiply-accumulates of work. The array is not the bottleneck at any level of this system — neither against its own operand supply, nor against the driver interface discussed in Section 8.

## 4.3 Clocking and timing closure

|                                 |               |               |            |         |
| ------------------------------- | ------------- | ------------- | ---------- | ------- |
| **Clock**                       | **Frequency** | **Endpoints** | **WNS**    | **TNS** |
| clk\_out1\_clk\_wiz\_0 (system) | 50 MHz        | 157,192       | \+0.556 ns | 0.000   |
| clk\_out2\_clk\_wiz\_0          | 200 MHz       | 1,706         | \+1.331 ns | 0.000   |
| clk\_ref\_mmcm\_400 (MIG)       | 400 MHz       | 14            | \+1.767 ns | 0.000   |
| clk\_out1/2\_clk\_wiz\_eth      | 125 MHz       | 3,820         | \+0.593 ns | 0.000   |

Both the VEGA core and the accelerator reside in the 50 MHz system domain. Setup timing is met on every clock with no failing endpoints. One hold violation of −0.659 ns appears on the Ethernet receive clock-domain crossing, within an organiser-supplied block; the Ethernet link operates correctly in practice and no team-authored logic is involved.

## 4.4 The invocation sequence

The handshake was taken from the organiser reference driver test and confirmed against hardware. Per tile: clear start; write the operands; program the address registers; assert start; poll the done bit; read the result; clear start again. The stream identifier registers are zero for all three input streams.

The final optimised kernel exploits the fact that the address registers are full 32-bit pointers. Every operand tile for a matrix multiply is uploaded to DDR once in 256-word bursts, after which the inner loop consists of five four-byte register writes and nothing else — clear start, set the weight tile address, set the activation tile address, assert start, read done. This removed all bulk data movement from the inner loop and delivered a 4.37× speed-up over the initial implementation, with bit-identical results.

The trailing clear of the start register is retained deliberately. Experiment showed that three chained tiles compute correctly without it, but only because each was separated by roughly 270 µs of unrelated traffic. In the tight optimised loop the next register write follows within 65 µs, and without the clear the done bit never de-asserts, so the poll would return a stale completion and the previous tile’s result would be read.

# 5 Offline Preparation and the Standalone Boundary

The standalone requirement draws a precise line. The host may convert checkpoints, quantize constant weights, emit static metadata, build reference outputs and transfer files. The host must not compute, for any particular runtime input, the detections, the prompt embedding, the activations, the candidate scores or the final answer.

|                                           |                                              |
| ----------------------------------------- | -------------------------------------------- |
| **Performed offline (input-independent)** | **Performed on the board (input-dependent)** |
| BatchNorm folding into convolutions       | JPEG decode and letterboxing                 |
| INT8 quantization of constant weights     | All activations at every layer               |
| Pre-packing weights into 8×8 tile order   | Activation quantization scales               |
| Flattening YOLOv8n into a 103-op list     | Every detection and its confidence           |
| Embedding the 80 fixed COCO label names   | Tokenising and embedding the task prompt     |
| Generating reference outputs for testing  | All suitability scores and the selection     |

Two points deserve explicit note. First, the eighty label embeddings are precomputed because they depend only on fixed class names, never on the runtime image or prompt; the result is mathematically identical to computing them on the board. Second, images are supplied as ordinary JPEG files and decoded on the board by a baseline JPEG decoder written for this project, so no image preparation whatsoever occurs off-board. A raw PPM path is retained as an alternative input format.

# 6 Hardware Validation Results

Development proceeded in four verifiable milestones, each demonstrated on the board before the next was begun.

|   |                                                     |                               |                              |
| - | --------------------------------------------------- | ----------------------------- | ---------------------------- |
|   | **Milestone**                                       | **Result on hardware**        | **Cost**                     |
| A | Plumbing: one real MiniLM layer-0 query projection  | 3,072 / 3,072 exact           | 2,304 invocations            |
| B | Complete MiniLM task→object scoring over 80 classes | wine glass 0.970027           | 259,104 invocations, 135.7 s |
| C | YOLOv8n detection, all 63 convolutions              | detections match host exactly | 342,160 invocations, 267.5 s |
| D | Integrated pipeline, marked image written           | SELECTED: wine glass 0.97006  | 517,888 invocations, 538.8 s |

## 6.1 Numerical validation

The scoring stage was validated against four independent implementations, which agree to every printed digit:

|                                                                |                                         |
| -------------------------------------------------------------- | --------------------------------------- |
| **Implementation**                                             | **Score for "serve wine" → wine glass** |
| NumPy reference read directly from the INT8 package (host)     | 0.970027                                |
| C reference implementation, software emulation (host, x86)     | 0.970027                                |
| C reference implementation, software emulation (board, RISC-V) | 0.970027                                |
| C on the 8×8 accelerator (board)                               | 0.970027                                |
| FP32 golden from the original checkpoint (host)                | 0.969409                                |

The complete INT8 pipeline — quantized weights and quantized activations across all ten matrix-multiply types — therefore costs 0.0006 on the top score and changes no rankings. Board and host results were bit-identical, so the anticipated last-place rounding differences between C library implementations did not materialise.

Detection was validated by three further independent checks. The flattened 103-operation graph, re-implemented in NumPy from the fused weights, matches PyTorch’s own forward pass to 1.8 × 10⁻⁷ on class scores and 1.8 × 10⁻⁴ pixels on box coordinates, confirming the operation list is a faithful description of the network. INT8 quantization tracks FP32 closely at every resolution. Finally, the C implementation matches the NumPy INT8 pipeline to approximately 0.005 in confidence under identical preprocessing, the residual being single- versus double-precision accumulation.

The baseline JPEG decoder was verified against a reference decoder on sixteen COCO photographs, with a mean absolute difference of 0.03–0.04 and a maximum of 3 on a 0–255 scale — inverse-DCT rounding and nothing more.

# 7 Stage 3B Submission Summary

The preceding sections describe what was built and prove that it runs. This section is written for a different purpose: it answers, one by one, the four items the organisers asked every team to supply with the Stage 3B submission. It repeats a small amount of material from elsewhere in the report so that it can be read on its own, and it points to the section or the file that carries the evidence for each claim.

## 7.1 Location of the project folder

This project lives in three places at once, because three different machines are involved. Understanding which is which makes the rest of this section easier to follow.

The first is the development PC — the lab machine hdg\_vivado-lab2, running CentOS 7, on which Vivado builds the FPGA design and the RISC-V cross-compiler builds the application. Nothing is computed there at demonstration time; it is a build machine.

The second is the Genesys 2 board itself, which holds the FPGA and the VEGA RISC-V processor. This is where the demonstration actually runs. It is reached over Ethernet from the development PC.

The third is the submission folder, which is what the organisers receive. It contains everything needed to read, review and re-run the work.

|                                      |                                     |                                                                                                                |
| ------------------------------------ | ----------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Machine or location**              | **Path**                            | **What is there**                                                                                              |
| **Submission folder (Google Drive)** | **43\_Silicon\_Syndicate\_stage3B** | **This report, the demonstration video, and the complete deliverable package described in Section 13**         |
| Development PC, hdg\_vivado-lab2     | /home/Team43/Desktop/DVCon\_2026    | The Vivado project sources, the TCL build flow, the Application build directory and the RISC-V cross toolchain |
| Genesys 2 board                      | /tmp/team43                         | The deployed executable, the two INT8 model packages, the WordPiece vocabulary and the test images             |

One caution about the board directory. /tmp on this board is a RAM disk — memory pretending to be a filesystem — so everything in it disappears when the board loses power. Restoring it means copying about 30 MB back over the network, which takes roughly six minutes at the measured 94 KB/s link rate. The commands are in QUICK\_START.md. Leaving the board powered on between sessions avoids the problem entirely.

Where to find the important files inside the submission folder:

|                                                    |                                                                                                                                                                                                                                                                                                                                                                                   |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **File**                                           | **What it is and why it matters**                                                                                                                                                                                                                                                                                                                                                 |
| **accelerator/bitstream/DVCon\_2026.bit**          | **The bitstream: the compiled FPGA configuration, 10.9 MB, containing the VEGA processor and our 8x8 accelerator. An FPGA is blank until a bitstream is loaded into it, so this file is what turns the Genesys 2 into the machine this report describes. The identical file is produced on the development PC at /home/Team43/Desktop/DVCon\_2026/GEN\_BIT\_OUT/DVCon\_2026.bit** |
| accelerator/build/run.sh                           | The organiser menu launcher. The normal way to program the board, synthesise, implement or simulate — see below                                                                                                                                                                                                                                                                   |
| accelerator/build/tcl/FPGA\_PROGRAMMING.tcl        | The script the launcher calls to load the bitstream onto the board                                                                                                                                                                                                                                                                                                                |
| accelerator/build/tcl/DVCon\_IMP.tcl               | The script that produced the shipped bitstream and the three Vivado reports in accelerator/reports/                                                                                                                                                                                                                                                                               |
| accelerator/rtl/ACCELERATOR\_IP/Accelerator\_Top.v | The wrapper that attaches our accelerator to the supplied SoC. Section 4 explains it                                                                                                                                                                                                                                                                                              |
| application/binaries/team43\_pipeline              | The RISC-V executable that runs the demonstration on the board                                                                                                                                                                                                                                                                                                                    |

**How to program the board.**

The organiser environment provides a menu launcher, run.sh, and that is the intended way in. Copy the files from accelerator/build/ back into the organiser project tree as described in accelerator/build/REBUILD.md, then run the launcher from the RUN directory:

cd /home/Team43/Desktop/DVCon\_2026/RUN

./run.sh

1\) Simulation

2\) FPGA Implementation -\> 1 synthesis only, 2 full build + bitstream

3\) FPGA Programming \<- loads DVCon\_2026.bit onto the board

4\) Exit

Choose option 3. The launcher is preferable to calling Vivado by hand for a concrete reason: before it shows the menu it rewrites the set MDIR\_PATH line inside all four TCL scripts to match wherever the project tree actually sits on that machine. Those scripts contain an absolute path that is otherwise wrong on any machine but the one they were written on, so using the launcher removes a whole class of "file not found" failures.

If the launcher is not available, option 3 is exactly equivalent to running the script it invokes:

vivado -mode batch -source $MDIR\_PATH/TCL/FPGA\_PROGRAMMING.tcl

with the difference that MDIR\_PATH must then be corrected by hand at the top of that file. In either case one further edit is needed: FPGA\_PROGRAMMING.tcl selects the JTAG cable by serial number, currently the one used in the lab, so that line must match the board in front of you.

Rebuilding the bitstream from RTL, rather than loading the one supplied, is menu option 2 then 2. That path additionally requires the organiser VEGA core netlist AS1061\_SYSTEM\_TOP.edn, which is not in this package because of its size; accelerator/build/REBUILD.md explains what is needed and where each file goes.

## 7.2 Work done and what has been achieved

The work was deliberately broken into four milestones, each one a program that could be built, deployed and run on the physical board before the next was attempted. The reason for working this way is that a failure in a large integrated program is hard to attribute: it could be the accelerator, the driver, the model export, the quantisation, or the C code. By proving one layer at a time against a known-good reference, each failure had only a few possible causes.

All four milestones passed on the physical board — not in simulation — and each is backed by a captured terminal log in the evidence/ directory of this package.

|        |                                                                                                                                                                                  |                                         |                                  |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- | -------------------------------- |
| **\#** | **Milestone and what it proved**                                                                                                                                                 | **Result on hardware**                  | **Evidence file**                |
| **A**  | **Driver plumbing: one real matrix multiply pushed through the accelerator and read back. Proved the register map, the handshake and the data layout were understood correctly** | **PASS, all 3,072 output values exact** | **board\_log\_milestone\_A.txt** |
| B      | The complete MiniLM task-to-object scorer, six encoder layers, running on the board. Proved the language half of the system                                                      | PASS, 135.71 s per query                | board\_log\_milestone\_B.txt     |
| C      | YOLOv8n object detection, all 63 convolution layers executed on the accelerator. Proved the vision half                                                                          | PASS, 267.53 s per image at 128 px      | board\_log\_milestone\_C.txt     |
| D      | The two halves integrated into one executable: image and task prompt in, marked image out                                                                                        | PASS, 538.8 s per query                 | board\_log\_milestone\_D.txt     |

Four further pieces of work sit alongside those milestones and are documented in the sections named:

  - The accelerator was integrated into the supplied SoC. The organiser top level, Top.vhd, already declared a component called Accelerator\_Top but did not provide it; we wrote that module with a matching port list, so no organiser file had to be edited. Synthesis, implementation and bitstream generation then completed with timing met. Section 4 covers this.

  - A baseline JPEG decoder was written from scratch for the board, because the platform has none. Without it the application could only accept raw raster images prepared on a PC, which would have weakened the claim that the board works standalone.

  - The driver interface was characterised by direct measurement rather than assumption. That produced the central performance finding of this report — that the cost of the system call dominates the arithmetic it saves at this tile size. Section 8 gives the numbers.

  - Ranking accuracy was evaluated using the protocol and dataset of the reference paper, so the figures in Section 9 are directly comparable with published results instead of being an internal score of our own devising.

Underpinning all of it: the ranking pipeline is implemented four independent times — in NumPy from the deployed weights, in C emulating the accelerator on a PC, in C emulating it on the board, and in C driving the real accelerator. All four agree to every printed digit. That agreement is what allows any hardware number in this report to be trusted rather than merely reported.

## 7.3 Completion status

**The Stage 3B objective is complete: 100 %.**

It is worth being precise about what that claim covers. The objective set for this stage was a genuinely standalone application on the Genesys 2 platform: one deployed executable together with its model files, accepting an image and a task prompt at runtime, computing every input-dependent quantity on the board itself, and producing the marked result. No part of that is outstanding. The application exists, it runs on the board, it accepts different images and prompts without being rebuilt, and its output has been reproduced against an independent reference implementation.

Four items remain open. None of them is missing scope; each is optimisation or characterisation work that would make a finished system better rather than make an unfinished one work. All four are recorded with their current state in STATUS.md.

  - Latency. A query takes 538.8 s at 128-pixel detection. The cause is understood and quantified in Section 8, and it is the driver interface rather than the arithmetic. The stage rules forbid modifying that driver, which bounds how far this can be improved without organiser changes.

  - Detector resolution sweep. Every end-to-end accuracy figure in this report was measured at 128 pixels. Running the same evaluation at 320 and 640 pixels would separate two effects that are currently entangled — how many objects the detector finds, and how well the ranker chooses among them. Those runs have not been executed.

  - JPEG decode time on the board is unmeasured. The decoder is numerically validated against a reference decoder on sixteen photographs and it is present in the shipped binary, but every timed run to date used raw PPM input, so the decode cost on the VEGA core is not known.

  - A weight-packing slowdown inside the integrated binary: roughly 116 s where the identical code in the standalone scoring binary takes 7.6 s. Memory pressure from holding both model files mapped at once is the suspected cause. It is not proven and not fixed.

## 7.4 Adherence to the original solution required

The stage set out a specific shape of solution rather than a free brief. The table below restates each requirement in the terms it was given and says where in this package it is demonstrated.

|                                                                                              |            |                                                                                                                                                                                 |
| -------------------------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Requirement**                                                                              | **Status** | **Where it is demonstrated**                                                                                                                                                    |
| **The application runs standalone on the board**                                             | **MET**    | **Milestone D board log. The development PC performs no computation while the demonstration runs; it is only a file server**                                                    |
| No host computation of any quantity that depends on the runtime input                        | MET        | Section 5 draws the line explicitly: the host may convert and quantize model weights ahead of time, but no detection, embedding, activation or score is ever computed off-board |
| One executable and one model set handle different image and task pairs without recompilation | MET        | Image path and prompt are command-line arguments; three test images and several prompts were run against a single unmodified binary                                             |
| The compute-heavy operations execute on the FPGA accelerator                                 | MET        | All 63 YOLOv8n convolutions and every MiniLM matrix product, totalling 517,888 accelerator invocations per query                                                                |
| The accelerator is integrated without modifying the organiser environment                    | MET        | Top.vhd is unchanged, SHA-256 a7ad9451...ecde3e. The driver, the TCL scripts and the system packages were not touched                                                           |
| Image plus text prompt in, marked image out                                                  | MET        | result.ppm is written by the board and is byte-identical to the host prediction of the same run                                                                                 |

Three choices in this implementation differ from the reference solution the stage was modelled on. They are deliberate, and they are stated here rather than left for a reviewer to discover.

  - YOLOv8n was used instead of a heavyweight detector such as YOLOv8x. Section 11 gives the reasoning and the measured evidence behind that decision in full.

  - The ranker understands fourteen task prompts and only those, because those are the fourteen it was fine-tuned on. Given any other prompt it will still print a winner, but that answer carries no meaning. The fourteen strings are listed in Section 2.

  - The detector runs at 128 pixels by default. This is a runtime option and not a built-in limit; larger sizes work and cost proportionally more time, as the table in Section 3 shows.

## 7.5 Implementation details and utilisation

Sections 3 to 5 carry the full implementation detail: how the detector and the ranker are structured, how the work is split between the FPGA and the processor, and how the accelerator is built. The figures below are the summary a reviewer usually wants first. They are taken from the Vivado reports shipped in accelerator/reports/, which are the tool output from the run that produced the shipped bitstream.

|                                   |                           |                                                                                                                                                                                                     |
| --------------------------------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Metric**                        | **Value**                 | **What it means**                                                                                                                                                                                   |
| **Accelerator LUTs**              | **11,945 (5.86 %)**       | **Look-up tables, the FPGA’s general-purpose logic elements. Our accelerator occupies under 6 % of the 203,800 available on the XC7K325T**                                                          |
| Accelerator flip-flops            | 12,706 (3.12 %)           | Storage elements, of 407,600 available                                                                                                                                                              |
| DSP48 slices used                 | 0                         | The FPGA’s dedicated hardware multipliers. We use none of them: the INT8 multiply-accumulate is built from ordinary fabric logic, so the dedicated multipliers remain entirely free for future work |
| Block RAM used                    | 0                         | On-chip memory blocks. Also untouched — operands stream in from external DDR through the DMA channels instead of being buffered on chip                                                             |
| Systolic array size               | 8 x 8 processing elements | INT8 inputs, INT32 accumulator. 64 multiply-accumulate units working together on one tile of a matrix multiplication                                                                                |
| System clock                      | 50 MHz, timing met        | Worst negative slack +0.556 ns, meaning the design closes timing with margin to spare                                                                                                               |
| Accelerator invocations per query | 517,888                   | 342,160 for detection and 175,728 for task-aware ranking                                                                                                                                            |

The two zeros in that table are worth dwelling on. Using no DSP slices and no block RAM means the accelerator competes with nothing else on the device for its scarcest resources, and it leaves the obvious headroom for a larger or replicated array — which is exactly the direction Section 12 identifies for future work.

## 7.6 Path forward if a demonstration is required at the conference

The demonstration is ready to give today; nothing further needs to be built for it. What follows is the plan for running it on a conference floor, including the parts that need care.

**What to bring.**

  - The Genesys 2 board, its power supply, a JTAG cable for programming the FPGA and an Ethernet cable for copying files to it.

  - A laptop holding the bitstream, the deployed binary, the two INT8 model packages, the vocabulary file and the test images — about 30 MB in total, all present in this submission package.

  - The pre-marked reference outputs, so an audience can see the expected result while a live run is still in progress.

**The demonstration itself.**

The strongest script is the contrast the reference paper opens with. One test image contains three wine glasses and a dining table; the second contains only a cup. The same binary, given the same prompt "serve wine", selects the wine glass in the first image at 0.97006 and the cup in the second at 0.934. Two runs are enough to show that the system is reasoning about which object suits the task rather than matching a fixed class list, which is the entire point of the work. Both results have already been reproduced on hardware and their logs are in this package.

**Timing, and how to handle it.**

One query at 128 pixels takes 538.8 seconds — a little under nine minutes. The program prints progress lines throughout, so it is visibly working rather than hung, but nine minutes is a long silence in front of an audience. Three ways to handle a conference slot:

  - Start one live run and present the architecture while it computes. Nine minutes is close to the natural length of that explanation, so the run finishes roughly as the explanation ends. This is the recommended shape.

  - Run the first image before the audience arrives and run the second live. The contrast still lands and only nine minutes of floor time is spent.

  - If the slot is shorter than nine minutes, play the recorded demonstration video and offer a live run to anyone who stays.

**Risks and their fallbacks.**

|                                                    |                                                                                                                           |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **Risk**                                           | **Fallback**                                                                                                              |
| **The board is power-cycled and /tmp is cleared**  | **Re-copy about 30 MB, roughly six minutes. Keeping the board powered between sessions avoids this entirely**             |
| The board loses its network address after a reboot | One ifconfig command, given in QUICK\_START.md section 2                                                                  |
| There is no time for a full run                    | The recorded video plus the marked output images                                                                          |
| A visitor asks for a task outside the fourteen     | Answer directly: the ranker is fine-tuned on those fourteen prompts only, and its behaviour on anything else is undefined |

Preparation before the conference is limited to rehearsing the sequence and confirming that the board still programs from the shipped bitstream. No further development is required for the demonstration to be given.

# 8 Performance Analysis

## 8.1 Measured driver primitives

A dedicated micro-benchmark measured each device-access primitive in isolation on hardware. Converting to cycles at the 50 MHz system clock makes the shape of the problem clear.

|                                           |                   |                      |
| ----------------------------------------- | ----------------- | -------------------- |
| **Operation**                             | **Measured time** | **Cycles at 50 MHz** |
| lseek (system-call floor, no device work) | 25.3 µs           | 1,265                |
| Register writes (4 bytes)                 | 65.3 µs           | 3,265                |
| Register read (4 bytes)                   | 62.1 µs           | 3,105                |
| DDR write, 64 bytes                       | 188.8 µs          | 9,440                |
| DDR write, 256 bytes                      | 267.5 µs          | 13,375               |
| DDR read, 256 bytes                       | 274.7 µs          | 13,735               |

The decisive observation is that cost is per call, not per byte: a single-word transfer costs 162 µs while a 256-word transfer costs 562 µs. This measurement directly motivated the final kernel design, in which all operands are uploaded once in large bursts and the inner loop performs only register writes.

## 8.2 Where the time goes

|                          |                 |                 |                 |
| ------------------------ | --------------- | --------------- | --------------- |
| **Phase**                | **Milestone B** | **Milestone C** | **Milestone D** |
| Accelerator inner loop   | 92.3 s          | 135.7 s         | 259.7 s         |
| Operand packing / im2col | 7.6 s           | 77.4 s          | 193.2 s         |
| Operand upload to DDR    | 7.0 s           | 4.1 s           | 11.4 s          |
| VEGA floating-point work | 26.0 s          | 51.3 s          | 52.3 s          |
| **Total per query**      | **135.7 s**     | **267.5 s**     | **538.8 s**     |

## 8.3 The interface cost, quantified

With operands pre-packed identically for both paths, one 8×8×8 tile takes 341–356 µs on the accelerator and 216 µs when the same tile arithmetic is emulated in software on the VEGA core. The fabric is not the limitation: it computes its 512 multiply-accumulates in roughly 55 µs, about four times the core’s rate for the same work. The remaining \~286 µs is five system calls at approximately 65 µs each.

At 50 MHz this is arithmetic rather than mystery. Five system calls are roughly 16,000 cycles of interface overheads against roughly 2,750 cycles of useful computation — the interface costs about five times the compute it saves. Five hundred and twelve multiply-accumulates is simply too little work to amortise a system-call round trip, and the fixed 8×8×8 granularity offers no way to enlarge the unit of work. Modifying the organizer-supplied driver was outside the permitted scope of this stage.

This is reported as a quantified property of the interface rather than a shortcoming of the design, and it points directly at the highest-value hardware improvement available: enlarging the work performed per command.

## 8.4 Memory behavior of the VEGA core

A single strided byte read through a large memory mapping costs approximately 670 cycles — a cache miss straight to DDR with no prefetch benefit. This dominated two early implementations. Rearranging weights into accelerator tile order by walking the destination and gathering from the source cost 154.7 s per query; reversing the loop to walk the source sequentially and scatter into the small tile buffer reduced this to 7.6 s, a 20.3× improvement with byte-identical output, verified by asserting equality of both orderings on four real weight tensors. A second instance of the same pathology, reading weight tiles directly from the memory-mapped package inside the emulation loop, cost 4,502 µs per tile until a single bulk copy per convolution removed it.

The engineering rule derived for this platform: never gather from a memory-mapped file in an inner loop; copy to RAM first and prefer sequential reads with scattered writes over the reverse.

# 9 Accuracy Evaluation

Accuracy was measured with the metric of the reference work: average precision at an intersection-over-union threshold of 0.5 against the preferred-object annotations, computed per task and averaged over the fourteen tasks. The suitability probabilities used were read from the deployed INT8 package, so the evaluation reflects the numbers the board computes. Following the reference protocol, the confidence used for the metric is the detection score multiplied by the suitability probability; on ground-truth boxes, where no detection score exists, the probability is used directly.

|                              |               |                     |                     |                    |                   |
| ---------------------------- | ------------- | ------------------- | ------------------- | ------------------ | ----------------- |
| **Box source**               | **This work** | **Pick best class** | **No visual input** | **Classification** | **Proposed GGNN** |
| Ground-truth boxes           | 0.648         | 0.386               | 0.589               | 0.616              | 0.742             |
| Faster R-CNN detections      | 0.288         | 0.141               | 0.237               | 0.288              | 0.326             |
| YOLOv8n detections at 128 px | 0.063         | —                   | —                   | —                  | 0.332             |

On identical Faster R-CNN detections the ranker reaches 0.288 mAP@0.5 — equal to the reference classification baseline, which uses ResNet-101 visual features, and 88 % of their full Gated Graph Neural Network result. The most instructive comparison is against their own no-visual-input ablation at 0.237, which this work exceeds by 0.05. The likely reason is architectural: the reference ablation encodes the object class as a one-hot identifier, whereas this work encodes it as language. In a sentence-embedding space "wine glass" and "cup" are neighbours in a way that category identifiers 46 and 47 are not, and that neighbourhood is precisely the fall-back behaviour the reference paper describes.

The final row is the honest cost of the edge deployment. Running YOLOv8n at 128 pixels loses a large fraction of the preferred objects before the ranker sees them, and average precision is bounded by recall. Of sixteen COCO photographs tested, only six produced any detection at all at 128 pixels. This figure characterises detector recall at reduced resolution, not the quality of the ranking stage, and a resolution sweep separating the two effects is identified in Section 10 as outstanding work.

## 9.1 Confidence in these figures

The average-precision implementation was validated by reproducing a published baseline. Re-implementing the reference paper’s pick-best-class baseline — rank classes by their preference rate on the training split, then retain per image only the highest-rated class present — yields 0.339 against their published 0.386. This is close enough to establish that the metric implementation is in the correct regime, but not close enough to claim exactness; the residual most likely lies in unspecified details of their support threshold and class ranking. Absolute figures should accordingly be read with a tolerance of about ±0.05. Comparisons on identical detections are firmer, since the same code scores both sides.

A second sensitivity was quantified. Because the ranker assigns an identical score to every instance of a class, average precision depends slightly on how ties are ordered. Across four randomised tie orderings the ground-truth-box result varied between 0.6464 and 0.6493, a spread of 0.003; the conservative mean of 0.648 is quoted throughout rather than the 0.658 obtained with natural annotation ordering.

# 10 Current Limitations

The following limitations are stated plainly; each is a consequence of a deliberate design choice or of a measured platform property.

## 10.1 Latency

A complete query takes 538.8 s at 128-pixel detection. This is dominated by the accelerator interface rather than by computation, as Section 8 quantifies. The system is a functional demonstration of an edge pipeline, not a real-time one.

## 10.2 No instance-level discrimination

The ranker scores a task against a class name, so every detected instance of the same class receives an identical suitability score, and ties are broken by detection confidence. The reference paper’s Figure 3 turns on exactly the cases this cannot address: choosing the handier chair over the table, one bed over another, the real toilet rather than its reflection. Resolving these requires visual features per instance, which the current architecture deliberately excludes in order to keep the label embeddings precomputable and the runtime cost bounded.

## 10.3 Detection recall at reduced resolution

At 128 pixels many preferred objects are not detected at all, which caps end-to-end average precision at 0.063 regardless of ranking quality. Detection quality improves markedly at 192 and 256 pixels, at 2.3× and 4.0× the accelerator cost respectively.

## 10.4 Task coverage

The ranker supports the fourteen COCO-Tasks prompts and no others. This is a property of the fine-tuned checkpoint, not of the deployed system, and the deployment would support a broader ranker without change.

## 10.5 Accelerator utilisation

Because the interface dominates, the accelerator is idle for most of each invocation. Well-organised software on the VEGA core is presently about 1.6× faster per tile than the accelerator path, even though the fabric itself is about 4× faster per multiply-accumulate. The accelerator is used throughout regardless, because demonstrating hardware/software integration is the purpose of this stage.

## 10.6 Known open items

  - The JPEG decoder has been verified on the host but its decode time on the VEGA core has not yet been measured; a PPM input path remains available if the decode proves costly.

  - MiniLM weight packing costs approximately 116 s inside the integrated pipeline against 7.6 s for identical code in the standalone scoring binary. The most plausible cause is memory pressure from holding both model mappings simultaneously, but this has not been proven.

  - A resolution sweep at 320 and 640 pixels, which would separate detector recall from ranking quality in the end-to-end accuracy figure, has not yet been run.

  - The decoder supports baseline JPEG only; progressive JPEG is rejected with a clear diagnostic rather than producing incorrect output.

# 11 Why YOLOv8n Was Selected for This Stage

YOLOv8n was selected to make complete end-to-end inference feasible within this stage and to verify seamless hardware and software integration on the Genesys 2 board. The goal of Stage 3B was a working standalone pipeline in which every matrix multiply of both networks executes on the custom accelerator, and that goal is only demonstrable if a full forward pass completes in a time that permits iteration, debugging and live demonstration.

The quantitative case is decisive. At the measured rate of 356 µs per accelerator invocation, YOLOv8x at its native 640-pixel resolution requires 128.9 G multiply-accumulates, which is 251.8 million invocations, or approximately 25 hours per image. Even assuming a hypothetical zero-overhead interface and counting only fabric time, it would still require approximately 3.8 hours. Reducing YOLOv8x to 128 pixels lowers this to 5.16 G multiply-accumulates and roughly 60 minutes per image, which remains impractical for demonstration, and additionally discards the very capacity that distinguishes the model — its advantage lies in resolving fine detail at full resolution, which a 128-pixel input removes.

|                          |           |                 |                      |                                |
| ------------------------ | --------- | --------------- | -------------------- | ------------------------------ |
| **Model and resolution** | **G MAC** | **Invocations** | **At measured rate** | **At zero interface overhead** |
| YOLOv8x @ 640            | 128.9     | 251.8 M         | ≈ 25 hours           | ≈ 3.8 hours                    |
| YOLOv8x @ 128            | 5.16      | 10.1 M          | ≈ 60 minutes         | ≈ 9 minutes                    |
| YOLOv8n @ 128            | 0.175     | 342,160         | ≈ 122 s              | ≈ 19 s                         |

YOLOv8n at 128 pixels requires 29.6× less computation than YOLOv8x at equal resolution while remaining a genuine COCO detector producing real bounding boxes and class confidences. That made it possible to complete, validate and demonstrate the entire integration path — driver handshake, tiled matrix multiply, im2col lowering, quantization, the operation-list interpreter, detection decoding, non-maximum suppression, the language ranker and image write-back — on real hardware within this stage. The system is correspondingly and consistently described as YOLOv8n at reduced resolution, never as YOLOv8x.

# 12 Future Path Toward YOLOv8x

The tiled accelerator can mathematically support the operations required by larger models such as YOLOv8x. Every convolution in YOLOv8x lowers to the same im2col general matrix multiply already implemented, decomposes into the same 8×8×8 INT8 tile products, and uses the same bias-path accumulation along the contraction dimension. No new arithmetic primitive is required; the operation-list interpreter and INT8 export tooling are model-agnostic by construction. What separates the present system from YOLOv8x inference is throughput and memory capacity, not mathematical capability.

Full YOLOv8x inference has not been demonstrated on this platform and no claim is made that the current system executes it efficiently. A practical next iteration would require the following, in approximate order of expected benefit:

  - Larger or multiple compute arrays. A 32×32 or 64×64 tile, or several 8×8 arrays operating in parallel, would raise the work performed per command by one to two orders of magnitude. The present accelerator consumes under 6 % of the fabric and no DSP slices, so considerable headroom exists on this device.

  - Local buffering. On-chip storage for weight and activation tiles would allow operand reuse across many tile products instead of re-presenting operands for each, and the design currently uses no block RAM at all.

  - Batched tile commands. A command queue or descriptor list allowing many tile operations to be issued per driver transaction would amortise the system-call cost, which Section 8 measures at roughly five times the compute it saves.

  - Improved AXI and DDR transfers. Burst or scatter-gather DMA for operand movement, rather than word-count reads and writes through the character device, would remove the per-call cost that presently dominates.

  - Reduced driver overhead. A memory-mapped control interface, or interrupt-driven completion in place of polled status reads, would eliminate most of the approximately 3,265 cycles per register access.

  - Better memory planning. YOLOv8x at full resolution produces im2col buffers in the hundreds of megabytes against 669 MB of system memory, and several of its weight tensors exceed the one-megabyte DDR scratch window, so tiling and streaming strategies would both need revision.

  - YOLOv8x INT8 export and calibration. Deeper networks are more sensitive to quantization, so per-channel weight scales would need to be complemented by calibrated activation ranges rather than the per-tensor dynamic scales used here.

With those changes, YOLOv8x may provide better detection capability than the present configuration — in particular recovering the small and distant objects that a 128-pixel input loses, which Section 9 identifies as the dominant contributor to end-to-end accuracy loss. That is a projection about a future optimised implementation and not a property of the system delivered here.

# 13 Deliverable Package

The submission is organised as follows. FILE\_MANIFEST.md records every file with its size, purpose and origin; SHA256SUMS.txt provides checksums for all files; QUICK\_START.md gives the build, deployment and execution commands; STATUS.md records any item that remains outstanding.

|                             |                                                                                                                     |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Directory**               | **Contents**                                                                                                        |
| application/source          | C sources for the pipeline, detector, scorer, GEMM kernel and driver probe                                          |
| application/headers         | Generated model layout headers, JPEG decoder, bitmap font                                                           |
| application/binaries        | Statically linked RISC-V executables as run on the board                                                            |
| application/model\_packages | INT8 model packages, vocabulary, manifests and goldens                                                              |
| accelerator/rtl             | SoC and accelerator RTL sources, including the Accelerator\_Top.v integration wrapper and INTEGRATION\_NOTES.md     |
| accelerator/build           | Vivado TCL flow, constraints, memory IP and testbench needed to regenerate the bitstream; see REBUILD.md            |
| accelerator/bitstream       | Implemented bitstream                                                                                               |
| accelerator/reports         | Vivado timing and utilization reports                                                                               |
| test\_data                  | Input images, board outputs and expected results                                                                    |
| offline\_export             | Model export, quantization and graph-flattening tooling                                                             |
| scripts                     | Evaluation and format-conversion utilities                                                                          |
| evidence                    | Board terminal logs and evaluation outputs                                                                          |
| (root)                      | This report, QUICK\_START.md and its PDF, STATUS.md, FILE\_MANIFEST.md, SHA256SUMS.txt, and the demonstration video |

Reproduction requires the organiser-supplied buildroot toolchain and the buildroot-2004\_new Docker image, and regenerating the bitstream additionally requires the organiser VEGA core netlist AS1061\_SYSTEM\_TOP.edn. None of the three is included, because of their size; every one of them is an organiser-held file. Their absence is recorded in STATUS.md and in accelerator/build/REBUILD.md.
