# Design & Simulation Connectivity Spec (for the block diagrams)

Drawing reference for two figures:
- **Part A** — the design block diagram (inside `top_axi_int`).
- **Part B** — the simulation-setup block diagram (`tb_top_axi_int`).

Parameters used (from `config.svh`): `R=2, C=2, WX=WK=8, WY=32`,
`AXI_WIDTH=32, AXIL_WIDTH=32, ADDR_WIDTH=32`, `AXI_ID_WIDTH=6`, DMA-side
`DMA_ID_WIDTH=4`, `AXIS_USER_WIDTH=8`.

Notation: **`SOURCE.port  ──[bus name, width]──▶  DEST.port`**. A "stream"
means an AXI4-Stream (tdata + tvalid/tready + tlast [+ tkeep/tuser]).

---

# PART A — Design block diagram (`top_axi_int` internals)

## A.0 Top-level interface (the box's outer pins)

`top_axi_int` has exactly two AXI interfaces plus clock/reset:

| Group | Dir | Signals | Width |
|---|---|---|---|
| Clock / reset | in | `clk`, `rstn` | 1, 1 |
| **AXI4-Lite slave** (config) | slave | `s_axil_aw*`, `s_axil_w*`, `s_axil_b*`, `s_axil_ar*`, `s_axil_r*` | addr 32, data 32, strb 4 |
| **AXI4 master** (DDR) | master | `m_axi_aw*`, `m_axi_w*`, `m_axi_b*`, `m_axi_ar*`, `m_axi_r*` | id 6, addr 32, data 32, strb 4 |

## A.1 Instance inventory (8 blocks to draw)

| Instance | Module | Role |
|---|---|---|
| `AXIL2RAM` | `alex_axilite_ram` | AXI-Lite slave → register-file port |
| `CONTROLLER` | `dma_controller` | register map → DMA descriptors + DONE status |
| `mm2s_0` | `alex_axi_dma_rd` | AXI read engine — fetches **K** |
| `mm2s_1` | `alex_axi_dma_rd` | AXI read engine — fetches **X** |
| `mm2s_2` | `alex_axi_dma_rd` | AXI read engine — fetches **bias A** |
| `s2mm` | `alex_axi_dma_wr` | AXI write engine — stores **Y** |
| `ADAPTER_MM2S_K/X/A` | `alex_axis_adapter_any` ×3 | stream width adaptation (32→16, 32→16, 32→64) |
| `ADAPTER_S2MM` | `alex_axis_adapter_any` | stream width adaptation (64→32) |
| `SKID_K/X/A/Y` | `skid_buffer` ×4 | pipeline registers on each stream |
| `SA` | `axis_sa` | systolic array (computes `X·K`) |
| `CROSSBAR` | `axi_crossbar` | merges 4 DMA masters → 1 AXI master |

(The bias-add and the stream-sync are small glue logic in `top_axi_int`, not
separate instances — draw them as little "+"/"&" bubbles.)

## A.2 Bus / width legend (label your arrows with these)

| Bus | Width | Carries |
|---|---|---|
| AXI4-Lite | a32/d32/s4 | config read/write |
| Reg-file port | addr(idx)/d32/strb4 | `AXIL2RAM`↔`CONTROLLER` register access |
| Descriptor | 64 = addr32+len32 (+tag8/user8) | `CONTROLLER`→DMA engines |
| Status | err4 + valid1 | DMA engines→`CONTROLLER` |
| AXIS @ DMA side | d32 + keep4 + last + user8 | DMA↔adapters |
| K stream | **16** (= C·WK) | into array `sk_data` |
| X stream | **16** (= R·WX) | into array `sx_data` |
| A (bias) stream | **64** (= R·WA) | into bias adder |
| Array result `m_data` | **64** (= R·WY) | array → bias adder |
| Y stream | **64** (= R·WY) | bias adder → S2MM adapter |
| AXI4 (DMA↔crossbar) | id4/a32/d32 | per-engine master |
| AXI4 (crossbar→top) | id6/a32/d32 | the one external master |

## A.3 Connection list — follow the dataflow

### (1) Configuration path — CPU writes registers
```
top.s_axil_*        ──[AXI4-Lite, a32/d32]──▶  AXIL2RAM.s_axil_*
AXIL2RAM.reg_wr_en/addr/data/strb ──[reg-file write]──▶ CONTROLLER.reg_wr_*
AXIL2RAM.reg_rd_en/addr           ──[reg-file read ]──▶ CONTROLLER.reg_rd_*
CONTROLLER.reg_rd_data            ──[d32]──▶            AXIL2RAM.reg_rd_data
```
Note: top converts the byte address to a register index:
`reg_*_addr_ctrl = (s_axil_*addr − 0xB000_0000) >> 2`. Draw this as a small
"−BASE,>>2" block between `AXIL2RAM` and `CONTROLLER`.

### (2) Descriptor path — controller launches engines
```
CONTROLLER.mm2s_0_desc/user/valid  ──[desc 64 +user8]──▶ mm2s_0  (read desc in)
CONTROLLER.mm2s_1_desc/user/valid  ──[desc 64 +user8]──▶ mm2s_1
CONTROLLER.mm2s_2_desc/user/valid  ──[desc 64 +user8]──▶ mm2s_2
CONTROLLER.s2mm_desc/tag/valid     ──[desc 64 +tag8 ]──▶ s2mm   (write desc in)

mm2s_0.status_error/valid  ──[status]──▶ CONTROLLER.mm2s_0_status_*
mm2s_1.status_error/valid  ──[status]──▶ CONTROLLER.mm2s_1_status_*
mm2s_2.status_error/valid  ──[status]──▶ CONTROLLER.mm2s_2_status_*
s2mm.status_error/valid    ──[status]──▶ CONTROLLER.s2mm_status_*   (sets DONE)
```

### (3) Operand read path — DDR → array (3 parallel lanes)
```
mm2s_0.m_axis (K)  ──[AXIS d32]──▶ ADAPTER_MM2S_K ──[K stream 16]──▶ SKID_K ──▶ s_k_t*
mm2s_1.m_axis (X)  ──[AXIS d32]──▶ ADAPTER_MM2S_X ──[X stream 16]──▶ SKID_X ──▶ s_x_t*
mm2s_2.m_axis (A)  ──[AXIS d32]──▶ ADAPTER_MM2S_A ──[A stream 64]──▶ SKID_A ──▶ s_a_t*
```
Backpressure detail (optional to draw): each read engine's `arvalid` is ANDed
with its downstream `s_axis_*_tready` before reaching the crossbar
(`..._arvalid_masked`) so it never reads faster than the array can accept.

### (4) Stream synchronization into the array (glue logic)
```
s_valid    = s_k_tvalid & s_x_tvalid          (array steps only when BOTH ready)
s_last     = s_k_tlast  & s_x_tlast
s_k_tready = s_ready & s_x_tvalid
s_x_tready = s_ready & s_k_tvalid
SA.sk_data ◀── s_k_tdata [16]
SA.sx_data ◀── s_x_tdata [16]
SA.s_valid ◀── s_valid ;  SA.s_last ◀── s_last ;  SA.s_ready ──▶ (back to sync)
```

### (5) Compute + bias add + result path
```
SA.m_data [64] ─┐
                ├─(per row r) m_y_tdata[r] = m_data[r] + s_a_tdata[r]   "+" bubble
s_a_tdata [64] ─┘
m_y_tvalid = s_a_tvalid & SA.m_valid
m_y_tlast  = s_last     & SA.m_last
SA.m_ready ◀── m_y_tready & s_a_tvalid
s_a_tready ◀── m_y_tready & SA.m_valid

m_y_t* [Y stream 64] ──▶ SKID_Y ──▶ ADAPTER_S2MM ──[AXIS d32]──▶ s2mm.s_axis (Y)
```

### (6) AXI master aggregation — 4 engines → 1 port
```
mm2s_0.m_axi  ──[AXI4 id4]──▶ CROSSBAR.s_axi[0]   (read only)
mm2s_1.m_axi  ──[AXI4 id4]──▶ CROSSBAR.s_axi[1]   (read only)
mm2s_2.m_axi  ──[AXI4 id4]──▶ CROSSBAR.s_axi[2]   (read only)
s2mm.m_axi    ──[AXI4 id4]──▶ CROSSBAR.s_axi[3]   (write only)
CROSSBAR.m_axi ──[AXI4 id6, a32/d32]──▶ top.m_axi_*   (the external DDR master)
```
The crossbar is configured S_COUNT=4, M_COUNT=1, single region
`0x0000_0000–0xFFFF_FFFF` (everything routes to the one master).

## A.4 Suggested layout for Figure A

```
                         ┌──────────────────────────── top_axi_int ───────────────────────────────┐
   AXI4-Lite  ──────────▶│  AXIL2RAM ──(reg port)──▶ CONTROLLER ──┐ desc/status                    │
   (config, from CPU)    │   (AXI-Lite slave)                     │                                │
                         │                                        ▼                                │
                         │      ┌── mm2s_0(K) ─▶ ADPT ─▶ SKID_K ─┐                                  │
                         │      ├── mm2s_1(X) ─▶ ADPT ─▶ SKID_X ─┤  sync  ┌─────────┐               │
                         │      │                                ├───────▶│  SA     │── m_data ─┐   │
                         │      │   mm2s_2(A) ─▶ ADPT ─▶ SKID_A ─┘        │ (array) │           │   │
                         │      │                                  s_a ───┼─────────┘    "+"────┤   │
                         │      │                                         └──────────▶ SKID_Y ──┤   │
                         │      │                                              ADAPTER_S2MM ◀────┘   │
                         │      │                                                    │ AXIS         │
                         │      │   ┌──────────── CROSSBAR (4S→1M) ──────────┐       ▼              │
                         │      └──▶│ s_axi[0..2]=reads   s_axi[3]=write  ◀── s2mm                  │
   AXI4 master ◀─────────┼─────────│ m_axi ───────────────────────────────────────────────────────┤
   (to DDR)              │         └────────────────────────────────────────┘                      │
                         └───────────────────────────────────────────────────────────────────────┘
```
Three visual "swim-lanes" read left→right: **config** (top), **operand read +
compute** (middle), **AXI aggregation to DDR** (bottom).

---

# PART B — Simulation setup block diagram (`tb_top_axi_int`)

## B.1 Blocks (3 boxes)

| Block | What it is | Plays the role of |
|---|---|---|
| `tb_top_axi_int` | the testbench | **CPU** (AXI-Lite master BFM) + scoreboard |
| `DUT` = `top_axi_int` | the accelerator | the accelerator (unchanged) |
| `MEM` = `axi_ram_model` | behavioral AXI4 slave RAM | **DDR memory** |

## B.2 Bus connections (solid arrows)

```
tb (clock gen)        ──[clk, rstn]──▶            DUT  and  MEM     (shared)

tb (AXI-Lite MASTER   ──[AXI4-Lite, a32/d32]──▶   DUT.s_axil_*
   BFM: axil_write/                                (CPU configures + polls DONE)
   axil_read tasks)   ◀──[rdata, bvalid]──────     DUT.s_axil_*

DUT.m_axi_*  (MASTER) ──[AXI4, id6/a32/d32]──▶     MEM.s_axi_*  (SLAVE)
             ◀──[rdata/rlast/bresp]──────────      MEM.s_axi_*
```
So the **two interfaces face opposite ways**: the TB is *master* on AXI-Lite
into the DUT; the DUT is *master* on AXI4 into the memory. Draw the DUT in the
middle with config on its left edge and the DDR master on its right edge.

## B.3 Backdoor paths (dashed arrows — not buses)

```
tb  ┄┄[preload]┄┄▶  MEM.mem[]      (writes X@0x100, K@0x000, bias@0x200 before START)
tb  ◀┄┄[readback]┄  MEM.mem[]      (reads Y@0x300 after DONE for the self-check)
```
These are hierarchical memory pokes/peeks, **not** AXI transactions — draw them
as dashed lines to distinguish from the live buses.

## B.4 Suggested layout for Figure B

```
        ┌──────────────────────── tb_top_axi_int ────────────────────────┐
        │  clk/rstn gen ─────────────────┬───────────────────────────┐   │
        │                                │                           │   │
        │  AXI-Lite MASTER BFM           │ (clk,rstn)        (clk,rstn)   │
        │  + scoreboard (x,k,a,          │                           │   │
        │   y_ref, y_word)               ▼                           ▼   │
        │        │ AXI4-Lite      ┌────────────┐  AXI4 master ┌──────────┐│
        │        └───────────────▶│   DUT =    │─────────────▶│  MEM =   ││
        │        ◀────────────────│ top_axi_int│◀─────────────│axi_ram_  ││
        │           (resp/DONE)   └────────────┘  (rdata/resp)│  model   ││
        │              ┊ preload/readback (backdoor) ┄┄┄┄┄┄┄┄▶└──────────┘│
        └─────────────────────────────────────────────────────────────────┘
```

## B.5 Mapping to the organizers' Stage-3 picture (for the caption)

| Stage-3 SoC element | Modeled in Stage-2B by |
|---|---|
| Vega AS1061 CPU | `tb` AXI-Lite master BFM (`axil_write`/`axil_read`) |
| AXI interconnect | direct point-to-point (single master/slave each side) |
| DDR3 memory | `axi_ram_model` (`MEM`) |
| Accelerator | `top_axi_int` (`DUT`) — identical RTL to the real block |

---

## Quick checklist before you draw

- Figure A: 8 instance boxes + 2 glue bubbles (`&` sync, `+` bias), 6 labeled
  bus groups, the two outer AXI interfaces.
- Figure B: 3 boxes, 2 live AXI buses (opposite directions), 1 shared clock/
  reset, 2 dashed backdoor lines.
- Always label arrows with **bus name + width** from the legend (A.2 / B.2).
