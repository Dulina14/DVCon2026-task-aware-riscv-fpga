# Stage-2B Waveform Analysis Guide

Tailored to the **actual** Vivado run in
`axis-systolic-array-new.sim/sim_1/behav/xsim` and the signals saved in your
`tb_top_axi_int_behav.wcfg`. Every value below is what you can confirm directly
in your wave window.

Confirmed timeline (from `simulate.log`):

```
[145 ns]  CPU: programming accelerator over AXI-Lite
[585 ns]  CPU: writing START
[625 ns]  CPU: polling DONE
[1105 ns] S2MM_DONE asserted after 12 polls
          Y_dut = [29 42 ; 73 90] == Y_ref   -> SELF-CHECK PASS
[1405 ns] $finish  (tb_top_axi_int.sv line 350)
```

Clock = 10 ns (so 1 cycle = 10 ns), `rstn` active-low, time resolution 1 ps.

---

## 0. How to read any transaction in this design

Every bus here is **AXI**, and AXI moves data on exactly one event: the rising
clock edge where a **`VALID` and its paired `READY` are both 1**. Find that
overlap and you've found the transaction. Everything below is just *which*
valid/ready pair to look at, and *what value* rides with it.

Your window already contains the three things you need:

| What | Signals in your wave | Tells you |
|---|---|---|
| CPU → DUT config | `s_axil_*` | the processor programming registers |
| DUT → DDR | `m_axi_*` | the accelerator reading/writing memory by itself |
| Compute | `DUT/s_valid`, `DUT/s_ready`, `DUT/m_valid`, `DUT/m_last` | the systolic array working |
| Scoreboard | `x`, `k`, `a`, `y_ref`, `y_word`, `rd`, `poll` | TB inputs / results |

Tip for navigation: in the xsim GUI type a time in the cursor box (e.g.
`585 ns`) or use the yellow marker; the value of every signal at the cursor is
shown in the "Value" column. Drop a second marker and xsim shows the Δt — handy
for the latency numbers in §9.

---

## 1. Reset & preload (0 – 145 ns)

`rstn` is low for the first 100 ns, then high. All `*valid` are 0.

The input matrices are loaded into the DDR model by **backdoor poke**
(`MEM.mem[...] = ...` in the TB), *not* over the bus — so you see no AXI
activity, but if you add `MEM/mem` to the wave you'll see K@0x000, X@0x100 and
bias@0x200 already present. You can also watch the TB arrays `x`, `k`, `a`
(already in your window) hold the test values, and `y_ref` = `[29 42 ; 73 90]`
computed before anything runs.

---

## 2. Phase 1 — AXI-Lite configuration (145 – 585 ns)

The CPU model writes 11 registers through the **AXI-Lite slave**. Put the cursor
in this window and watch `s_axil_awaddr`, `s_axil_wdata`, and the handshakes.

**One write = ~4 cycles**, looks like this:

```
clk            __/‾\__/‾\__/‾\__/‾\__
s_axil_awvalid ___|‾‾‾‾‾|___          CPU drives address...
s_axil_awready _____|‾‾‾|___          ...slave accepts (AW handshake)
s_axil_wvalid  ___|‾‾‾‾‾|___
s_axil_wready  _____|‾‾‾|___          (W handshake)
s_axil_awaddr  ==<B0000008>=
s_axil_wdata   ==<00000100>=
s_axil_bvalid  ________|‾|_           OKAY response  (s_axil_bresp = 00)
```

11 writes × ~40 ns = 440 ns ⇒ spans 145→585 ns, matching the log.

**Decode:** register index = `(s_axil_awaddr − 0xB000_0000) >> 2`. The exact
sequence you'll see on `s_axil_awaddr` / `s_axil_wdata`:

| # | awaddr | idx | Register | wdata | Meaning |
|---|---|---|---|---|---|
| 1 | B0000008 | 2 | MM2S_0_ADDR | 00000000 | K source addr |
| 2 | B000000C | 3 | MM2S_0_BYTES | 00000004 | K length = 4 B |
| 3 | B0000010 | 4 | MM2S_0_TUSER | 00000000 | (unused) |
| 4 | B0000018 | 6 | MM2S_1_ADDR | 00000100 | X source addr |
| 5 | B000001C | 7 | MM2S_1_BYTES | 00000004 | X length = 4 B |
| 6 | B0000020 | 8 | MM2S_1_TUSER | 00000000 | — |
| 7 | B0000028 | 10 | MM2S_2_ADDR | 00000200 | bias source addr |
| 8 | B000002C | 11 | MM2S_2_BYTES | 00000010 | bias length = 16 B |
| 9 | B0000030 | 12 | MM2S_2_TUSER | 00000000 | — |
| 10 | B0000038 | 14 | S2MM_ADDR | 00000300 | Y dest addr |
| 11 | B000003C | 15 | S2MM_BYTES | 00000010 | Y length = 16 B |

`s_axil_wstrb` = `F` (all 4 bytes valid) on every write. To watch the registers
fill, add `DUT/CONTROLLER/cfg` — each `cfg[idx]` latches as its write lands.

---

## 3. Phase 2 — START (585 ns)

A single AXI-Lite write to `s_axil_awaddr = B0000000` (index 0),
`s_axil_wdata = 00000001`. Inside `dma_controller` this launches all four DMA
descriptors on one cycle, then **auto-clears** — so `DUT/CONTROLLER/cfg[0]` is a
1-cycle pulse. Add it to your wave; it's the cleanest "compute started" marker.

From here the CPU does nothing but poll — the accelerator runs on its own.

---

## 4. Phase 3 — AXI master READS (operand fetch, ≈ 625 ns on)

Now `m_axi_*` comes alive: the DUT is the **bus master** pulling operands from
DDR. The design uses `AXI_MAX_BURST_LEN = 1`, so every transfer is a
**single-beat burst**. In your wave the burst-qualifier signals are therefore
constant and worth a glance to prove it:

```
m_axi_arlen   = 00   (1 beat)
m_axi_arsize  = 2    (2^2 = 4 bytes/beat)
m_axi_arburst = 1    (INCR)
m_axi_rresp   = 0    (OKAY)
```

**One read beat:**

```
m_axi_arvalid __|‾|____      DUT asks for an address
m_axi_arready __|‾|____      DDR accepts        (AR handshake)
m_axi_araddr  =<00000100>=
m_axi_rvalid  ____|‾|__      DDR returns data
m_axi_rready  ____|‾|__      DUT accepts        (R handshake)
m_axi_rdata   ==<04020301>=
m_axi_rlast   ____|‾|__      =1 (single beat)
```

**The 6 reads you will see on `m_axi_araddr`** (all funneled through the
crossbar onto the one master port):

| Source | Region | Bytes | beats | addresses | rdata |
|---|---|---|---|---|---|
| MM2S_0 (K) | 0x000 | 4 | 1 | `0x000` | `07080506` |
| MM2S_1 (X) | 0x100 | 4 | 1 | `0x100` | `04020301` |
| MM2S_2 (bias) | 0x200 | 16 | 4 | `0x200,0x204,0x208,0x20C` | `0A,1E,14,28` |

The **interleave order** of these on the bus is decided by the `axi_crossbar`
arbiter, so it's timing-dependent — don't expect strict address order. To tell
*which engine* owns a beat, read `m_axi_arid` (each DMA has a distinct ID;
the crossbar widens 4-bit source IDs to the 6-bit master ID).

The 32-bit words don't enter the array raw — the `alex_axis_adapter_any` blocks
reshape them: K and X words are **down-sized 32→16** into 2 stream beats each
(that "2" is the contraction depth T), and bias is **up-sized 32→64**.

---

## 5. Phase 4 — systolic-array compute (overlaps the reads)

These four are already in your window — this is the heart of the demo:

| Signal | What to look for |
|---|---|
| `DUT/s_valid` & `DUT/s_ready` | 2 beats high = the T=2 operand pairs entering the array (`s_valid = s_k_tvalid & s_x_tvalid` — both operands must be present) |
| `DUT/m_valid` & `DUT/m_ready` | result beats leaving the array (C=2 columns) |
| `DUT/m_last` | high on the final column |

The array accumulates `Y[r][c] = Σ_t x[r][t]·k[t][c]` over the 2 operand beats,
then streams the 2×2 result out over 2 beats. Immediately after the array,
`top_axi_int` adds the bias (`m_y = m_data + a`), so what flows to the write DMA
is the final `Y = X·K + bias`. (Add `DUT/SA/en_mac` if you want to see the MAC
enable; expand `DUT/SA` to see the `tri_buffer` diagonal staggering.)

---

## 6. Phase 5 — AXI master WRITES (result write-back, → 1105 ns)

The S2MM engine drives the `m_axi` **write** channels. 16 bytes ÷ 4 =
**4 single-beat writes**. Qualifiers: `m_axi_awlen=00`, `m_axi_awsize=2`,
`m_axi_awburst=1`, `m_axi_wstrb=F`, `m_axi_wlast=1` each beat, `m_axi_bresp=0`.

```
m_axi_awvalid _|‾|___       dest address
m_axi_awready _|‾|___       (AW handshake)
m_axi_awaddr  <00000300>
m_axi_wvalid  __|‾|__       data
m_axi_wready  __|‾|__       (W handshake)
m_axi_wdata   <0000001D>    0x1D = 29
m_axi_wlast   __|‾|__
m_axi_bvalid  ____|‾|_      OKAY
```

The four `m_axi_wdata` values, in address order:

| awaddr | wdata | dec | meaning |
|---|---|---|---|
| 0x300 | 0000001D | 29 | Y[0][0] |
| 0x304 | 00000049 | 73 | Y[1][0] |
| 0x308 | 0000002A | 42 | Y[0][1] |
| 0x30C | 0000005A | 90 | Y[1][1] |

(64-bit output beat per column = `{row1,row0}`, downsized to two 32-bit words:
low word = row0, high word = row1.)

---

## 7. Phase 6 — DONE polling (625 – 1105 ns)

Concurrently, the CPU model polls `S2MM_DONE` (register 13) with AXI-Lite
**reads** at `s_axil_araddr = B0000034`:

```
s_axil_arvalid _|‾|__
s_axil_arready _|‾|__
s_axil_araddr  <B0000034>
s_axil_rvalid  __|‾|_
s_axil_rdata   <0000000X>    0 while busy, 1 when done
```

You'll count **12 read bursts**. Watch the TB variable `rd` (in your window):
it reads `0` eleven times, then `1` at **1105 ns**. The `poll` counter ticks up
to 12. That `0→1` on `s_axil_rdata`/`rd` is the definitive "done" event — it
happens because S2MM reported `status_valid` with no error and the controller
set `cfg[A_S2MM_DONE]=1`.

---

## 8. Phase 7 — read-back & self-check (1105 – 1405 ns)

The TB backdoor-reads Y from the DDR model into `y_word` (in your window — watch
it change from `DEADBEEF` to `29,73,42,90`) and prints the PASS table. No bus
activity. `$finish` at 1405 ns.

---

## 9. Numbers to quote in your write-up

- **Config latency:** ~440 ns (11 AXI-Lite writes) — one-time CPU setup cost.
- **Compute + DMA latency:** START→DONE = 585→1105 ns = **520 ns = 52 clk** for
  the full 2×2×2 GEMM-with-bias *including* all DDR traffic.
- **Bus traffic:** 6 read beats + 4 write beats = 10 single-beat bursts, all
  arbitrated onto one shared AXI4 master by `axi_crossbar`.
- **Functional proof:** `Y_dut = [29 42 ; 73 90]` exactly equals the software
  reference — the complete chain (AXI-Lite config → AXI-master read → systolic
  array → bias add → AXI-master write → DONE) is verified end-to-end.

> Honest framing for the report: at 2×2 the handshake/DMA overhead dominates and
> the array's throughput edge isn't visible — the purpose of this bench is to
> prove **correct, autonomous AXI integration and the full data flow**, which it
> does. Larger R, C and contraction depth are where speed-up appears.

---

## 10. About the elaboration warnings (so you can answer questions on them)

`elaborate.log` shows 6 `VRFC 10-3091` warnings like *"actual bit length 32
differs from formal bit length 16 for port s_axi_awqos"*. These come from the
`axi_crossbar` instantiation where unused optional sideband ports (`awqos`,
`awuser`, `wuser`, `arqos`, `aruser`, `ruser`) are tied to the literal `0`,
which Verilog treats as 32-bit. The crossbar truncates to the real width.
**They are harmless** — no functional impact — but you can silence them by
sizing the constants (e.g. `4'd0`, `16'd0`) if you want a clean log.
```
