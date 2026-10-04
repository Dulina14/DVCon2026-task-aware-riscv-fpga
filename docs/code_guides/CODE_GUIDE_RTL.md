# The accelerator RTL — a reading guide

**Team 43 — Silicon Syndicate** · companion to `README_TEAM43_PROJECT.md`

This document explains the hardware — the ~1,200 lines of SystemVerilog that
became 11,945 LUTs on the Kintex-7. Its three companions cover the rest:
`CODE_GUIDE_FIRMWARE.md` (the firmware in depth, with `firmware_study/README.md`
as the shorter reading guide), `CODE_GUIDE_PROBES.md` (the six hardware probes)
and `CODE_GUIDE_TOOLCHAIN.md` (the host export and evaluation scripts).

> **Provenance note (added for the public repository).** The systolic-array core
> (`rtl/sa/*`) and the DMA/AXI wrapper (`rtl/sys/top_axi_int.v`,
> `dma_controller.sv`, `skid_buffer.sv`) are **adapted from Abarajithan G's
> open-source [`axis-systolic-array`](https://github.com/abarajithan11/axis-systolic-array)
> (MIT licence)**. Team 43 configured and integrated it (8×8, INT8/INT32, 64-bit
> AXI, 12-bit IDs, register base `0x2004_0000`, `s2mm_tag` tie-off,
> Vivado-compatible ID assignments) and wrote `Accelerator_Top.v`, the firmware,
> the drivers and every verification artefact. `rtl/ext/*` is Alex Forencich's
> `verilog-axi`/`verilog-axis` (MIT). See `THIRD_PARTY_NOTICES.md`.

No source file is modified by this document. Every line reference is to the
files under `DVCon_SoC_SRC/ACCELERATOR_IP/` (in this repository: `hardware/accelerator_ip/`).

```
Accelerator_Top.v          the wrapper Top.vhd declared but did not supply
rtl/sys/config.svh         R, C, widths — every other size is derived from these
rtl/sys/top_axi_int.v      AXI integration, DMA instances, bias adders
rtl/sys/dma_controller.sv  register file and descriptor generation
rtl/sys/skid_buffer.sv     2-deep valid/ready decoupler
rtl/sa/axis_sa.sv          the 8x8 array and its control
rtl/sa/pe.sv               one processing element
rtl/sa/mac.sv              one multiply-accumulate cell
rtl/sa/tri_buffer.sv       the operand skew
rtl/sa/n_delay.sv          an N-stage shift register with taps
rtl/ext/                   third-party AXI infrastructure (Alex Forencich)
```

---

## 1. Read it in this order

The array is easiest to understand from the inside out. Four small files carry
the whole idea; everything else is plumbing.

1. **`mac.sv`** (27 lines) — one multiply-accumulate.
2. **`pe.sv`** (42 lines) — a MAC plus the registers that pass operands on.
3. **`tri_buffer.sv`** (37 lines) — the skew that makes the wavefront work.
4. **`axis_sa.sv`** (114 lines) — the array, and the control that drains it.

Then `top_axi_int.v` for how operands arrive and results leave, and
`dma_controller.sv` for how software commands it.

---

## 2. `config.svh` — the only numbers you change

```systemverilog
`define R    8     // PE rows
`define C    8     // PE columns
`define WK   8     // weight width  (INT8)
`define WX   8     // activation width (INT8)
`define WY   32    // accumulator and result width (INT32)
`define AXI_WIDTH   64
`define AXIL_WIDTH  32
`define ADDR_WIDTH  64
```

**Every internal bus width is derived from these**, which is why the design
scales without hand-editing:

| bus | formula | value | one word carries |
|---|---|---|---|
| `K_BUS_W` | C × WK | 64 b | 8 weight bytes, one per column |
| `X_BUS_W` | R × WX | 64 b | 8 activation bytes, one per row |
| `A_BUS_W` | R × WY | 256 b | 8 INT32 biases |
| `Y_BUS_W` | R × WY | 256 b | 8 INT32 results |

`K_BUS_W` and `X_BUS_W` both land on **exactly the 64-bit AXI width**. That is
not a coincidence — it is the reason one AXI beat is one array step, why the K
and X stream adapters optimise away to nothing in synthesis, and why the
software packs tiles as `kk*8 + rr`. Change `R`, `C` or the operand widths and
that alignment is lost unless `AXI_WIDTH` moves with them.

Two entries are dead and should not be trusted: `DIR` points at an old
simulation path, and `AXIL_BASE_ADDR` is **unused** — the real base address is a
parameter in `top_axi_int.v`. The `A_*` register-offset defines use a malformed
literal syntax (`h'0` rather than `'h0`) and are likewise not the live map; §7
of the project README has the authoritative offsets.

---

## 3. `mac.sv` — the multiply-accumulate cell

Twenty-seven lines, and the whole arithmetic contract of the accelerator.

```systemverilog
always_ff @(posedge clk)
  if (!rstn)    mul_result <= '0;
  else if (en)  mul_result <= $signed(x) * $signed(k);
```

**`$signed` on both operands is the entire INT8 signedness story.** Drop either
cast and the multiplier goes unsigned; the design still builds, still passes a
smoke test on small positive data, and produces silently wrong results the
moment a weight is negative. The Stage 3A test vectors were deliberately built
to span negative values for this reason — an unsigned misread changes 56 of 64
outputs.

The accumulator:

```systemverilog
else if (en && m_valid)
  a_result <= WY'($signed(m)) + $signed(m_first ? WY'(0) : a_result);
```

`m_first` is the accumulate-or-restart control. When high, the cell starts a
fresh sum rather than adding to what it holds — that is how one PE serves
successive output tiles without an explicit clear. The widening cast
`WY'($signed(m))` sign-extends the 16-bit product into the 32-bit accumulator;
without `$signed` inside the cast the extension would be zero-fill and negative
partial sums would corrupt.

`LM` and `LA` are the multiplier and accumulator pipeline latencies, realised by
two `n_delay` instances. They are parameters rather than constants so the array
can be retimed for a faster clock without touching the control logic — the
diagonal delay in `axis_sa` is computed from them.

---

## 4. `pe.sv` — one processing element

A `mac` plus three registers. The PE's job is to compute **and** to pass its
operands to its neighbours:

```systemverilog
else if (en_mac) begin
  ko <= ki;    // weight continues downward,  row r -> r+1
  xo <= xi;    // activation continues right, col c -> c+1
end
```

This is what makes it systolic. No PE fetches anything; each one receives from
its neighbour, uses the value, and hands it on one cycle later. A value read
once from DDR is consumed by an entire row or column as it travels, which is why
operand bandwidth stays tractable.

The third register is the output path:

```systemverilog
if      (r_copy)   ro <= ao;   // capture this PE's own accumulator
else if (en_shift) ro <= ri;   // otherwise shift a neighbour's result through
```

**One register, two jobs.** `r_copy` loads the local result; `en_shift` turns the
same register into one stage of a shift chain carrying results out of the array.
Results leave along the row, PE by PE, through storage that already exists. That
is why there is no separate output network and no output FIFO — and it is the
main reason the array costs ~108 LUTs per cell rather than several hundred.

---

## 5. `tri_buffer.sv` — the skew

The file documents itself with a diagram at the top. A triangular register
array: input row 0 passes through combinationally, row 1 is delayed one cycle,
row 2 by two, and so on.

```systemverilog
assign buffer[0] = x;                        // row 0: no delay
for (r=1; r<N; r++)
  for (c=r; c<N; c++)                        // triangular: shorter each row
    buffer[r][c] <= buffer[r-1][c];
for (c=0; c<N; c++)
  assign y[c] = buffer[c][c];                // read the diagonal
```

**Why a triangle and not a square.** Element *c* only ever needs *c* cycles of
delay, so row *r* only needs columns *r..N-1*. The saving is not marginal: a
square buffer would be N² registers, the triangle is N(N−1)/2 — for N=8 that is
28 registers per operand instead of 64.

**Why the delay is needed at all.** PE(r, c) must see activation row *r* and
weight column *c* on the same cycle. The activation has travelled *c* PEs to get
there and the weight has travelled *r* PEs, so they would arrive skewed. Pre-
skewing the inputs by exactly the opposite amount makes them meet. The result is
the diagonal wavefront: after `D = R + C − 1 = 15` cycles every PE is busy.

Two instances are used, one for each operand — `TRI_X` skews by row, `TRI_K` by
column.

---

## 6. `n_delay.sv` — the delay primitive

An N-stage shift register with **all intermediate stages exposed** on port `d`:

```systemverilog
assign d = data[N-1:0];   // every tap
assign o = data[N];       // the final output
```

The tap bus is the point. `axis_sa` needs to know the state of a control signal
at *every* diagonal position simultaneously, not just at the end — so `valid`
and `vlast` are delayed once and read at 15 different depths. The `N == 0`
generate branch degenerates to a wire, which lets `LM`/`LA` be set to 1 without
a special case.

---

## 7. `axis_sa.sv` — the array and its control

The datapath is nine lines; the other hundred are flow control.

### 7.1 The array

```systemverilog
for (r=0; r<R; r++) begin:PER
  for (c=0; c<C; c++) begin:PEC
    wire xi = c == 0 ? xi_delayed[r] : xo[r][c-1];   // from the west
    wire ki = r == 0 ? ki_delayed[c] : ko[r-1][c];   // from the north
    wire ri = c == 0 ? WY'(0)        : ro[r][c-1];   // result chain
```

Edge PEs take skewed inputs, interior PEs take their neighbour's output. The
`PER`/`PEC` generate labels are what the Stage 3A synthesis check counted to
confirm 64 distinct PE instances survived — if the array were ever optimised
away, the utilisation would still look plausible, so the instance count was
asserted explicitly.

### 7.2 Column reversal

```systemverilog
for (c=0; c<C; c++)
  assign sk_reversed[c] = sk_data[C-1-c];
```

Results drain from the last column, so without this the output order would be
reversed. Flipping the weights on the way **in** is free — it is pure wiring —
whereas reordering 256-bit results on the way out would cost a multiplexer
network. A one-line fix in the right place.

### 7.3 Diagonal control

Every control signal is indexed by `` `DIAG(r,c) = r+c ``, because all PEs on one
anti-diagonal are at the same pipeline stage:

```systemverilog
.m_first (m_first[`DIAG(r,c)]),
.m_valid (valid[LM+`DIAG(r,c)]),
.r_copy  (r_copy[`DIAG(r,c)]),
```

So control is 15 signals, not 64 — a 4× saving, and it is why the control logic
is a small fraction of the array's area.

### 7.4 The drain handshake

This is the subtlest part of the design. Each diagonal has a one-deep output
register, and three conditions govern it:

```systemverilog
assign conflict [d] = a_valid[d] &&  r_valid[d];  // result ready, register full
assign can_copy [d] = a_valid[d] && !r_valid[d];  // result ready, register free
assign m_stall  [d] = (d >= C-1) && m_valid && !m_ready;
assign reg_stall[d] = can_copy[d] && m_stall[d];
assign r_copy   [d] = can_copy[d] && !m_stall[d];
```

and then the global brake:

```systemverilog
assign en_mac  = !(|conflict) && !(|reg_stall);
assign s_ready = en_mac;
```

**If any diagonal cannot place its result, the entire array stops.** `en_mac`
gates every MAC, both `tri_buffer`s and both `n_delay`s together, so the
computation freezes coherently rather than losing a beat — and because
`s_ready` is the same signal, back-pressure propagates out to the DMA
automatically. There is no FIFO anywhere in the array; correctness under stall
comes from stopping everything at once.

### 7.5 The two streams are locked

```systemverilog
assign s_valid = s_k_tvalid & s_x_tvalid;   // in top_axi_int
assign s_last  = s_k_tlast  & s_x_tlast;
```

The array steps only when **both** operands have a beat. If one DMA starves, the
other waits — no partial products, no re-synchronisation logic.

**The `s_last` line is the most consequential signal in the design.** It ends an
accumulation, and the DMA asserts `tlast` at the end of a *descriptor*, whose
length is just the BYTES register. That means **how much of K accumulates in one
invocation is a software choice, not a hardware constant** — the discovery that
cut a 320 px query from 713 s to 218 s (§9.1 of the project README). The RTL
always permitted it; the firmware simply had not asked.

---

## 8. `skid_buffer.sv` — the decoupler

A three-state machine (`EMPTY`/`PARTIAL`/`FULL`) over exactly **two** storage
locations, `m_data` and `buffer`.

**Not a FIFO.** The purpose is to register both the data path and the `ready`
path so `valid` never has to be driven combinationally from `ready` — that
combinational loop across a module boundary is what destroys timing closure in
AXI-Stream designs. Two slots is the minimum that absorbs one cycle of
back-pressure without dropping a beat, and depth beyond two buys nothing here
because the array stalls as a unit.

Cost, at the four instantiation widths:

| instance | width | breakdown | flip-flops |
|---|---|---|---|
| `SKID_K` | 73 b | 64 data + 8 tuser + 1 tlast | 146 |
| `SKID_X` | 73 b | 64 data + 8 tuser + 1 tlast | 146 |
| `SKID_A` | 265 b | 256 data + 8 tuser + 1 tlast | 530 |
| `SKID_Y` | 257 b | 256 data + 1 tlast | 514 |

≈ **1,336 flip-flops** total — about 10 % of the accelerator's registers spent
on making the interfaces timing-clean.

---

## 9. `top_axi_int.v` — integration

Three things happen here that are not in any other file.

**The bias is added outside the array.** Eight parallel 32-bit adders:

```verilog
m_y_tdata[r] = SA[r] + A[r];
m_y_tvalid   = s_a_tvalid & m_valid;
```

The PEs never see the bias. This is why `Y = X·K + bias` is a single pass, and —
because B is a genuinely independent read stream with its own descriptor — why
the bias-window optimisation in §9.2 of the project README was legal: B and Y had
never needed to share an address, they had merely been pointed at the same one.

**The stream adapters.** K and X are 64→64, identity, and vanish in synthesis.
A is 64→256, gathering four beats into one bias word; Y is 256→64, splitting one
result word into four. Only the two that change width cost anything.

**Burst length.** `AXI_MAX_BURST_LEN = 1` at line 93, so
`AXI_MAX_BURST_SIZE = 1 << 3 = 8 bytes` and `m_axi_arlen = 0`. **Every AXI
transaction moves exactly one 64-bit beat.** A 64-byte K tile is therefore eight
separate transactions, each with its own address handshake. Per tile: 24 AXI
transactions for 512 MACs. This single constant is the largest remaining
streaming-side inefficiency and is item 2 on the future-work list.

---

## 10. `dma_controller.sv` — the command interface

The register file software writes through `/dev/accelerator_test`, and the
descriptor generator that turns those registers into DMA transfers. Four
channels: three read (MM2S_0 for K, MM2S_1 for X, MM2S_2 for bias) and one write
(S2MM for Y).

The interface contract that matters to firmware authors:

- `pread`/`pwrite` counts are in **32-bit words, not bytes**
- the address and byte-count registers are **constant for an entire run** — the
  firmware hoists them out of the tile loop and touches only START per tile
- `Y_DONE` **must be cleared before every START**. It used to be only read, so
  after the first tile the poll was really one syscall of delay — correct by
  accident, and that accident fails from K=512 upward once invocations get long
- one `pwrite` spanning several register words sets **at most the first**; five
  single-word writes per tile is the floor (proved by `team43_regstress.c`)

One fix was made here during Stage 3A: `s2mm_tag` was left undriven and appeared
as X in simulation. It is now explicitly tied off, with a comment saying why —
the descriptor tag is unused by this design, but an undriven output is a latent
hazard rather than a harmless one.

---

## 11. `rtl/ext/` — the third-party infrastructure

The AXI crossbar, AXI-Lite interface, DMA engines and stream adapters are from
Alex Forencich's `verilog-axi` / `verilog-axis` libraries, unmodified. They are
listed here so a reader can tell instantly which files are third-party. `rtl/sa/`
and `rtl/sys/` are adapted from Abarajithan G's MIT-licensed `axis-systolic-array`
(see the provenance note at the top); `Accelerator_Top.v` is team-authored.

Two observations from synthesis worth recording:

- The crossbar is configured `M_BASE_ADDR = 0`, `M_ADDR_WIDTH = 64` — four
  slaves onto one master with **no address decode**. Every slave interface
  reports the full address range in the simulation log, which looks alarming and
  is correct for this topology.
- `priority_encoder.sv` emits two *zero replication count* warnings at line 52.
  These are the degenerate base case of its recursive generate and are benign.

---

## 12. `Accelerator_Top.v` — the wrapper

The organisers' `Top.vhd` already declared a component named `Accelerator_Top`
but did not supply the module. We wrote it with a matching port list, so **no
organiser file had to be edited** — which was also the cleanest possible answer
to the Stage 3A instruction not to modify their sources.

Inside it: an `axi_axil_adapter` bridging the 64-bit AXI4 slave port down to the
32-bit AXI-Lite the register file expects, and `top_axi_int` holding everything
else. `awqos`/`arqos` are tied to zero.

---

## 13. Things that look like cruft and are not

1. **`$signed` on both multiplier operands.** Remove either and the design still
   builds and still passes small positive tests. Negative weights break silently.
2. **The column reversal in `axis_sa`.** Looks like a wiring quirk; it is what
   makes results emerge in C=0-first order without an output multiplexer.
3. **`en_mac` gating everything together.** Looks over-broad. It is what makes
   stalling correct without a FIFO anywhere in the array.
4. **The triangular — not square — `tri_buffer`.** Saves 36 of 64 registers per
   operand at N=8, and the shape is load-bearing, not cosmetic.
5. **One register doing both `r_copy` and `en_shift` in `pe.sv`.** The output
   network is free because it reuses storage that had to exist anyway.
6. **`AXI_MAX_BURST_LEN = 1`.** Not a default anyone chose deliberately; it is
   the reason a 64-byte tile costs eight AXI transactions, and the single
   highest-leverage RTL constant left to change.

---

## 14. Verification

The array was verified against the full SoC in simulation before it ever met
silicon — self-checking firmware with UART phase markers, an independently
written golden model, and a VCD analyser that reconstructed every AXI
transaction and checked 22 protocol properties. §10.6 of the project README has
the detail. The practical result: **hardware bring-up required no RTL change.**

---

*Companion documents: `README_TEAM43_PROJECT.md` (the system),
`firmware_study/README.md` (the firmware).*
