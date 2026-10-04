# Stage 3A — Waveform Reading Guide (8×8)

The simulation writes `GEN_BIT_OUT/stage3a_8x8_accelerator.wcfg` next to the VCD.
Load it: the signals are already arranged into eleven labelled groups and the view
is pre-zoomed to the execution window. Every time below is from the verified run.

---

## Part 1 — What the program does

### The maths

```
X[r][k] = ((3r + 2k) % 11) - 5     signed int8      8×8
K[k][c] = ((5k + 3c) %  9) - 4     signed int8      8×8
B[r][c] = 2r - c                   signed int32     8×8
Y       = X · K + B                signed int32     8×8
```

All operands live in internal SRAM (`0x20000`–`0x20400`), safely below the
firmware's data/stack region at `0x37000`. DDR3 is not used.

### Step by step, and what each step looks like on the bus

| # | Firmware action | What you see |
|---|---|---|
| 1 | `init_uart(0x1b)` | UART configured, 115200 8N1 |
| 2 | `tx_uart('B')` | `uart_out` sends `0x42` — processor booted |
| 3 | Write X, K, bias into SRAM | CPU stores — not visible on the accelerator ports |
| 4 | `flush_cache()` | pushes data out of the CPU cache into real SRAM — **essential**, or the accelerator would read stale memory |
| 5 | 11 register writes | `s_axi_awaddr` steps `0x2004_0008 → 0x2004_003C` |
| 6 | `tx_uart('C')` | `0x43` — registers configured |
| 7 | `START = 1` | `s_axi_awaddr = 0x2004_0000`, `wdata = 1` |
| 8 | `tx_uart('S')` | `0x53` — accelerator started |
| 9 | *(accelerator runs autonomously)* | `m_axi` reads → compute → `m_axi` writes |
| 10 | Poll `S2MM_DONE` | `s_axi_araddr = 0x2004_0034` |
| 11 | Recompute all 64 outputs in software and compare | CPU reads SRAM |
| 12 | Write signature | `0xC0DE0008` pass / `0xBADD0008` fail, at `0x20400` |
| 13 | `tx_uart('P')` or `'F'` | `0x50` = **PASS**, `0x46` = fail |
| 14 | `while(1)` | idle forever |

### The register writes in step 5 (base `0x2004_0000`)

| Offset | Register | Value | Meaning |
|---|---|---|---|
| `0x08` | MM2S_0_ADDR | `0x00020000` | fetch **K** from here |
| `0x0C` | MM2S_0_BYTES | `0x40` | K is **64** bytes |
| `0x10` | MM2S_0_TUSER | `0` | unused |
| `0x18` | MM2S_1_ADDR | `0x00020100` | fetch **X** from here |
| `0x1C` | MM2S_1_BYTES | `0x40` | X is **64** bytes |
| `0x20` | MM2S_1_TUSER | `0` | unused |
| `0x28` | MM2S_2_ADDR | `0x00020200` | fetch **bias** from here |
| `0x2C` | MM2S_2_BYTES | `0x100` | bias is **256** bytes (64 × int32) |
| `0x30` | MM2S_2_TUSER | `0` | unused |
| `0x38` | S2MM_ADDR | `0x00020300` | write **Y** to here |
| `0x3C` | S2MM_BYTES | `0x100` | Y is **256** bytes (64 × int32) |
| `0x00` | **START** | `1` | **go** |

### How the data is packed

| Data | Layout | Address formula | Bytes | 64-bit beats |
|---|---|---|---|---|
| K | **row-major** | `0x20000 + k*8 + c` | 64 | 8 |
| X | **column-major** | `0x20100 + k*8 + r` | 64 | 8 |
| Bias | **column-major** | `0x20200 + c*32 + r*4` | 256 | 32 |
| Y | **column-major** | `0x20300 + c*32 + r*4` | 256 | 32 |

Beat *k* of K is row *k*; beat *k* of X is column *k*; beat *c* of bias/Y is
column *c*.

---

## Part 2 — Verifying it from the waveform

### The single rule

> **An AXI transfer happens only on the clock edge where a `VALID` and its paired
> `READY` are both 1.**

Find that overlap; the address/data on that edge is the real value. Everything
below is just *which* VALID/READY pair to look at. Sampling anywhere else — for
example reading `WDATA` at the address handshake — gives a stale value.

### Where to look

| Window | What is there |
|---|---|
| **460 – 500 µs** | all the AXI activity — configuration, operand fetch, compute, write-back. This is what the supplied `.wcfg` pre-zooms to. |
| **0 – 1200 µs** | the four UART markers. Each character takes ~86 µs to shift out, so they are spread across the whole run. |

### The eight checks, in order

**1 — the system came out of reset**
`s_axi_aresetn` goes `0 → 1` at **≈ 3.79 µs**. If it stays low, nothing else can
work; it is a clock or reset problem.

**2 — the processor booted (UART `B`)**
`uart_out` first activity at **≈ 29.87 µs**, decoding to `0x42` = `'B'`.
Flat forever means a wrong or missing ROM MIF.

**3 — the CPU programmed the accelerator**
`s_axi_awvalid` + `s_axi_awready` + `s_axi_awaddr` + `s_axi_wdata`.
Expect **11 write handshakes between 466.45 µs and 471.07 µs**, with `s_axi_awaddr`
stepping through `0x2004_0008, 0x0C, 0x10, 0x18, 0x1C, 0x20, 0x28, 0x2C, 0x30,
0x38, 0x3C`. Check `s_axi_bresp = 00` (OKAY) after each.
This proves the AXI4→AXI-Lite bridge and the register decode work.

**4 — START was written**
`s_axi_awaddr = 0x2004_0000` with `s_axi_wdata = 1`, at **≈ 473.55 µs** — the last
write. Everything after this is the accelerator acting on its own.

**5 — the accelerator fetched its operands** *(the key proof)*
`m_axi_arvalid` + `m_axi_arready` + `m_axi_araddr`, with data on
`m_axi_rvalid` + `m_axi_rready`.

| Stream | Base address | Beats | Bytes |
|---|---|---|---|
| K | `0x00020000` | 8 | 64 |
| X | `0x00020100` | 8 | 64 |
| bias | `0x00020200` | 32 | 256 |

Check `m_axi_rresp = 00` (OKAY) and that `m_axi_rlast` asserts.
This is the heart of Stage 3A: the accelerator is a **bus master** reading memory
autonomously, which also proves the addresses programmed in check 3 were latched.
**You must NOT see** any address in `0x8000_0000`–`0xBFFF_FFFF` — DDR is not used.

**6 — the array actually computed** *(look inside)*
The VCD logs the internal streams, so you can watch the datapath rather than infer it:

| Group | Signals | Beats |
|---|---|---|
| K into array | `s_k_tvalid` / `s_k_tready` / `s_k_tdata` (64 b) | 8 |
| X into array | `s_x_tvalid` / `s_x_tready` / `s_x_tdata` (64 b) | 8 |
| bias into array | `s_a_tvalid` / `s_a_tready` / `s_a_tdata` (**256 b**) | 8 |
| result out of array | `m_y_tvalid` / `m_y_tready` / `m_y_tdata` (**256 b**) | 8 |
| result to memory | `m_axis_s2mm_t*` (64 b) | 32 |

The bias bus is 256 bits because the adapter upsizes 64→256 to pre-load a whole
column of accumulators per beat. The 8 → 32 expansion on the last row is the
4:1 downsize back to the 64-bit memory bus.

**7 — the result was written back correctly** *(the functional proof)*
`m_axi_awvalid`/`awready`, `m_axi_awaddr`, `m_axi_wdata`, `m_axi_wstrb`, `m_axi_wlast`.
Expect **32 write beats**, the first with `m_axi_awaddr = 0x00020300` at
**≈ 476.59 µs**, its data one cycle later at **≈ 476.61 µs**.

Y is column-major, so the first 64-bit beat at `0x20300` carries `Y[0][0]` in the
low word and `Y[1][0]` in the high word. Measured from the submitted VCD:

| Beat | Address | `m_axi_wdata` | Decoded |
|---|---|---|---|
| 1 | `0x00020300` | `0x00000012_0000001B` | low `0x1B` = **27** = Y[0][0], high `0x12` = **18** = Y[1][0] |

Both agree with the golden matrix, so this single 64-bit word is the most compact
piece of functional evidence in the whole waveform.

Check `m_axi_wstrb = 0xFF` (all 8 bytes valid), `m_axi_wlast` on the final beat,
`m_axi_bresp = 00`. The full expected matrix is in §5.2 of the report and in
`GEN_BIT_OUT/stage3a_8x8_vcd_analysis.md`.

**8 — completion signalled and the firmware confirmed PASS**
`cfg_S2MM_DONE` asserts at **≈ 479.23 µs**; the CPU's polling read of
`s_axi_araddr = 0x2004_0034` observes it. The final UART character at
**≈ 1109.87 µs** decodes to `0x50` = `'P'`.
That `'P'` means the firmware independently re-read all 64 values from SRAM and
recomputed all 512 multiply-accumulate terms in software. `'F'` (`0x46`) would
mean the accelerator finished but the numbers were wrong.

### Reference timeline

| Event | Time |
|---|---|
| reset release | 3.79 µs |
| UART `B` | 29.87 µs |
| config writes | 466.45 – 471.07 µs |
| START | 473.55 µs |
| `cfg_S2MM_DONE` | 479.23 µs |
| UART `C` | 487.79 µs |
| UART `S` | 574.19 µs |
| UART `P` | 1109.87 µs |

### Decoding a UART byte by eye (8N1)

`uart_out` idles high. A character is
`start bit (low)` → `8 data bits, LSB first` → `stop bit (high)`,
each bit **8.64 µs** wide (measured). So `'B'` = `0x42` = `0100_0010` appears on
the wire, LSB first, as `0 1 0 0 0 0 1 0`.

### Doing it automatically instead

All of the above is what `Application/stage3a_test/analyze_vcd_8x8.py` performs
mechanically, plus 22 AXI protocol properties. It runs as part of the simulation
and exits non-zero on any failure. Its full decode is written to
`GEN_BIT_OUT/stage3a_8x8_vcd_analysis.md`.
