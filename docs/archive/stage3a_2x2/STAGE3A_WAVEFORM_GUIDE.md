# Stage 3A — What the Program Does, and How to Verify It From the Waveform

Companion to `README_STAGE3A.md`. Keep this open beside Vivado.

---

# PART 1 — What the currently running program does

The ROM contains `Application/stage3a_test/stage3a_main.c`. The VEGA processor
boots from ROM at `0x00010000` and runs this program. Its whole job is to prove
the processor can **drive your accelerator through AXI**, and that the
accelerator can **fetch, compute and write back on its own**.

## The maths being computed

```
X = [1  2]      K = [5  6]      bias = [10  20]
    [3  4]          [7  8]             [30  40]

X · K  = [1·5+2·7   1·6+2·8]  = [19  22]
         [3·5+4·7   3·6+4·8]    [43  50]

Y = X·K + bias = [29  42]
                 [73  90]
```

The accelerator streams results out **column by column**, so the order on the
bus is **29, 73, 42, 90** (col 0 rows 0-1, then col 1 rows 0-1).

## Step-by-step: what each line of firmware does on the bus

| # | Firmware action | What you see in the waveform |
|---|---|---|
| 1 | `init_uart(0x1b)` | UART configured, 115200 8N1 |
| 2 | `tx_uart('B')` | `uart_out` sends `0x42` — "processor booted" |
| 3 | Write X, K, bias into SRAM | CPU stores (not visible on accelerator ports) |
| 4 | `flush_cache()` | pushes data out of CPU cache into real SRAM — **essential**, or the accelerator would read stale memory |
| 5 | 11 register writes | `s_axi_awaddr` steps `0x2004_0008 → 0x2004_003C` |
| 6 | `tx_uart('C')` | `0x43` — "registers configured" |
| 7 | `START = 1` | `s_axi_awaddr = 0x2004_0000`, `wdata = 1` |
| 8 | `tx_uart('S')` | `0x53` — "accelerator started" |
| 9 | *(accelerator runs autonomously)* | `m_axi` reads, computes, `m_axi` writes |
| 10 | Poll `S2MM_DONE` | `s_axi_araddr = 0x2004_0034` repeatedly |
| 11 | Verify Y == 29/73/42/90 | CPU reads SRAM |
| 12 | Write signature | `0xC0DE0001` pass / `0xBADD0000` fail |
| 13 | `tx_uart('P')` or `'F'` | `0x50` = **PASS**, `0x46` = fail |
| 14 | `while(1)` | idle forever |

**Successful UART sequence: `B` → `C` → `S` → `P`  ("BCSP")**

## The register writes in step 5 (accelerator base `0x2004_0000`)

| Byte offset | Register | Value written | Meaning |
|---|---|---|---|
| `0x08` | MM2S_0_ADDR | `0x00020000` | fetch **K** from here |
| `0x0C` | MM2S_0_BYTES | `4` | K is 4 bytes |
| `0x10` | MM2S_0_TUSER | `0` | unused |
| `0x18` | MM2S_1_ADDR | `0x00020100` | fetch **X** from here |
| `0x1C` | MM2S_1_BYTES | `4` | X is 4 bytes |
| `0x20` | MM2S_1_TUSER | `0` | unused |
| `0x28` | MM2S_2_ADDR | `0x00020200` | fetch **bias** from here |
| `0x2C` | MM2S_2_BYTES | `16` | bias is 4 × 32-bit |
| `0x30` | MM2S_2_TUSER | `0` | unused |
| `0x38` | S2MM_ADDR | `0x00020300` | write **Y** to here |
| `0x3C` | S2MM_BYTES | `16` | Y is 4 × 32-bit |
| `0x00` | **START** | `1` | **go!** |

## How the input data is packed in memory

| Address | Raw value | Decoded |
|---|---|---|
| `0x20000` (K) | `0x08070605` | bytes 05,06,07,08 = k00=5, k01=6, k10=7, k11=8 |
| `0x20100` (X) | `0x04020301` | bytes 01,03,02,04 = x00=1, x10=3, x01=2, x11=4 |
| `0x20200` (bias) | `10, 30, 20, 40` | column-major: (r0c0, r1c0, r0c1, r1c1) |

Everything is **signed INT8 in, signed INT32 out**, and lives in **internal SRAM**
(`0x20000`–`0x20400`), safely below the firmware's data/stack region at `0x37000`.

---

# PART 2 — How to verify correct functionality from the waveform

## The single rule you need

> **An AXI transfer happens only on the clock edge where a `VALID` and its paired
> `READY` are both 1.**

Find that overlap; the address/data on that edge is the real value. Everything
below is just *which* VALID/READY pair to look at.

## Where to look

All the AXI action is in a **~10 µs window: 44 µs → 56 µs**. The UART markers
trail behind (each character takes ~86.8 µs to shift out), finishing around
289 µs. So: **drag-zoom the waveform from 44 µs to 56 µs** for the transaction,
and zoom out to ~300 µs to see the UART markers.

## The 7 checks — do these in order

### ✅ CHECK 1 — the system came out of reset
- **Signal:** `s_axi_aresetn`
- **Expect:** goes `0 → 1` at **≈ 3.79 µs**
- **If it stays 0:** nothing else can work; clock/reset problem.

### ✅ CHECK 2 — the processor booted (UART `B`)
- **Signal:** `uart_out`
- **Expect:** first activity at **≈ 29.7 µs**, decoding to `0x42` = `'B'`
- **Meaning:** the ROM image loaded and the CPU is executing your firmware.
- **If flat forever:** wrong/missing ROM MIF.

### ✅ CHECK 3 — the CPU programmed the accelerator
- **Signals:** `s_axi_awvalid` + `s_axi_awready` + `s_axi_awaddr` + `s_axi_wdata`
- **Expect:** ~11 write handshakes between **45.7 µs and 50.3 µs**, with
  `s_axi_awaddr` stepping through `0x2004_0008, 0x0C, 0x10, 0x18, 0x1C, 0x20,
  0x28, 0x2C, 0x30, 0x38, 0x3C`
- **Also check:** `s_axi_bresp = 00` (OKAY) after each write.
- **Meaning:** the AXI4→AXI-Lite bridge works and the register decode is correct.

### ✅ CHECK 4 — START was written
- **Signal:** `s_axi_awaddr` = **`0x2004_0000`** with `s_axi_wdata = 1`
- **Expect:** at **≈ 52.9 µs** — the last write, right after the config block.
- **Meaning:** the accelerator has been launched. Everything after this is the
  accelerator acting *on its own*, with no CPU involvement.

### ✅ CHECK 5 — the accelerator fetched its operands (**the key proof**)
- **Signals:** `m_axi_arvalid` + `m_axi_arready` + `m_axi_araddr`
- **Expect:** read handshakes at **≈ 53.07 – 53.31 µs** with addresses
  **`0x00020000` (K), `0x00020100` (X), `0x00020200` (bias)**
- **Also check:** `m_axi_rresp = 00` (OKAY), and `m_axi_rlast` asserts.
- **Meaning:** this is the heart of Stage 3A — the accelerator is a **bus master**
  reading memory autonomously. It also proves the addresses you programmed in
  CHECK 3 were latched correctly.
- **Must NOT see:** any address in `0x8000_0000`–`0xBFFF_FFFF` (DDR is not
  available in Stage 3A).

### ✅ CHECK 6 — the result was written back correctly (**the functional proof**)
- **Signals:** `m_axi_awvalid`/`awready`, `m_axi_awaddr`, `m_axi_wdata`,
  `m_axi_wstrb`, `m_axi_wlast`, `m_axi_bresp`
- **Expect:** at **≈ 53.77 µs**, `m_axi_awaddr = 0x00020300` and two 64-bit beats:

| Beat | `m_axi_wdata` | Decoded |
|---|---|---|
| 1 | `0x00000049_0000001D` | low = `0x1D` = **29**, high = `0x49` = **73** |
| 2 | `0x0000005A_0000002A` | low = `0x2A` = **42**, high = `0x5A` = **90** |

- **Check:** `m_axi_wstrb = 0xFF` (all 8 bytes valid), `m_axi_wlast = 1` on the
  final beat, `m_axi_bresp = 00` (OKAY).
- **Meaning:** `Y = X·K + bias = [29 42; 73 90]` — **the accelerator computed the
  right answer.** This single data word is your strongest evidence.

### ✅ CHECK 7 — completion signalled and firmware confirmed PASS
- **Signals:** `s_axi_araddr` = `0x2004_0034` (the CPU polling `S2MM_DONE`), then
  `uart_out`
- **Expect:** the poll eventually reads bit0 = 1, then the final UART character at
  **≈ 289 µs** decodes to **`0x50` = `'P'`**
- **Meaning:** the accelerator reported done, **and the firmware independently
  read back all four values from SRAM and confirmed they equal 29/73/42/90.**
  `'F'` (`0x46`) would mean it finished but the numbers were wrong.

## Decoding a UART byte by eye (8N1)

`uart_out` idles high. A character is:
`start bit (low)` → `8 data bits, LSB first` → `stop bit (high)`, each bit
≈ **8.68 µs** wide.

To read one: find a high→low edge, sample the middle of each of the next 8 bit
windows, then write the bits **right-to-left**.

| Char | Binary (LSB-first on the wire) | Hex |
|---|---|---|
| `B` | 0 1 0 0 0 0 1 0 → `01000010` | `0x42` |
| `C` | 1 1 0 0 0 0 1 0 → `01000011` | `0x43` |
| `S` | 1 1 0 0 1 0 1 0 → `01010011` | `0x53` |
| `P` | 0 0 0 0 1 0 1 0 → `01010000` | `0x50` |
| `F` | 0 1 1 0 0 0 1 0 → `01000110` | `0x46` |

## Quick verdict

**PASS** if you can point at:
1. `m_axi_araddr` = `0x20000`, `0x20100`, `0x20200`
2. `m_axi_awaddr` = `0x20300` with `wdata` = `0x00000049_0000001D` and `0x0000005A_0000002A`
3. all `RRESP`/`BRESP` = `00`
4. UART = `B C S P`

That is the complete Stage 3A story: **CPU configures → accelerator reads →
computes → writes back → reports done → firmware verifies.**

## If something looks wrong — symptom → likely cause

| Symptom | Likely cause |
|---|---|
| No UART at all, no AXI | ROM MIF missing/stale → firmware not running |
| `B` appears, no config writes | CPU stuck before the accelerator code |
| Config writes but wrong offsets | register base decode wrong |
| Config writes but no `m_axi` reads | START never written, or DMA descriptor not issued |
| Reads happen, addresses wrong | address register value wrong / byte-count wrong |
| Reads OK but wrong `wdata` | data packing or matrix layout wrong |
| `wdata` correct but UART shows `F` | firmware read-back mismatch (ordering assumption) |
| `RRESP`/`BRESP` ≠ `00` | addressing a non-existent slave (e.g. DDR) |
| Signals show `X` after reset | uninitialised logic — real bug |
| Signals show `U`/`Z` on ddr3_/eth_/JTAG/SD | **normal** — unconnected TB signals & unmodelled external devices |

---

## Signals that are *not* your problem

The testbench declares many signals it never connects (`TRST, TCK, TDI, TMS,
TDO, WP, HOLD, gpio_port0, sd_*, serial_rst_n`) — `Top` has no such ports, so
they sit at `'U'` forever. `ddr3_*` and `eth_*` sit at `'Z'` because those
external models aren't present (documented in `DDR3_SIM_INVESTIGATION.md`).
None of these affect Stage 3A.
