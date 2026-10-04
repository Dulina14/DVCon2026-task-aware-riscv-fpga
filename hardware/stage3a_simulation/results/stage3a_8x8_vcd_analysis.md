# Stage 3A 8x8 - VCD Transaction Analysis

**VCD:** `D:/HDL/verilator/DVCon_2026_SUBMISSION/Env/GEN_BIT_OUT/stage3a_8x8_accelerator.vcd`
**SHA-256:** `c76ef580749dc0482c82f6c490ce2743be2f95b1a9595684e54d747b9b52264f`

Golden reference computed inside this script from the mathematical
definition - never from firmware or RTL output.

## Overall result

**ACCEPTED** - 0 acceptance check(s) failed.

## Timeline

| Event | Time (us) |
|---|---|
| reset release | 3.7876 |
| UART `B` (0x42, stop=1) | 29.8676 |
| UART `C` (0x43, stop=1) | 487.7876 |
| UART `S` (0x53, stop=1) | 574.1876 |
| UART `P` (0x50, stop=1) | 1109.8676 |
| cfg_S2MM_DONE=1 | 479.2275 |

Decoded UART: `BCSP`  (bit period 8.640 us)  -> required `BCSP`: YES

## Configuration writes (posedge-sampled)

| # | time (us) | awaddr | off | wdata |
|---|---|---|---|---|
| 1 | 466.4475 | 0x0000000020040008 | 0x08 | 0x00020000 |
| 2 | 466.9275 | 0x000000002004000C | 0x0C | 0x00000040 |
| 3 | 467.3675 | 0x0000000020040010 | 0x10 | 0x00000000 |
| 4 | 467.8475 | 0x0000000020040018 | 0x18 | 0x00020100 |
| 5 | 468.3075 | 0x000000002004001C | 0x1C | 0x00000040 |
| 6 | 468.7475 | 0x0000000020040020 | 0x20 | 0x00000000 |
| 7 | 469.2275 | 0x0000000020040028 | 0x28 | 0x00020200 |
| 8 | 469.6875 | 0x000000002004002C | 0x2C | 0x00000100 |
| 9 | 470.1275 | 0x0000000020040030 | 0x30 | 0x00000000 |
| 10 | 470.6075 | 0x0000000020040038 | 0x38 | 0x00020300 |
| 11 | 471.0675 | 0x000000002004003C | 0x3C | 0x00000100 |
| 12 | 473.5475 | 0x0000000020040000 | 0x00 | 0x00000001 |

Processor DONE reads (s_axi AR): 2

## Master reads (AR queued, R paired by ID)

* K    beats=8 (exp 8) bytes=64 unique=8
* X    beats=8 (exp 8) bytes=64 unique=8
* Bias beats=32 (exp 32) bytes=256 unique=32

Read-data vs golden operands: ALL MATCH


## Master writes (AW queued, W paired, addr from AW)

AW=32  W=32  B=32  unique write addrs=32

## Internal streams

| stream | beats | expected |
|---|---|---|
| K (s_k_t*, 64-bit) | 8 | 8 |
| X (s_x_t*, 64-bit) | 8 | 8 |
| Bias (s_a_t*, 256-bit after 64->256 upsize) | 8 | 8 |
| systolic input (s_valid&s_ready) | 8 | 8 |
| raw 256-bit (m_valid&m_ready) | 8 | 8 |
| final 256-bit (m_y_t*) | 8 | 8 |
| S2MM 64-bit (m_axis_s2mm_t*) | 32 | 32 |
* raw m_data: MATCH
* m_y_tdata: MATCH
* m_axis_s2mm_tdata: MATCH

## Final results (64/64 decoded)

All 64 decoded outputs are present and match the golden matrix.

| r\c | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| **0** | 27 | -28 | 16 | 24 | -31 | 13 | 21 | -34 |
| **1** | 18 | -13 | -17 | 15 | -16 | -20 | 12 | -19 |
| **2** | -2 | 24 | 5 | -5 | 21 | 2 | -8 | 18 |
| **3** | -22 | -5 | -6 | -25 | -8 | -9 | -28 | -11 |
| **4** | 2 | 10 | 27 | -1 | 7 | 24 | -4 | 4 |
| **5** | 26 | -8 | -6 | 23 | -11 | -9 | 20 | -14 |
| **6** | 39 | 29 | -17 | 36 | 26 | -20 | 33 | 23 |
| **7** | -3 | -22 | 49 | -6 | -25 | 46 | -9 | -28 |

## Protocol checks

| check | result |
|---|---|
| RRESP OKAY | OK |
| BRESP OKAY | OK |
| WSTRB 0xFF | OK |
| WLAST | OK |
| RLAST | OK |
| ARLEN=0 | OK |
| AWLEN=0 | OK |
| ARSIZE=3 | OK |
| AWSIZE=3 | OK |
| AR burst INCR | OK |
| unpaired R | OK |
| unpaired W | OK |
| S2MM TKEEP | OK |
| S2MM TLAST | OK |
| DDR accesses | OK |
| outside buffers | OK |
| duplicate write addrs | OK |
| unexpected read addrs | OK |
| unexpected write addrs | OK |
| duplicate read addrs | OK |
| descriptor handshakes | mm2s_0, mm2s_1, mm2s_2, s2mm |
| DMA status errors | OK |
