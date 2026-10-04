# DVCon India 2026 — Stage 3A: Processor ↔ Accelerator Integration

## 1. Purpose
Prove, in the supplied VEGA AS1061 SoC simulation, that the VEGA processor boots
from ROM, configures the custom accelerator over its AXI4 slave, the accelerator
reads inputs from **internal SRAM** over its AXI4 master, computes **Y = X·K + bias**,
writes the result back to SRAM, signals completion, and reports progress over UART.
**Internal SRAM only — no DDR** (DDR is unavailable in Stage 3A; see §"DDR note").

## 2. Accelerator parameters (verified from `ACCELERATOR_IP/rtl/sys/config.svh`)
| Param | Value |
|---|---|
| R × C (systolic array) | 2 × 2 |
| WX, WK (operand widths) | 8, 8 (signed INT8) |
| WY (accumulator/output) | 32 (signed) |
| AXI_WIDTH (master data) | 64 |
| AXIL_WIDTH (config data) | 32 |
| ADDR_WIDTH | 64 |
| AXI_ID_WIDTH (master) | 12 |
Keep the 2×2 configuration for Stage 3A (smallest valid config). Do **not** change
to 64×64 — the folder name `dummy_firmware_accelarator_64_64_matrix_20000` refers
to the organizer's *example*, not this RTL.

## 3. Data formats (verified from `axis_sa.sv`, `mac.sv`, `top_axi_int.v`)
- X, K elements: signed 8-bit. MAC output / bias / Y: signed 32-bit.
- Compute: `Y = X·K + bias`.
- Streams: MM2S_0 = **K**, MM2S_1 = **X**, MM2S_2 = **bias**, S2MM = **Y**.
- Observed output stream order (from the passing VCD): **29, 73, 42, 90**.
- 64-bit result write beats: `0x00000049_0000001D` (29,73) then `0x0000005A_0000002A` (42,90).

## 4. SRAM memory map (Stage 3A test)
| Buffer | Address | Bytes |
|---|---|---|
| K matrix | 0x00020000 | 4 |
| X matrix | 0x00020100 | 4 |
| bias | 0x00020200 | 16 |
| Y result | 0x00020300 | 16 |
| signature | 0x00020400 | 4 |
All below `0x00037000` (linker data/BSS/stack start — `mbl.lds:21`, `.bss` at
`0x37020`), so no collision with firmware runtime memory.

## 5. Accelerator register map (verified from `dma_controller.sv`, base 0x20040000)
| Off | Reg | | Off | Reg |
|---|---|---|---|---|
| 0x00 | START | | 0x24 | MM2S_2_DONE |
| 0x04 | MM2S_0_DONE | | 0x28 | MM2S_2_ADDR (bias) |
| 0x08 | MM2S_0_ADDR (K) | | 0x2C | MM2S_2_BYTES |
| 0x0C | MM2S_0_BYTES | | 0x30 | MM2S_2_TUSER |
| 0x10 | MM2S_0_TUSER | | 0x34 | S2MM_DONE (bit0) |
| 0x14 | MM2S_1_DONE | | 0x38 | S2MM_ADDR (Y) |
| 0x18 | MM2S_1_ADDR (X) | | 0x3C | S2MM_BYTES |
| 0x1C | MM2S_1_BYTES | | | |
| 0x20 | MM2S_1_TUSER | | | |
Base decode: `top_axi_int.v:423` does `(reg_wr_addr − 0x20040000) >> 2`, i.e. the
SoC presents **full** addresses and the accelerator subtracts the base. Confirmed
by the passing waveform (`s_axi_awaddr = 0x20040008…`). TUSER is written 0 (unused
for this 2×2 test). Addresses are 32-bit values zero-extended onto the 64-bit master.

## 6. Deterministic test vector
```
X=[[1,2],[3,4]]  K=[[5,6],[7,8]]  bias=[[10,20],[30,40]]
X·K = [[19,22],[43,50]] ;  Y = [[29,42],[73,90]]
Packed inputs: X word 0x04020301, K word 0x08070605,
               bias words @B_ADDR: 10,30,20,40
Signature: 0xC0DE0001 = pass, 0xBADD0000 = fail
```

## 7. Firmware (`Application/stage3a_test/stage3a_main.c`)
Single `main()`. Uses only `init_uart(0x1b)` + `tx_uart()` from the supplied
`uart.c` (no `rx_uart`, no long prints). Flow:
`B` (boot) → write inputs → clear sig → flush cache → program DMA regs → `C`
(configured) → START → `S` (started) → poll S2MM_DONE (finite timeout) → verify
Y == 29/73/42/90 → write signature → `P` pass / `F` fail → flush → spin.
**Successful UART sequence: `BCSP`.**

### Build (RISC-V toolchain / Docker `dvcon_2:latest`)
```
cd Application/stage3a_test
export PATH=$PATH:../dummy_firmware_accelarator_64_64_matrix_20000/../toolchain-bare/bin:../../Application/Util
make clean && make
riscv64-unknown-elf-objdump -D build/accelarator_test > build/accelarator_test.dump
riscv64-unknown-elf-objcopy -I elf64-littleriscv -O binary --remove-section .got \
  --remove-section .got.plt --remove-section .sdata --remove-section .sbss \
  --remove-section .bss --remove-section .comment --gap-fill 0x00 \
  build/accelarator_test build/accelarator_test.bin
bin2hex --bit-width 64 build/accelarator_test.bin build/accelarator_test.hex
hex2mif.sh build/accelarator_test.hex        # -> build/accelarator_test.hex.mif (64-bit, USE THIS)
riscv64-unknown-elf-size build/accelarator_test
riscv64-unknown-elf-objdump -h build/accelarator_test
```
Verified: exactly one `main`; `.text.init` @ `0x00010000`; binary 9,264 B < 32 KB ROM;
`.bss` @ `0x00037020`. **Use the 64-bit `*.hex.mif`, NOT `_32.hex.mif`** (ROM is 64-bit).

## 8. Install the ROM MIF (back up first)
```
cd DVCon_SoC_SRC/MEMORY_IP/rom_32KB_axi
copy rom_32KB_axi.mif rom_32KB_axi.mif.stage2_backup     :: preserve previous
copy <build>\accelarator_test.hex.mif rom_32KB_axi.mif   :: install new 64-bit MIF
```
Then **create a fresh Vivado project** (do not reuse a stale elaborated sim).

## 9. Vivado simulation (Windows) — ONE command
```
vivado -source D:/HDL/verilator/DVCon_2026/TCL/DVCon_SIM_win.tcl
```
That is all. It creates the project, elaborates, and automatically sources
`TCL/stage3a_sim_all.tcl`, which (in this exact order) builds the
divider-organized wave window, opens + logs the VCD, runs, then saves both
artifacts:
- `GEN_BIT_OUT/stage3a_accelerator.vcd`
- `GEN_BIT_OUT/stage3a_accelerator.wcfg`

**Why the order matters:** xsim only records a signal from the moment it is added
to the wave window. Adding waves *after* a run leaves those rows blank. This
script adds them at t=0, so every signal is traced for the whole run and **no
`restart` is ever needed**.

Optional — shorten the run (default 1000 us; 300 us still captures `B,C,S,P`):
```
set STAGE3A_RUN_US 300
source D:/HDL/verilator/DVCon_2026/TCL/stage3a_sim_all.tcl
```
Then zoom to **44–56 us** to see config → SRAM read → compute → write-back.

*(The older two-step scripts `dump_vcd_stage3a.tcl` + `make_wcfg_stage3a.tcl`
still exist for manual use, but must be sourced at t=0 in that order.)*

## 10. Synthesis + utilization (Windows)
```
vivado -mode batch -source D:/HDL/verilator/DVCon_2026/TCL/DVCon_SYN_win.tcl
```
Writes `GEN_BIT_OUT/stage3a_utilization_{total,hier,accelerator}.rpt`.
(Do not run implementation/bitstream — outside Stage 3A.)

## 11. UART configuration
`init_uart(0x1b)`; 8N1, ~115200 baud on the supplied 50 MHz UART reference; each
frame ≈ 86.8 µs, so `BCSP` adds ≈ 347 µs — the reason the VCD run is 1 ms.
Testbench loops TX→RX (`test_bench.vhd:295-296` `sin0=>uart_out, sout0=>uart_out`);
the firmware does not depend on loopback. Decode `sout0`/`uart_out` as 8N1
(idle high, start low, 8 data bits LSB-first, stop high).

## 12. Expected results / acceptance
- `s_axi` config writes at `0x20040008…0x2004003C` then `0x20040000` (START).
- `m_axi` reads `0x20000`(K), `0x20100`(X), `0x20200`(bias); writes `0x20300`
  with `0x00000049_0000001D` and `0x0000005A_0000002A`.
- All BRESP/RRESP = OKAY; addresses stay in SRAM; no DDR access; S2MM_DONE = 1.
- Signature `0xC0DE0001`; UART `BCSP`.

## 13. DDR note (Stage 3A)
DDR is **not modeled** in the supplied SoC netlist: in `AS1061_SYSTEM_TOP.v` all 32
`ddr3_*` pins are tied to `const0`, `init_calib_complete = const0` (tri-stated → Z),
and the processor reset is forced with `init_calib_complete = const1`. The two
`ddr3_model` instances in `test_bench.vhd` (lines 212, 234) are unbound black boxes.
These are **acceptable** Stage 3A warnings. Do not use `0x80000000–0xBFFFFFFF`.
See `DDR3_SIM_INVESTIGATION.md` for full evidence.

## 14. Generated file locations
- Firmware src: `Application/stage3a_test/stage3a_main.c` (+ `crt.S,uart.c,stdlib.c,mbl.lds,Makefile`)
- 64-bit ROM MIF: `Application/stage3a_test/build/accelarator_test.hex.mif`
- Installed ROM MIF: `DVCon_SoC_SRC/MEMORY_IP/rom_32KB_axi/rom_32KB_axi.mif`
- VCD: `GEN_BIT_OUT/stage3a_accelerator.vcd`
- WCFG: `GEN_BIT_OUT/stage3a_accelerator.wcfg`
- Utilization: `GEN_BIT_OUT/stage3a_utilization_*.rpt`
- Vivado projects: `VIVADO_PROJECT/DVCon_PJT_SIM_2026_*`, `.../DVCon_PJT_SYN_2026_*`

## 15. Clean rerun procedure
1. `cd Application/stage3a_test && make clean && make` (+ objcopy/bin2hex/hex2mif as §7).
2. Back up + install the 64-bit MIF (§8).
3. `vivado -source TCL/DVCon_SIM_win.tcl` → `source TCL/dump_vcd_stage3a.tcl` → `source TCL/make_wcfg_stage3a.tcl`.
4. `vivado -mode batch -source TCL/DVCon_SYN_win.tcl`.
5. Analyze VCD (transaction table / UART decode).
