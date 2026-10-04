# docs/archive/stage3a_2x2/

Our **first** Stage 3A integration (late July 2026), with a **2×2** array, before we moved to the final 8×8 design. Kept for history — the final version is in [`hardware/stage3a_simulation/`](../../../hardware/stage3a_simulation/README.md).

| File | What it is |
|---|---|
| `README_STAGE3A.md` | How the 2×2 design was integrated and simulated; the DDR-tied-off finding (all buffers in SRAM) |
| `STAGE3A_WAVEFORM_GUIDE.md` | Seven hand checks with times (UART `B` at 29.7 µs, START at 52.9 µs, result at 53.77 µs, `P` at ~289 µs) |
| `SUBMISSION_CHECKLIST.md` | Our clean-up audit before submission (missing ROM MIF, a stray second `main()` in the organiser example folder, expected MD5s) |

2×2 results: simulation PASS (Y = [29, 73, 42, 90]); synthesis 5,214 LUTs (2.56 %).
