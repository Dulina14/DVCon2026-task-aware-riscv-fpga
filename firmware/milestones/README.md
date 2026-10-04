# firmware/milestones/

Programs used to bring the system up on the real board in Stage 3B, one step at a time. Each had to pass **on the board** before the next was attempted. Board logs: [`results/board_logs/milestones/`](../../results/board_logs/milestones/).

| File | Milestone | What it proves | Board result |
|---|---|---|---|
| `hello43.c` | 0 | A static RISC-V binary built in the buildroot container runs on VEGA Linux | `TEAM43_STANDALONE_OK` |
| `team43_probe.c` | probe | Cost of driver calls (lseek, 1-word and multi-word `pread`/`pwrite`) and experiments E1–E3 | register write 65 µs, read 62 µs — cost is per call, not per byte |
| `team43_drv_test_128_rect_gemm.c` | probe | Rectangular GEMMs `Y[M,N] = X[M,K]·W[K,N] + bias[N]` with deterministic synthetic tensors at real layer shapes | self-checking: prints PASS/FAIL per shape against a software reference |
| `team43_minilm_a.c` | **A** | One real MiniLM layer (layer-0 Q projection) on the accelerator; legacy vs "hoisted" kernel | **3,072 / 3,072 exact**, hoist 1.37× faster |
| `team43_gemm2.c` | A+ | "Operands uploaded once, 5 register writes per tile" kernel vs the Milestone-A kernel | **4.37×** faster, bit-identical |
| `team43_minilm_b.c` | **B** | Full 6-layer MiniLM + head, ranks all 80 classes for a task | wine glass **0.970027** (FP32 reference 0.969409) |
| `team43_yolo_c.c` | **C** | YOLOv8n detection from a PPM image, 103-op graph | detections match the PC `--sim` golden |
| `team43_pipeline_milestoneD.c` | **D** | The complete pipeline: image + task → marked image | **PASS**, wine glass 0.97006, 538.8 s at 128 px |

The Stage 3B helpers carry the header *"DVCon India 2026, Stage 3B, Team 43"*. All build like the main program (static, `-lm -lrt`).
