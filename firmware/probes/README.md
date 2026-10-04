# firmware/probes/

Each probe is a small, standalone C program that answers **one** hardware question in seconds, instead of running the full multi-minute pipeline. The full write-up — the question, the method, the discipline that keeps a probe honest, and the one that gave the wrong answer — is in [`docs/code_guides/CODE_GUIDE_PROBES.md`](../../docs/code_guides/CODE_GUIDE_PROBES.md).

| Probe | Question | Answer |
|---|---|---|
| `team43_kdepth.c` | Can one accelerator call accumulate the whole K dimension? | **Yes** — exact for every K from 8 to 1024 (method C). This enabled the 3.3× "K-depth collapse". It also showed that without clearing `Y_DONE` results go wrong from K = 512 |
| `team43_regstress.c` | Can several registers be set with one multi-word `pwrite`? | **No** — only the first word is written. Five single-word writes per tile is the minimum |
| `team43_regbatch.c` | The first, inadequate version of the question above | Said "yes" because its addresses were constant — kept as a warning |
| `team43_memcheck.c` | Is this board's memory healthy right now? (no accelerator needed) | Healthy ≥ ~6 MB/s copy; degraded ~1.9 MB/s. Run it first in every session |
| `team43_im2col.c` | Does blocking the im2col write set help? | Hashes the output of every variant (FNV) to prove they are identical |

Build (board):

```bash
riscv64-buildroot-linux-gnu-gcc -std=gnu99 -O2 -static -o team43_kdepth team43_kdepth.c -lrt
```

All except `team43_memcheck` need `/dev/accelerator_test` on the board.
