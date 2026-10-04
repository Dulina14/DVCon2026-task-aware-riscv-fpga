# Code guide — the hardware probes

Team 43 · DVCon India 2026 · companion to `README_TEAM43_PROJECT.md`
(see also `CODE_GUIDE_RTL.md`, `CODE_GUIDE_TOOLCHAIN.md`, `CODE_GUIDE_FIRMWARE.md`)

---

## Why the probes exist at all

Every probe in this directory was written to answer **one question about the real
board** that could not be answered by reading the RTL, by reading the firmware, or
by running the pipeline and looking at the wall clock.

That last point is the important one. A full query takes four to fourteen minutes.
A hypothesis tested by editing the pipeline and re-running it costs a quarter of an
hour per attempt, produces a single number with no isolation, and — if it is wrong
in a way that still produces plausible output — can be believed for weeks. Three of
the findings below were *invisible* in the pipeline's own output.

So each probe is a standalone `main()` that:

1. states its question in the header comment, in prose, before any code;
2. computes a **software reference of its own** and checks correctness *before* it
   reports a single timing number;
3. runs in seconds, not minutes;
4. depends on as little as possible — `team43_memcheck` needs no device node, no
   model files and no network, so it runs the moment Linux boots.

All of them build with the same line:

```sh
riscv64-buildroot-linux-gnu-gcc -std=gnu99 -O2 -static -Wall -Wextra \
    -o team43_<name> team43_<name>.c -lrt
```

`-static` because the board's rootfs is minimal; `-lrt` for `clock_gettime`.
Copies live in `firmware_study/probes/` (reading copies) and `app/` (build tree).

### Reading order

| order | probe | why here |
|---|---|---|
| 1 | `team43_memcheck.c` | no device needed; run it first, always |
| 2 | `team43_kdepth.c` | the largest single optimisation in the project |
| 3 | `team43_regbatch.c` | a probe that gave the **wrong** answer |
| 4 | `team43_regstress.c` | the probe that corrected it, and why |
| 5 | `team43_im2col.c` | the only probe that touches no hardware register |
| 6 | `team43_syscost_opt1c.c` | **not present in this tree — see §7** |

---

## 1. `team43_memcheck.c` — is this board's memory healthy?

**82 lines. No device node, no model files, no network.**

### The question

On 27 August a power cycle and FPGA reprogram left the board running a 256 px query
in **1108 s instead of 275 s** — four times slower — while *every accelerator number
stayed normal*. The array was fine. The syscall path was fine. `team43_kdepth` ran
and reported its usual figures.

The reason the accelerator probes could not see it: `team43_kdepth`'s entire working
set is about 8 KB, so it never leaves cache. The pipeline streams multi-megabyte
tensors. The degradation was in DRAM bandwidth, and only code that misses cache can
observe it.

### The method

Three measurements over a 4 MB working set (`#define N (4L*MB)`, chosen as
"bigger than any plausible cache"):

```c
for (int r = 0; r < 3; r++) memcpy(b, a, N);            /* bulk copy      */
for (long i = 0; i < nf; i++) { ... amax ... }           /* sequential scan */
for (long i = 0; i < N; i += 64) acc += a[i];            /* one byte/line   */
```

The third loop touches one byte per 64-byte line, so it measures line-fill latency
rather than streaming bandwidth — it isolates *why* the first two are slow.

`acc` and `amax` are printed with the comment *"ignore, keeps the loops honest"* —
they exist so `-O2` cannot delete the loops as dead code. That is a real hazard in a
memory benchmark and worth saying out loud.

### The thresholds

Measured on this board, both states:

| | memcpy | sequential scan | 256 px query |
|---|---|---|---|
| healthy (26 Aug) | ≥ 6 MB/s | ≥ 6 M elem/s | ~275 s |
| degraded (27 Aug) | ~1.9 MB/s | ~0.9 M elem/s | ~1108 s |

The verdict logic has three arms, not two:

```c
if (copy_mb >= 6.0 && scan_me >= 6.0)        /* healthy  */
else if (copy_mb < 3.0 || scan_me < 3.0)     /* DEGRADED */
else                                          /* marginal */
```

The middle band exists because a benchmark that only knows "good" and "bad" will
confidently call a 4 MB/s board good. `marginal` says *reprogram and re-check*.

### Why it is step one of every session

It costs five seconds and it runs before anything else is on the board. The
alternative is spending eight minutes transferring 46 MB of model packages to a
board that is going to be too slow to demonstrate.

---

## 2. `team43_kdepth.c` — can one invocation accumulate the whole K dimension?

**314 lines. The single largest optimisation in the project: 713 s → 218 s.**

### The question

Every GEMM was issued as `(M/8) × (N/8) × (K/8)` invocations. On the deployed graph
the mean is **~38 K-tiles per output tile**, so a 320 px query cost **2,339,064
invocations for only 62,196 output tiles**. At ~50 µs per system call and five calls
per invocation that is **587 s of a 713 s query**, and essentially none of it is
arithmetic — the arithmetic underneath is about 0.16 µs.

So: does the `k0` loop have to exist?

### The argument from the RTL, written before any measurement

The header comment lays out the chain, and it is the clearest single paragraph in
the codebase about why hardware and software meet where they do:

- in `axis_sa` the accumulator is reset by `m_first`, which follows `s_last`;
- in `top_axi_int`, `s_last` is `s_k_tlast & s_x_tlast`, which the DMA asserts at
  the **end of a descriptor**;
- in `dma_controller` the descriptor length is simply the `MM2S_n_BYTES` register,
  a full 32-bit field fed straight through.

**Nothing anywhere fixes K at 8.** K per invocation is a software choice wearing the
costume of an RTL constant. Stream `8*K` bytes instead of 64 and the array should
accumulate all of K in one invocation and emit the same 8×8 tile.

The packed buffers already suited it: `gemm_run` puts the X tile for `(m0,k0)` at
`(m0*KT + k0)*64` and the W tile for `(n0,k0)` at `(n0*KT + k0)*64`, so for a fixed
`m0` the whole K run is **already contiguous and already k-major** — exactly the beat
order the array wants. No repacking was needed.

### The hazard the same probe exposed

`wait_done()` polled `Y_DONE`, but nothing ever *cleared* it. After the first tile it
was permanently high, so the "poll" was really one system call's worth of delay —
about 50–100 µs. Comfortably longer than a K=8 tile, which is why the pipeline was
correct. **Correct by accident.** A K=1024 tile streams 1024 beats at 50 MHz ≈
20.5 µs plus DMA latency: still under the accidental margin, but now a guess rather
than an accident, and shrinking as K grows.

That is why the probe measures **three** methods, not two:

| | method | what it tests |
|---|---|---|
| A | `KT` invocations of K=8, seed and harvest Y around them | what shipped at the time |
| B | one invocation of the full K, same loose one-read wait | fast — but racy? |
| C | one invocation of the full K, `Y_DONE` cleared then polled properly | fast and correct |

### The correctness discipline

> *"Correctness is decided against a software reference computed here, never against
> method A, so a wrong A cannot make B or C look right."*

This is the sentence to point at when asked how the project avoided fooling itself.
The probe additionally aborts if method A itself disagrees with the reference, with
the message *"method A is wrong at K=%d, which should not happen"* — a harness that
fails its own control is a broken harness, and nothing it says afterwards counts.

### The result

Swept over K = 8 … 1024 (`KMAX 1024`), `--iters 20` random operand sets per K:

- **C is exact at every K.** One invocation accumulates the whole K dimension.
- **B is exact to K=256 and wrong from K=512** — max error 127,452 at K=512 and
  169,893 at K=1024.

B's failure is the `Y_DONE` bug becoming visible for the first time. It had been
present and harmless for months; only making invocations longer exposed it. The
pipeline now clears `Y_DONE` before every start, which is the `rw(REG_Y_DONE, 0u)`
line in the shipped inner loop (README §9.1).

### Constants

```c
#define K_ADDR      0xBF000000UL
#define X_ADDR      0xBF100000UL
#define Y_ADDR      0xBF300000UL
#define KMAX        1024
#define POLL_LIMIT  2000000u
```

`POLL_LIMIT` matters: a probe that hangs on a board with no console is worse than
one that fails. Every wait in every probe is bounded and dies with a message.

---

## 3. `team43_regbatch.c` — the probe that gave the wrong answer

This one is in the tree **on purpose**, and the README lists it as *"the first,
inadequate version of the above."* Keeping a failed instrument is a verification
result in itself.

### The question

Five system calls per tile — `START=0`, `K_ADDR`, `X_ADDR`, `START=1`, poll — at
~485 µs per invocation and 517,888 invocations per query, is 251 s: the largest
remaining item once the K collapse landed. The register file is a contiguous block
of 32-bit words and the driver takes `pwrite()` counts **in words**, so in principle
one `pwrite` can set a *run* of registers. Nobody had ever tried it: every register
write in every Team 43 program up to that point was a single word.

The header documents the full word map, which is the canonical version:

```
word 0  0x00  START            word  8  0x20  MM2S_1 (X) TUSER
word 1  0x04  MM2S_0 (K) DONE  word  9  0x24  MM2S_2 (B) DONE
word 2  0x08  MM2S_0 (K) ADDR  word 10  0x28  MM2S_2 (B) ADDR
word 3  0x0C  MM2S_0 (K) BYTES word 11  0x2C  MM2S_2 (B) BYTES
word 4  0x10  MM2S_0 (K) TUSER word 12  0x30  MM2S_2 (B) TUSER
word 5  0x14  MM2S_1 (X) DONE  word 13  0x34  S2MM DONE
word 6  0x18  MM2S_1 (X) ADDR  word 14  0x38  S2MM ADDR
word 7  0x1C  MM2S_1 (X) BYTES word 15  0x3C  S2MM BYTES
```

Note the geometry that drives everything below: **K_ADDR is word 2 and X_ADDR is
word 6**, so *any* single write covering both must also pass over **word 5, the X
DONE flag**. That is unavoidable, and it is the prime suspect.

### Why it was wrong

It ran **one tile, 200 times, with the same `K_ADDR` and `X_ADDR` every time**, and
checked a single result. It reported both batched variants **EXACT**.

In the full pipeline the three-call variant then produced complete garbage: **53
detections instead of 4, every confidence pinned at 1.000.**

The probe was not buggy. It was *inadequate*: with constant addresses, a write that
fails to update an address register is indistinguishable from one that succeeds,
because the register already holds the right value. It exercised none of what the
real loop does.

**The lesson, stated plainly:** a probe whose inputs do not vary cannot detect a
mechanism that fails to propagate inputs. Constant test data hides exactly the class
of bug that batched register writes produce.

---

## 4. `team43_regstress.c` — does batching survive a *real* GEMM pattern?

The correction. It mirrors `gemm_run()` rather than approximating it:

```
for each (m0, n0) tile-pair
    seed Y with the bias vector
    for k0 in 0..KT-1
        issue one tile with ADDRESSES THAT CHANGE EVERY TIME
        (the accelerator accumulates into Y in place)
    read Y back
```

so it covers changing addresses, the accumulation chain across `k0`, and the
seed/harvest traffic interleaved between tile groups. **Every accumulator of every
tile-pair is compared** — with the default `--mt 6 --nt 6 --kt 6` that is
6 × 6 × 64 = 2,304 accumulators per method per round, against one value of one tile
in the earlier probe.

Operands are generated so that **every tile is distinct**:

```c
s = s * 1103515245u + 12345u;
Ktile[i][j] = (int8_t)((s >> 16) % 127 - 63);
```

A stale address now shows up immediately. That single change is the entire
difference between the two probes.

### Four methods, chosen to bisect the register file

| | calls | what it merges | which flag it disturbs |
|---|---|---|---|
| A | 5 | nothing | — (known good) |
| B | 4 | words 2..6 in one write | X DONE |
| C | 3 | words 0..6 in one write | both DONEs **and** START |
| E | 4 | words 0..2 in one write, then X_ADDR alone | K DONE only |

These are not four guesses at a speedup. They are a **bisection**: B, C and E each
disturb a different subset, so whichever survives *names the register that is
actually fatal* instead of leaving it a guess. That is the difference between
"batching doesn't work" and a result.

The same control as `kdepth`: A is checked against a software reference first, and
if A is wrong the probe exits with status 2 and the message *"the harness or the
packing convention is wrong, so nothing below can be trusted."*

The software reference also documents the packing convention precisely, which is
useful on its own:

```c
partial[cc*8 + rr] = bias[cc*8 + rr] + Σ_k Σ_kk  X[kk*8 + rr] * W[kk*8 + cc]
```

### The result

**One `pwrite` covering several register words sets at most the first.** Five
single-word writes per tile is the floor. The finding is retained in the pipeline
behind `--unsafe-batched-regs` for timing comparison only, clearly labelled as
producing wrong results (README §10.5).

Note the closing line of the program when nothing fails:

> *"All variants exact. Re-run with --rounds 5 before believing it."*

Given the history above, the probe is explicitly sceptical of its own PASS.

---

## 5. `team43_im2col.c` — does blocking the im2col write set help?

**149 lines. The only probe here that touches no hardware register at all.**

### The question

The pipeline's largest remaining packing cost is one loop: gathering input patches
into accelerator tile order. The loop swept **every row-tile before revisiting any of
them**, so its write working set is `row-tiles × 64` bytes. Once that exceeds the
data cache, each 8 bytes stored costs a line fill.

On a healthy board the pass costs ~0.66 s per call. On a degraded one the same call
was measured at **8.44 s — 557 s of an 816 s query** — while every *sequential* pass
in the program was unaffected. That asymmetry is the signature of a write set that
does not fit.

Blocking the row-tiles bounds the write set. It should remove most of that gap
**without changing a single output byte.**

### The method

One function, parameterised by block size, used for both the old and new behaviour:

```c
for (int mt0 = 0; mt0 < MT; mt0 += blk) { ... }
```

with `blk = 1 << 24` reproducing the original single sweep exactly, so the baseline
and the candidate are literally the same code path — no risk of comparing against a
straw man.

Block sizes swept: `none, 256, 128, 64, 32, 16, 8`, over three shapes chosen to span
the detector:

| shape | rows | why |
|---|---|---|
| `early  C=32  128×128 3×3` | many | wide early layer, huge row count |
| `middle C=128  32×32  3×3` | — | the common middle case |
| `deep   C=256  16×16  3×3` | few | few rows, long K |

`rows` is clamped to 4096 so the probe stays in seconds rather than the fourteen
minutes a full query takes.

### The correctness check

Every variant's output buffer is hashed with FNV-1a and compared to the unblocked
hash:

```c
h == ref ? "identical" : "*** DIFFERENT -- BUG ***"
```

and the closing instruction is unambiguous:

> *"Every row must read 'identical'; the reordering must not change a byte."*

This is the right test for this optimisation. A reordering that is faster and
*almost* identical is not a win, it is a silent accuracy regression — and a hash over
the entire packed buffer catches it in one comparison instead of eyeballing tensors.

The chosen block size is passed to the pipeline as `--im2col-block N`; the build
defaults to 64.

---

## 6. What the probes have in common

Five patterns worth naming, because they are the reason the hardware numbers in the
README can be believed:

1. **The question is in the first line of the file.** Not the method, not the
   result — the question. Every header opens with `probe_name -- <question>?`
2. **Correctness gates timing.** No probe prints a speedup for a method that has not
   first reproduced a reference exactly. `regstress` prints `bad accums` and
   `max error` columns *before* it is willing to be believed.
3. **The reference is independent of the thing under test.** `kdepth` says it
   outright; `regstress` enforces it with an early exit; `im2col` uses a hash of the
   full buffer rather than spot checks.
4. **Every wait is bounded.** `POLL_LIMIT 2000000u`, `die("Y_DONE never asserted")`.
   A probe that hangs teaches nothing.
5. **They are cheap enough to re-run.** Seconds, not minutes. This is what made it
   possible to test three hypotheses in an afternoon instead of one per morning.

And one pattern that is really a warning: **`regbatch` passed.** The discipline above
is not a guarantee. What caught `regbatch` was not a better rule, it was noticing
that the probe's inputs did not vary the way the real loop's do. The four rules make
wrong answers rarer; only comparing against the real pattern makes them visible.

---

## 7. `team43_syscost_opt1c.c` is referenced but not present

**Action needed.** README §9.2 cites `team43_syscost_opt1c` as the probe that
validated the bias-window optimisation on hardware — *"5/5 cases, 64/64 match"* —
§8 cites it for the per-access syscall cost table (*"mean over 200 calls, array
idle"*), and §16 lists it in the probe table. There is also a `run_n_opt1c.sh` in
the scripts list.

A search of `D:\HDL\final_phase` finds no `team43_syscost_opt1c.c` in any of
`.`, `app/`, `app/TEAM43_FROZEN/`, `eval/`, `eval_s/`, `firmware_study/probes/` or
`firmware_study/src/`.

Two acceptable resolutions, in order of preference:

1. **Recover it** from whichever machine it was built on and drop it into
   `firmware_study/probes/` alongside the others. Two numbers currently quoted in
   the README rest on it, and it is the only evidence for the bias-window
   correctness claim.
2. **Mark it explicitly as not shipped** in §9.2, §8 and §16 — for example
   *"validated on hardware by `team43_syscost_opt1c` (probe source not retained)"* —
   so a reader who goes looking is not left thinking the tree is incomplete.

What is not acceptable is leaving it cited as if it were in the package. This is the
one gap in the evidence chain, and a judge who greps for it will find it before
anyone else does.
