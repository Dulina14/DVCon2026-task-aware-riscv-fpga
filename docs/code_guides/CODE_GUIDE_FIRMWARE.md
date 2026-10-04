# Code guide — the firmware, in depth

Team 43 · DVCon India 2026 · companion to `README_TEAM43_PROJECT.md`
(see also `CODE_GUIDE_RTL.md`, `CODE_GUIDE_PROBES.md`, `CODE_GUIDE_TOOLCHAIN.md`)

`firmware_study/README.md` is the **reading guide** — where to start, what order,
five things that cost days. This document is the **deep dive**: what each stage
actually does, why it is written the way it is, and which lines you would break by
tidying them.

Everything the board runs is one C file, `team43_pipeline.c`, 1,955 lines. It takes
a JPEG and an English sentence and writes back the same picture with one object
outlined.

---

## 1. The shape of the program

```
argv --image X.jpg --task "serve wine"
  │
  ├─ read_image      JPEG or PPM, magic-sniffed          (on board)
  ├─ letterbox       aspect-preserving resize + grey pad (on board)
  │
  ├─ STAGE 1  detection
  │     for each of 103 ops in YOP_TYPE[]:
  │         op_conv / op_maxpool / op_upsample / op_concat / op_slice / op_add
  │     decode_detect  → boxes + per-class confidences + NMS
  │
  ├─ STAGE 2  tokenize_task + encode_task
  │     6-layer transformer over the RUNTIME prompt      (accelerator)
  │
  ├─ STAGE 3  score_classes
  │     fused head over ONLY the detected classes        (accelerator)
  │
  ├─ STAGE 4  rank = minilm × yolo, argmax
  └─ STAGE 5  draw boxes + label, write PPM
```

Four things about that shape are worth naming before any code:

**The graph is data, not code.** `main` walks `YOP_TYPE[]` from the generated
header and dispatches. The firmware does not know which detector it is holding —
swapping YOLOv8n for YOLOv8s needed no code change beyond sizing two buffers from
the graph instead of from a constant.

**Every GEMM in the program goes through one function.** `gemm_run` serves the
detector's 63 convolutions, MiniLM's Q/K/V/O projections and FFN, attention's QKᵀ
and A·V, and the scoring head. One place to optimise, one place to get wrong.

**MiniLM's cost is bounded by the *detected* classes, not by 80.** Stage 3 builds a
`uniq[]` list from the detections and scores only those. That is why scoring costs
~9,300 invocations against Milestone B's 92,640.

**Nothing about a particular image or prompt is computed off-board.** The same
binary and the same two model files handle any (image, task) pair with no
recompilation.

---

## 2. The accelerator interface — lines 63 to 334

### The device is a file

```c
static void rw(off_t off, uint32_t v)   { pwrite(g_fd, &v, 1, off); }
static uint32_t rr(off_t off)           { pread (g_fd, &v, 1, off); }
```

**The counts are in 32-bit words, not bytes.** `1` means one register. This is the
single most common trip-up in the driver interface and it silently does the wrong
thing rather than failing. The same convention applies to `dwr`/`drd` moving blocks
to DDR: `dw_bulk` divides `bytes / 4` before looping.

`g_sys++` in every one of these is not decoration. The syscall count is *the* cost
model for this system — see §8.

### `CHUNK_WORDS 256`

```c
int n = (words - done) > CHUNK_WORDS ? CHUNK_WORDS : (words - done);
```

The driver will not accept an arbitrarily large transfer, so bulk uploads are cut
into 1 KB pieces. Note the asymmetry recorded in the window comment: **the
activation upload is chunked, the weight upload is not.** That is why the YOLOv8s
window overflow described below could not have been caught at runtime.

### `wait_done` and the bug it now prevents

```c
static void wait_done(void)
{
    for (unsigned g = 0; g < POLL_LIMIT; g++) if (rr(REG_Y_DONE) & 1u) return;
    die("Y_DONE never asserted at invocation %ld", g_inv);
}
```

The comment above it is one of the most important in the file:

> *It used to be only read, never cleared, so after the very first tile it was
> permanently high and this "poll" was really one syscall of delay, about 100 µs.
> That was enough for a K=8 tile, which is why the pipeline was correct — by
> accident.*

`team43_kdepth` measured the accident failing: exact to K=256, **wrong from K=512**
(max error 127,452 at 512; 169,893 at 1024). The fix is `rw(REG_Y_DONE, 0u)` before
every START — one extra syscall per invocation, and the timing coincidence is gone.

`POLL_LIMIT 20000u` and the `die()` matter too: a completion that never arrives
produces a message naming the invocation number, not a hang on a board with one
serial line.

### `set_stream_bytes` — caching the registers that do not change

```c
static uint32_t g_cur_bytes = 0xFFFFFFFFu;
static void set_stream_bytes(uint32_t b)
{
    if (b == g_cur_bytes) return;
    rw(REG_K_BYTES, b); rw(REG_X_BYTES, b);
    g_cur_bytes = b;
}
```

K_BYTES and X_BYTES change once per GEMM, not once per tile. The sentinel
`0xFFFFFFFF` initial value guarantees the first call always writes. Two syscalls
saved per invocation, across millions of invocations — this is the whole
optimisation strategy of the program in five lines: **find state that is being
restated and stop restating it.**

### `sim_tile` — the software twin

```c
static void sim_tile(const int8_t *xt, const int8_t *wt, int32_t *par)
{
    for (int cc = 0; cc < TILE; cc++)
        for (int r = 0; r < TILE; r++) {
            int32_t a = par[cc*TILE + r];
            for (int kk = 0; kk < TILE; kk++)
                a += (int32_t)xt[kk*TILE + r] * (int32_t)wt[kk*TILE + cc];
            par[cc*TILE + r] = a;
        }
}
```

Eleven lines that must agree exactly with 8×8 processing elements of RTL. Read
alongside `mac.sv`'s `$signed(x) * $signed(k)` (`CODE_GUIDE_RTL.md` §3): the C cast
to `int32_t` and the Verilog `$signed` are the same statement about signedness, made
in two languages.

The index expressions are the tile convention, stated here in executable form:

| buffer | index | meaning |
|---|---|---|
| X tile | `kk*8 + r` | k-major, one value per array **row** |
| W tile | `kk*8 + cc` | k-major, one value per array **column** |
| Y tile | `cc*8 + r` | column-major result |

Get this wrong and **the output is plausible but silently transposed.** The same
convention appears in `pack_w` in `export_yolov8n_int8.py`, in
`team43_regstress.c`'s software reference, and in the RTL's beat order. Four places,
one convention.

`--sim` selects this path everywhere, which is what lets the entire pipeline run on
a PC with no FPGA — and it is how `sweep.sh` produced the accuracy numbers.

---

## 3. `gemm_run` — the heart

```c
static void gemm_run(const int8_t *packX, const int8_t *packW,
                     int M, int K, int N, const int32_t *bias, int32_t *out,
                     int upload_w)
```

`Y[M,N] = X[M,K] · W[K,N] + bias`, operands already tile-packed. Everything in the
project that is a matrix multiply arrives here.

### The K-depth collapse

```c
int KC = (g_kdepth > 0 && g_kdepth < KT) ? g_kdepth : KT;   /* default: whole K */
```

and then

```c
for (int k0 = 0; k0 < KT; k0 += KC) {
    int kc = (KT - k0 < KC) ? KT - k0 : KC;   /* last chunk may be short */
    set_stream_bytes((uint32_t)kc * INPUT_BYTES);
    rw(REG_Y_DONE, 0u);
    rw(REG_START,  0u);
    rw(REG_K_ADDR, (uint32_t)(K_ADDR + ((unsigned long)n0*KT + k0)*64));
    rw(REG_X_ADDR, (uint32_t)(X_ADDR + ((unsigned long)m0*KT + k0)*64));
    rw(REG_START,  1u);
    wait_done();
}
```

**320 px: 713 s → 218 s, byte-identical, md5-verified.** The reasoning is in the
comment and is worth restating because it is the project's best example of reading
RTL to find a software win:

> *The array ends an accumulation on `s_last`, which the DMA asserts at the end of a
> descriptor, and the descriptor length is just the BYTES register — so K per
> invocation is a software choice, not an RTL constant.*

And the second half, which is why it cost nothing:

> *The packed buffers are already contiguous and k-major over the whole K run for a
> fixed m0 (or n0), which is exactly the beat order the array consumes, so no
> repacking is needed either.*

Before the collapse, five syscalls bought 512 MACs. After it, the same five buy the
entire K run — up to 36,864 MACs for a K=576 convolution. **This is also what
reversed the earlier finding that software emulation beat the accelerator per
tile.**

`--kdepth 1` restores the original path exactly, which is what makes the claim
measurable rather than asserted.

### `upload_w` — the flag that saves a re-upload per M-chunk

```c
if (upload_w) dw_bulk(K_ADDR, packW, (long)NT * KT * 64);
```

> *The weight tiles are identical for every M-chunk of one convolution, so they are
> uploaded on the first chunk only.*

`op_conv` sets `first_chunk = 1` before its M loop and clears it after the first
call. A K=576 convolution at 320 px has many M-chunks and one weight block; this is
one megabyte of upload saved per chunk.

### The bias seed

```c
for (int cc = 0; cc < TILE; cc++)
    for (int r = 0; r < TILE; r++) {
        int gc = n0*TILE + cc;
        par[cc*TILE + r] = (gc < N) ? bias[gc] : 0;
    }
if (!g_sim) dwr(Y_ADDR, par, OUTPUT_WORDS);
```

The accumulator is *seeded* with the bias rather than the bias being added
afterwards. That is what the MM2S_2 (B) channel is for: the array starts its
accumulation from whatever is at `REG_B_ADDR`. `gc < N` handles the partial tile at
the end of a channel count that is not a multiple of 8 — the padding lanes seed to
zero and their results are discarded on harvest.

**Note that `REG_B_ADDR` and `REG_Y_ADDR` are both set to `Y_ADDR` here.** See §10:
the bias-window optimisation described in README §9.2 is *not* present in this
source.

### `--unsafe-batched-regs`

The `else` branch builds a 7-word array and issues it in one `rwn`. It is labelled
`PROVEN INCORRECT, kept only for A/B timing`, and the file-level comment carries the
measured evidence:

```
A  5 calls, nothing merged        416.7 us/tile   exact
B  4 calls, merges over X DONE    353.7 us/tile   CORRUPT, max err 52556
C  3 calls, merges over both      298.1 us/tile   CORRUPT, max err 69316
E  4 calls, merges over K DONE    354.1 us/tile   CORRUPT, max err 45309
```

Keeping a proven-wrong code path, clearly labelled, with the numbers that condemn
it, is better documentation than deleting it: it makes the temptation and its cost
both visible. The comment ends *"Five single-word writes per tile is the floor this
driver allows, and the driver may not be modified."*

One inconsistency worth knowing: the comment above `rwn()` itself (line 177) still
describes the *first*, superseded probe's result — *"five calls 322.4 µs/tile, four
264.8, three 192.8 — all bit-exact."* That is the `team43_regbatch` finding that
`team43_regstress` disproved. The authoritative comment is the one at `g_batched_regs`.

---

## 4. `op_conv` — the pattern every layer follows

Five phases, each separately timed, because *"each of those phases turned out to
cost more than expected at least once."*

### Phase 1 — quantise, in two passes, timed apart

```c
float amax = 0.0f;
for (long i = 0; i < nin; i++) { float a = fabsf(X->d[i]); if (a > amax) amax = a; }
...
for (long i = 0; i < nin; i++) {
    long q = lrintf(g_exact_float ? X->d[i] / sx : X->d[i] * inv_sx);
```

> *They are timed apart because the arithmetic in each is trivial and the cost is
> expected to be the memory traffic — if that is so, the fix is fewer passes, not
> cheaper operations.*

That is a hypothesis written into the instrumentation before the measurement, which
is the right way round. `g_amax_us` and `g_qcvt_us` are how it was confirmed.

The reciprocal trick — `inv_sx = 127.0f / amax`, multiply instead of divide —
is guarded by `g_exact_float` so the default path is bit-compatible with the host
simulation. A soft-float divide on a 50 MHz core with no FPU divider costs several
times a multiply, and this runs on nearly 5 million elements at 320 px.

### Phase 2 — the weight memcpy that looks redundant and is not

```c
memcpy(packW_buf, w_packed(wi), (size_t)NT * KT * 64);
```

> *This is NOT just for the upload: `--sim` used to read tiles straight from the
> mapping, and every W tile is re-read once per M row-tile. On the board that cost
> 4,502 µs per tile (1,540 s for one image) against 216 µs for the same `sim_tile`
> in Milestone B, because VEGA's mapped-file reads are pathologically slow. One
> bulk copy fixes it for both paths.*

**21× on one memcpy.** `mmap` looks free and on this core it is not. Delete this
line as a redundant copy and the simulation path becomes unusable.

### Phase 3 — im2col, blocked

```c
for (int mt0 = 0; mt0 < MT; mt0 += g_im2col_blk) { ... }
```

The write set was `MT*64` bytes — 29 KB for a K=576 conv — swept entirely before any
tile was revisited, so every 8 bytes stored cost a line fill. `g_im2col_blk = 64`
bounds it at 4 KB.

> *Each (kk_g, r) pair is still written exactly once with exactly the same value, so
> the output is bit-identical; only the order changes.*

The measured stakes: 0.66 s per call on a healthy board, **8.44 s on a degraded one
— 557 s of an 816 s query** — while every sequential pass in the program was
unaffected. `team43_im2col.c` is the probe that chose the block size and hashed every
variant to prove the bit-identity claim.

Note the loop ordering rationale: *"Outer loop over k so the READS walk the input
tensor along a row; the scattered part lands on writes."* Given one sequential side
and one scattered side, put the scatter on the writes — a write can be buffered, a
read stalls.

The bounds test is a neat idiom:

```c
if ((unsigned)iy < (unsigned)X->h && (unsigned)ix < (unsigned)X->w)
```

One unsigned compare per side instead of two signed ones; a negative index wraps to
a huge unsigned and fails the test. This is in the innermost loop of the hottest
packing loop in the program.

### Phase 4 — `gemm_run`

Bias is pre-divided into accumulator units first:

```c
double comb = (double)sx * (double)sw[n];
bias_i32[n] = (int32_t)lround((double)bf[n] / comb);
```

The float bias has to enter an INT32 accumulator whose units are `sx * sw[n]`. Doing
it per output channel is what makes per-channel weight scales work end to end. Note
`double` — this runs `N` times per chunk, not per element, so precision is free here.

### Phase 5 — dequantise, blocked

```c
const int BN = 32, BR = 64;
```

> *The obvious loop nest — n outer, r inner — reads the accumulator as `src[r * N]`,
> striding N*4 bytes on every single element. That is a transpose against a
> row-major buffer, so essentially every read is a cache miss. Measured on the board
> it cost 11.18 µs per element while the quantise pass, which does more arithmetic
> over more elements, cost 1.55 µs. **The difference was never compute; it was the
> stride.**"*

And the reason it needed no accuracy review:

> *The arithmetic per element is unchanged and every element is still visited exactly
> once, so the result is bit-identical — this needs no accuracy argument, only a
> timing one.*

`comb_buf[]` is precomputed outside the block loops so the `sx * sw[n]` multiply is
not repeated per row. The output is written planar (`Y[n*M + ...]`) because that is
what the next op expects.

### M-chunking, and the bound that was missed the first time

```c
long per_rowtile = (long)KT * 64;
int  MT_max = (int)(PACKX_CAP / per_rowtile);
if (MT_max > ROWCAP / 8) MT_max = ROWCAP / 8;
```

> *Chunk M so that (a) `packX` fits the X window and (b) the INT32 accumulator buffer
> is bounded regardless of how small K is. Without (b) a K=32 convolution asks for
> 32,768 rows at once and overruns `acc_buf`.*

Condition (a) is the one you think of. Condition (b) is the one that bites: a *small*
K means *more* rows fit, and the accumulator is sized by rows. The two bounds pull in
opposite directions and both are needed.

---

## 5. `silu` and `print_fp_fingerprint` — the numerics story

This is the part of the firmware that is most worth reading as a verification
exercise, because the optimisation, the accuracy analysis, the bug and the retraction
are all in one place.

### The optimisation that was measured and rejected

```c
static int g_exact_float = 1;   /* the table is OPT-IN, not the default */
```

The obvious `expf` per output element is 3,472,000 soft-float exponentials at
320 px. A table looked like an easy win:

```
expf, FMA allowed          VEGA float 78.1 s   query 217.3 s
table, FMA allowed         VEGA float 62.0 s   query 201.5 s
table, -ffp-contract=off   VEGA float 76.2 s   query 215.8 s
```

> *Almost the whole apparent 16 s win was the compiler contracting the
> interpolation's a\*b+c chains into FMA instructions — and that contraction is
> exactly what made the board disagree with the host simulation. … leaves the table
> worth about 2 s in 216. That is not worth carrying a change that alters output.*

An optimisation that measured well, was investigated, turned out to be mostly an
artefact of a compiler setting that broke something else, and was **reverted to
opt-in with the numbers recorded**. `--silu-table` exists so anyone can repeat the
measurement.

### The accuracy analysis, which is better than it needed to be

Linear interpolation on a 1/256 step errs by `h²/8 · max|σ''| = 1.8 × 10⁻⁷`.
That sounds harmless. The comment does not stop there:

> *…which sounds harmless and is not: an INT8 activation step here is around 0.05,
> so roughly 1.6 × 10⁻³ of the 3.47 M elements land close enough to a rounding
> boundary to flip, and a few thousand flipped INT8 values change the detection
> list. Quadratic drops the error to about 5 × 10⁻¹⁰, which flips nothing.*

The right question is not *how big is the error* but *how many quantisation
boundaries does it cross*. Three-point interpolation costs three multiplies against
one soft-float `expf` saved.

### `fpcheck`

```c
volatile float a = 1.0000001f, b = 3.0000002f, c = -3.0000005f;
p1.f = a * b + c;          /* catastrophic cancellation: fused != unfused */
p2.f = silu(0.7351f);
p3.f = silu(-2.3174f);
printf("fpcheck: %08x %08x %08x   (host and board must match)\n", ...);
```

Three deliberate choices in four lines: `volatile` stops the compiler from folding
the expression at compile time; the constants are chosen so `a*b+c` is a
*catastrophic cancellation*, which maximises the difference between one rounding and
two; and the output is **raw bits**, not a printed float, so no formatting can hide
a low-order difference.

> *Forcing FMA on with `-march=haswell -ffp-contract=fast` made the host reproduce
> the board's 0.45823 exactly, which is what identified it.*

That is how the bug was found: by making the *host* wrong in the same way as the
board, deliberately, and seeing the numbers match. The fix — `-ffp-contract=off` on
both builds — is enforced by this line, printed in the header of every run, *before*
a 200 s query rather than after.

---

## 6. The detector's tail — `decode_detect`

```c
for (int c = 0; c < YOLO_NCLASSES; c++) {
    float v = 1.0f / (1.0f + expf(-C->d[(size_t)c*A + a]));
    if (v > bs) { bs = v; best = c; }
}
if (bs < conf_thr) continue;
```

Classification first, box decode second — the `continue` skips the expensive DFL
softmax for every anchor that will not survive the threshold. At `--conf 0.02` over
three levels that is the large majority of anchors.

### Distribution Focal Loss decode

```c
for (int side = 0; side < 4; side++) {
    float mx = -INFINITY, sum = 0.0f, acc = 0.0f;
    for (int r = 0; r < YOLO_REG_MAX; r++) { ... if (v > mx) mx = v; }
    for (int r = 0; r < YOLO_REG_MAX; r++) {
        float e = expf(B->d[...] - mx);
        sum += e; acc += e * r;
    }
    d[side] = acc / sum;
}
```

YOLOv8 does not regress a distance; it predicts a **distribution** over 16 bins per
box side and takes its expectation. Two passes with `- mx` is the numerically stable
softmax — subtracting the max before exponentiating, without which `expf` overflows
on large logits. The expectation `Σ e·r / Σ e` is fused into the same pass.

Then the anchor-relative decode, distances scaled by the level stride:

```c
float ax = (a % W) + 0.5f, ay = (a / W) + 0.5f;
out[n].x1 = (ax - d[0]) * st;   out[n].y1 = (ay - d[1]) * st;
out[n].x2 = (ax + d[2]) * st;   out[n].y2 = (ay + d[3]) * st;
```

The `+ 0.5f` is the anchor cell centre. The four distances are left/top/right/bottom
*from* that centre, which is why two are subtracted and two added.

### Per-class NMS

```c
qsort(out, n, sizeof(det_t), det_cmp);
for (int i = 0; i < n; i++) {
    int drop = 0;
    for (int j = 0; j < keep; j++)
        if (out[j].cls == out[i].cls && iou(&out[i], &out[j]) > iou_thr) { drop = 1; break; }
    if (!drop) out[keep++] = out[i];
}
```

Sort by confidence, then greedily keep any box that does not overlap an already-kept
box **of the same class**. The class test matters for this application: a person
standing in front of a chair should yield both, and class-agnostic NMS would
suppress one — removing a candidate the ranker never gets to see.

The in-place compaction (`out[keep++] = out[i]`, with `keep <= i` always) needs no
second buffer.

---

## 7. The language side

### Tokenizer — lines 914 to 1033

A complete WordPiece tokenizer in about 100 lines, because the board has no Python.

**`vocab_load`** reads the 30,522-line file into one blob, replaces each newline with
`'\0'` in place — so every token is a C string pointing into the blob, with no
per-token allocation — and builds an open-addressed hash table keyed by FNV-1a. The
`\r` strip handles a file that travelled through Windows.

The last line is the guard:

```c
if (VOC.n != MINILM_VOCAB)
    die("vocab has %d entries, package says %d", VOC.n, MINILM_VOCAB);
```

`MINILM_VOCAB` comes from the generated layout header. A vocab file that does not
match the package it is being used with fails at startup instead of producing
plausible nonsense. (Its counterpart on the host is the contiguity assert in
`dump_vocab.py` — `CODE_GUIDE_TOOLCHAIN.md` §6.)

**`wordpiece`** is the standard greedy longest-match-first:

```c
while (end > start) {
    if (start) { buf[k++]='#'; buf[k++]='#'; }   /* continuation prefix */
    ...
    found = vlook(buf, k);
    if (found >= 0) { fe = (int)end; break; }
    end--;
}
```

Try the longest remaining substring; shorten by one until it is in the vocabulary;
prefix `##` on anything that is not the start of a word. A word with no match at all
becomes `[UNK]`. Note it does not emit partial results before knowing the whole word
tokenises — `local[]` is staged and copied out only on success.

**`tokenize_task`** wraps the prompt as `"task: <prompt>"` — matching the
`f"task: {task}"` in `export_minilm_int8.py` exactly — then `[CLS] … [SEP]`, padded
to `MAXTOK`. The truncation warns on stderr rather than silently dropping words.

### `encode_task` — a transformer written out longhand

Lines 1258–1321 are the same six-layer forward pass as `export_minilm_int8.encode`,
in C. Per layer:

```c
dense(MX, MAXTOK, HID, HID, &LW[L].q, &LB[L].qb, MQ);      /* Q */
dense(MX, MAXTOK, HID, HID, &LW[L].k, &LB[L].kb, MK);      /* K */
dense(MX, MAXTOK, HID, HID, &LW[L].v, &LB[L].vb, MV);      /* V */
for (int h = 0; h < NHEADS; h++) {
    ... gather head h, transposing K on the way ...
    matmul_act(qh, MAXTOK, HDIM, MAXTOK, kh, att);         /* QK^T */
    ... scale, mask, stable softmax ...
    matmul_act(att, MAXTOK, MAXTOK, HDIM, vh, ctxh);       /* A·V  */
}
dense(MCTX, ...);  residual;  layernorm;
dense(MX, ..., FFNDIM, ...);  gelu;  dense(..., HID, ...);  residual;  layernorm;
```

**`dense` vs `matmul_act` — the distinction that matters.**

| | operands | weight packing | bias |
|---|---|---|---|
| `dense` | activation × **constant package weight** | `mpack_w_pkg` (cacheable) | from the package |
| `matmul_act` | activation × **activation** | `mpack_w_act` | zero |

QKᵀ and A·V are activation × activation: both operands are runtime data, so nothing
about them can be precomputed or cached. `matmul_act` quantises *both* sides
(`mquant` on each) and dequantises by `sa * sb`. That is why attention is the one
GEMM in MiniLM that gets no help from the packed cache.

The `kh[d*MAXTOK+t] = MK[t*HID + h*HDIM + d]` gather transposes K during the
per-head extraction, so QKᵀ is a plain GEMM rather than needing a transposed
operand.

**The softmax is masked before it is stabilised:**

```c
att[r*MAXTOK+c] = att[r*MAXTOK+c] / sqrt((double)HDIM) + (mask[c] ? 0.0 : -10000.0);
if (att[r*MAXTOK+c] > mx) mx = att[r*MAXTOK+c];
```

Scale by `1/√d`, add `-10000` to padded positions, find the max, then
`exp(x - mx)`. The `-10000` convention matches the export script's
`(1.0 - mask) * -10000.0`, which matters because the two implementations are
compared numerically.

**Everything here is `double`.** MiniLM runs 8 tokens, once per query; the detector
runs millions of elements. Spending double precision where it is cheap and float
where it is not is a deliberate allocation, not an inconsistency.

### `score_classes` — where the architecture pays off

```c
tensor_t oe = TEN(T_OBJ_EMB);
for (int i = 0; i < n; i++)
    for (int h = 0; h < HID; h++) {
        double o = t_get(&oe, cls[i], h);
        fused_buf[i*FUSED + h]         = temb[h];
        fused_buf[i*FUSED + HID + h]   = o;
        fused_buf[i*FUSED + 2*HID + h] = fabs(temb[h] - o);
        fused_buf[i*FUSED + 3*HID + h] = temb[h] * o;
    }
dense(fused_buf, n, FUSED, HID, &h0w, &h0b, hid_buf);   /* 1536 -> 384 */
for (...) if (hid_buf[i] < 0.0) hid_buf[i] = 0.0;       /* ReLU */
dense(hid_buf, n, HID, 1, &h3w, &h3b, logit_buf);       /* 384 -> 1 */
out[i] = 1.0 / (1.0 + exp(-logit_buf[i]));              /* sigmoid */
```

**`t_get(&oe, cls[i], h)` is a table lookup, not a forward pass.** The class side of
the comparison was encoded offline (`CODE_GUIDE_TOOLCHAIN.md` §0) because it depends
only on the fixed label strings. Only the prompt goes through the transformer.

Combined with `n` = the number of *detected* classes rather than 80, that is why
MiniLM costs ~17 s against detection's ~250 s. **The language half of a
"task-aware" system is 7% of its runtime.**

### The packed-weight sidecar

```
--emit-packed FILE   run once on the host, writes the cache
--packed FILE        use the cache; each pack becomes one memcpy
```

`mpack_w_pkg` rearranges a weight matrix into 8×8 tiles. Trivial arithmetic —
a byte copy — but scattered writes, and *"on this core every scattered access is a
DDR round trip. Measured on hardware: 10.7 MB rearranged in 130 s, about 0.08 MB/s,
or roughly 270 cycles per byte moved."*

The weights are constant, so it can be done once. Two design details make the cache
safe:

```c
if (mb != (uint64_t)MODEL.size)
    die("packed cache was built for a %llu byte package, this one is %llu", ...);
```

and blocks keyed by `(offset-in-package, K, N)` rather than by call order:

> *so a cache built from a different model cannot be silently mistaken for a valid
> one — the lookup fails and the program stops.*

A cache keyed by call order would silently serve layer 3's weights for layer 4 after
any reordering. A magic string, a version, a package-size check and a content key —
four independent ways for a mismatched cache to fail loudly. And *"the original
package is untouched and still works on its own"*: the sidecar is an optimisation,
not a format change.

---

## 8. The cost model, and what it implies

Two numbers explain every optimisation in this firmware:

| | measured |
|---|---|
| arithmetic of one 8×8×8 tile | **~0.16 µs** |
| fabric time, 8 beats + 17-cycle fill/drain | **~0.5 µs** |
| one accelerator invocation, five syscalls | **~485 µs** (416.7 µs in `team43_regstress`) |
| wall clock per **output tile**, before the k0 collapse | **~1,800 µs** |

The last two are different quantities and README §11.3 keeps them apart: an output
tile took ~38 invocations on average before `--kdepth 0` made it one.

**Cost is memory accesses, not arithmetic.** Everything that helped removed
*accesses*; nothing that reduced *operations* mattered. That is why widening the
array was analysed and rejected — a 16×16 array would quadruple the MACs behind the
same syscall.

Which is why the instrumentation is structured the way it is. `g_pack_us` remains
the sum of every packing site *"so old logs remain comparable"*, with new counters
breaking it down:

```c
static long g_im2col_us, g_im2col_n;   /* YOLO activation im2col   */
static long g_wcopy_us,  g_wcopy_n;    /* YOLO weight-tile memcpy  */
static long g_mpx_us,    g_mpx_n;      /* MiniLM activation pack   */
static long g_mpw_us,    g_mpw_n;      /* MiniLM weight pack (pkg) */
static long g_mpa_us,    g_mpa_n;      /* MiniLM act-as-weight pack*/
static long g_jpeg_us;                 /* on-board JPEG decode     */
```

*"timers only; no behavioural change"* — and keeping the aggregate stable while
adding detail means a log from August is still comparable with one from September.
Every counter has a matching `_n` so the report can print a per-call mean rather
than only a total.

`g_inv` counts invocations `(m0,n0,k0)`; `g_tiles` counts output tiles `(m0,n0)`.
Their ratio *is* the K-depth saving, measured rather than predicted.

---

## 9. `main` — the defaults are the findings

Three defaults in `main` are each backed by a measurement table in the comment
beside them. This is the most quotable part of the file.

### `--conf 0.02`, not 0.25

```
conf   mAP@0.5   top-1 selection   right answer on the list at all
0.01    0.2046        0.3036              0.5714
0.02    0.2008        0.3107              0.5214
0.05    0.1914        0.2857              0.4500
0.10    0.1786        0.2643              0.3857
0.25    (the old default) far below all of these
```

> *Task-driven selection is not object detection: a box only has to be good enough
> for the ranker to consider it, and **every box withheld is a candidate the ranker
> never sees**.*

A detector tuned for detection is mistuned for this system. The third column is the
oracle from `eval_coco_tasks.py` — the ceiling the ranker is working against.

The stopping rule is honest about the trade: *"Accuracy still climbs below 0.02 but
the shortlist stops being readable."*

### `--rank-by product`

```
rank-by product  mAP@0.5 0.0975   top-1 selection 0.1976
rank-by minilm   mAP@0.5 0.0705   top-1 selection 0.1976
```

Identical on the single box the demo marks, 38% better over the ranked list — and
the reason for the change is stated as a correction rather than an improvement:
*"Shipping minilm-alone while evaluating product was an inconsistency."* Measured on
2,150 COCO-Tasks test images with the deployed INT8 models.

### `--draw-top 8`

> *At a low confidence floor `nd` can be 40+, and a picture carrying 46 boxes tells
> the viewer nothing about which one the system chose or why.*

A presentation decision that follows directly from the `--conf` decision, kept
separate from it so neither compromises the other: low threshold for the ranker,
short list for the eye. The printout says what it hid:

```
... %d further candidate%s below rank %.5f not shown or drawn
```

### `--rect` / `--imgh`

> *A canvas matching the photo aspect wastes no pixels on padding: 128×192 costs
> 1.5× a 128 square for the same object scale, where 192×192 costs 2.25×.*

Letterboxing a 4:3 photo into a square spends about a third of every convolution on
grey. Since cost scales with canvas *area* and accuracy with object *scale*, a
rectangular canvas buys scale at 1.5× where a bigger square costs 2.25×.

### The DDR window guard

```c
#define WINDOW    0x200000L               /* 2 MB per scratch window */
```

> *These were 1 MB each, which is enough for YOLOv8n: its largest packed weight
> block is 294,912 bytes. YOLOv8s is twice as wide and its largest block is
> 1,179,648 bytes, which would have run 128 KB past the end of the K window and
> silently corrupted the activations sitting in X. **The weight upload is not
> chunked — only the activation upload is — so nothing would have caught it at
> runtime; the detections would just have been wrong.***

`main` now asserts the fit at startup against the graph actually compiled in, by
scanning every weight for the largest packed block:

```c
for (int i = 0; i < YOLO_NWEIGHTS; i++) {
    long b = (long)((YW_N[i]+7)/8) * ((YW_K[i]+7)/8) * 64;
    if (b > worst) { worst = b; wi_worst = i; }
}
if (worst > WINDOW)
    die("weight %d packs to %ld bytes but the K window is %ld; "
        "enlarge WINDOW or chunk the weight upload", wi_worst, worst, (long)WINDOW);
```

A bound derived from the model rather than assumed is the difference between a
startup error naming the offending weight and a wrong answer. (Minor: the comment at
line 76 calls this `check_windows()`; the check is inline in `main` and no function
by that name exists. Worth renaming one or the other.)

The same principle sizes the buffers immediately below it — *"sized from the graph,
not from a constant: YOLOv8n tops out at 256 output channels and YOLOv8s at 512, and
a hardcoded 256 silently overruns on the wider model"* — which is the two-line change
that let the detector be swapped at all.

### Stage 4, ranking

```c
rank[i] = product ? ms * dets[i].conf : ms;
if (rank[i] > best_rank) { best_rank = rank[i]; best = i; }
```

The whole selection is two lines. Everything before it exists to make `ms` and
`conf` mean something.

---

## 10. Two discrepancies to resolve

Found while writing this guide. Both are documentation-vs-code mismatches, not bugs
in running code.

### 10.1 The bias-window optimisation (opt1c) is documented but not in the source

README §9.2 describes it in the present tense — *"The bias tiles **now** live in
their own 64 KB window at `0xBF600000`, uploaded once per GEMM, with the tile loops
swapped so `n0` is outer and `REG_B_ADDR` is set once per column-tile"* — and README
§14's memory map lists a row `B (opt1c) | 0xBF600000 … 0xBF610000 | 64 KB`.

In every copy of `team43_pipeline.c` in this tree — `./`, `app/`,
`app/TEAM43_FROZEN/yolov8n/`, `eval/`, `eval_s/`, `firmware_study/src/` — `gemm_run`
contains:

```c
rw(REG_B_ADDR, (uint32_t)Y_ADDR); rw(REG_B_BYTES, OUTPUT_BYTES);
...
for (int m0 = 0; m0 < MT; m0++)
    for (int n0 = 0; n0 < NT; n0++) {
        ... seed par from bias ...
        if (!g_sim) dwr(Y_ADDR, par, OUTPUT_WORDS);
```

B still points at `Y_ADDR`, `m0` is still the outer loop, and the bias tile is still
re-uploaded per output tile. `0xBF600000` appears nowhere outside the README. So the
shipped firmware is the **pre-optimisation** version.

Either the optimised source was not brought into this tree, or §9.2 documents work
that was prototyped and not merged. Worth resolving before submission, because §9.2
currently describes the shipped binary inaccurately.

### 10.2 `team43_syscost_opt1c.c` is cited in three places and is not present

Covered in `CODE_GUIDE_PROBES.md` §7. It is the only cited evidence for the §9.2
correctness claim (*"5/5 cases, 64/64 match"*) and for the §8 per-access cost table,
and it is not anywhere in `final_phase`. `run_n_opt1c.sh` is likewise listed in §16's
scripts line and absent.

These two are almost certainly the same gap: an optimisation branch, and its probe,
that exist somewhere else. Recovering that branch resolves both; if it cannot be
recovered, §9.2, §8 and §16 should say so plainly rather than describe it as shipped.

---

## 11. If you read only one thing

`gemm_run`, lines 258–334. It is where the RTL's `s_last`, the DMA descriptor
length, the tile packing convention, the bias-seed channel, the syscall cost model
and the one proven-wrong optimisation all meet — 77 lines that every other part of
the system is arranged around.
