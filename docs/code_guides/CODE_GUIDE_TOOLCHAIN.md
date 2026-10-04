# Code guide — the host-side toolchain

Team 43 · DVCon India 2026 · companion to `README_TEAM43_PROJECT.md`
(see also `CODE_GUIDE_RTL.md`, `CODE_GUIDE_PROBES.md`, `CODE_GUIDE_FIRMWARE.md`)

---

## 0. The constraint that shapes every file here

The Stage 3B rule is explicit about the division of labour:

> The host **may**: perform one-time checkpoint conversion; quantize constant
> model weights; generate static model metadata; create independent golden
> references; transfer images, executables, weights and output files over SCP.
>
> The host **must not** compute, for a particular runtime input: YOLO detections;
> prompt embeddings; runtime activations; candidate task scores; the final
> selected object.

Everything in this guide is host code, so the first question about each script is
not *what does it do* but **is it allowed to do it**. The answer is the same in
every case and rests on one test:

> **Does the output depend on a runtime input?**
> If no, it is constant model data and may be produced once, offline.
> If yes, it belongs on the board.

Two consequences are worth naming because they look like edge cases and are not:

- **The 80 COCO label embeddings are precomputed offline.** They are a function of
  the trained weights and the fixed label strings — `"object: wine glass"` — and of
  nothing else. No image, no task prompt. The result is *mathematically identical*
  to computing them at runtime; the board would just spend 80 encoder passes
  arriving at the same numbers. This works only because MiniLM scoring never sees
  pixels: it scores *task text* against *class name text*, and the detector's job is
  only to say which class names are present. That architectural fact is what makes
  the precomputation legal rather than a shortcut.
- **The task prompt is not embedded here.** `export_minilm_int8.py` embeds one
  prompt, `"serve wine"`, and writes it to `team43_minilm_golden.json` — labelled
  *host golden reference for validation*. Golden references are explicitly allowed.
  The board computes the runtime prompt embedding itself, and the golden exists so
  that the board's answer can be checked against something that was written
  independently of it.
- **JPEG → PPM conversion is container conversion, not inference.** The board has no
  JPEG decoder. `export_yolov8n_int8.py` writes `bus.ppm` / `zidane.ppm` as raw P6,
  and its header states the boundary precisely: *"converting the container is not
  inference — the board still does letterbox, normalise, detect, NMS and scoring
  itself."*

Every export script carries a `WHAT IS LEGAL HERE` / `WHAT IS NOT DONE HERE` block
in its docstring. That is deliberate: the compliance argument lives next to the code
it describes, so it cannot drift away from it.

---

## 1. Map of the toolchain

Two locations, because the two halves have different dependencies.

**`C:\Users\<user>\Documents\Codex\2026-08-15\` — the export half.**
Needs torch, transformers, ultralytics. Runs on the Windows PC. Runs *once*.

| script | produces |
|---|---|
| `yolo_flatten.py` | the 103-op graph + a NumPy re-implementation (library, not a CLI) |
| `yolo_int8_eval.py` | the INT8 simulation of that graph (library) |
| `export_yolov8n_int8.py` | `team43_yolo_int8.bin`, `team43_yolo_graph.h`, `team43_yolo_golden.json`, `*.ppm` |
| `export_minilm_int8.py` | `team43_minilm_int8.bin`, `..._manifest.json`, `..._golden.json` |
| `make_layout_header.py` | `team43_minilm_layout.h` from the manifest |
| `dump_vocab.py` | `vocab.txt` (30,522 tokens, 262,030 bytes) |

**`D:\HDL\final_phase\eval\` — the evaluation half.**
Needs only Python and JSON for two of its three box sources. Runs repeatedly.

| script | does |
|---|---|
| `make_image_list.py` | picks the image subsample, once, so runs are comparable |
| `eval_coco_tasks.py` | scores mAP@0.5 and top-1 selection against COCO-Tasks |
| `sweep.sh`, `step.sh`, `step2.sh`, `sim14.sh` | drive the pipeline binary over those lists |

Plus `firmware_study/build.sh` and `scripts/verify.sh`, covered in §6 and §7.

---

## 2. `yolo_flatten.py` — turning a PyTorch module tree into a list

**The idea.** The board cannot execute a `torch.nn.Module`. It can execute a flat
list of six op types. So the export walks the fused YOLOv8n module tree once and
emits that list:

```
CONV     in, w_idx, k, s, pad, act(silu|none), out
MAXPOOL  in, k, s, pad, out          (SPPF, k=5 s=1 pad=2)
UPSAMPLE in, scale, out              (nearest)
CONCAT   [ins], out
SLICE    in, c0, c1, out             (C2f split)
ADD      a, b, out                   (Bottleneck shortcut)
```

Six ops cover the whole detector. That is the finding, and it is why a C
interpreter in ~200 lines can replay a network that PyTorch describes in thousands.

**The validation, stated in the docstring before any code:**

> *"If the NumPy output matches torch's, the op list is a faithful description of
> the network and the C interpreter can just replay it."*

This is the load-bearing sentence of the whole detector port. The `run_numpy`
function re-implements the forward pass **from the fused weights only** — from the
same arrays the board will receive, not from the module tree. If that agrees with
torch, then the op list plus the weight blob is a complete description of the
network, and everything downstream is a transcription problem rather than a
modelling problem. Measured agreement: **1.8 × 10⁻⁷ on class scores, 1.8 × 10⁻⁴
pixels on box coordinates** (README §10.1).

**Two details that matter downstream.**

`emit_conv` reshapes each weight to `[Cout, Cin*kh*kw]` at emit time:

```python
wi = g.add_weight(W.reshape(Cout, Cin * kh * kw), B)     # im2col order
```

so convolution has *already* become a matrix multiply before quantisation, before
packing, before the board ever sees it. The board's `op_conv` does im2col on the
activations and a GEMM; it never implements a convolution as such.

`convmod` is taken **after** `.fuse()`, so BatchNorm is already folded into the
conv weights and bias. The comment says so — `"convmod is a torch Conv2d (already
BN-fused, has bias)"` — and it is the reason the board has no BatchNorm op. Folding
BN offline is constant-weight arithmetic; it is legal, and it removes an entire op
type from the runtime.

**C2f, the one structurally interesting block.** `emit_c2f` shows how a
split-and-accumulate block becomes flat ops: one conv to `2c` channels, two SLICEs,
then each bottleneck's output appended to a growing list that is CONCATed at the
end. No control flow survives into the op list — the branching is resolved at
export time, which is exactly what makes the list replayable.

---

## 3. `yolo_int8_eval.py` — the quantisation decision, made where it can be measured

This file is the INT8 simulation of the flattened graph, and it is where the
quantisation scheme was chosen rather than assumed.

**Per-output-channel weight scales, and why they are free:**

```python
PERCH = True    # per-output-channel weight scales (free: dequant is per column)
```

The dequantisation step is

```python
y = (acc + bi[None, :]) * (sx * sw)[None, :]
```

`sw` is a vector over output channels and it multiplies a *column* of the result.
Per-tensor scaling would multiply by a scalar instead. Both are one multiply per
output element. So per-channel accuracy costs nothing at runtime — the only price
is `N` floats of scale table per weight, which the package already carries. This is
the kind of decision that is obvious once written down and easy to get wrong by
defaulting to per-tensor because it is simpler to describe.

**Activations are quantised per-GEMM, dynamically:**

```python
def q_act(x):
    s = np.abs(x).max() / 127.0
```

`q_act` runs on the im2col matrix each time, so there is no calibration dataset and
no calibration step anywhere in this project. The board does the same thing — one
pass over the packed activations to find the max, then scale. It costs a pass but
removes a whole class of "the calibration set did not represent the deployment
distribution" failure.

**`s == 0 → s = 1.0` appears in every quantiser here** (`q_act`, `q_w`, and `q8` in
the MiniLM export). An all-zero tensor is not hypothetical in a pruned or
zero-initialised layer, and without the guard it produces a division by zero that
turns into silent NaNs three ops later.

**Accumulation is `int64` in the simulation:**

```python
acc = xq.astype(np.int64) @ wq.astype(np.int64).T
```

The hardware accumulates in INT32 (`WY = 32`). int64 here is deliberately *wider*
than the hardware, so if the simulation and the board ever disagree, the difference
is an overflow the hardware hit and the simulation did not — a real result rather
than a shared blind spot. Worth knowing when reading a mismatch.

**`letterbox()` is duplicated on the board.** Resize preserving aspect ratio, pad to
square with `(114,114,114)`, transpose to CHW, divide by 255. The board implements
the identical operation in C. That duplication is intentional: it is the *input*
side of the four-implementation agreement described in README §10.1.

---

## 4. `export_yolov8n_int8.py` — packing into the accelerator's tile order

The one function worth reading closely is `pack_w`, because it is where host and
RTL meet:

```python
def pack_w(wq, K, N):
    """[N,K] int8 -> accelerator tile order, tile (n0,k0), byte kk*8+cc."""
```

`kk*8 + cc` — k-major within a tile — is the beat order `axis_sa` wants, the same
convention `team43_regstress.c`'s software reference states as
`W[kk*8 + cc]`, and the same one the K-depth collapse depends on for contiguity
(`CODE_GUIDE_PROBES.md` §2). **One convention, written down in four places, and it
has to be the same in all four.** When it is right, the board's `gemm_run` can point
the DMA at an offset and stream; when it is wrong, nothing works and the failure
looks like a numerical problem rather than a layout problem.

Doing the packing offline means the board never permutes weights at all — 63 conv
weight tensors arrive already in tile order.

**Padding to multiples of 8** is done on the scale and bias tables:

```python
blob += np.pad(sw, (0, (-N) % 8)).astype(np.float32).tobytes()
```

`(-N) % 8` is the idiomatic "round up to 8" in Python. The array is 8 wide, so a
channel count that is not a multiple of 8 leaves a partial tile; padding the *side
tables* rather than special-casing the last tile keeps the board's inner loop
branch-free.

**Tensor last-use analysis:**

```python
lastuse = [-1] * g.n_tensors
for oi, op in enumerate(g.ops):
    for t in op['i']: lastuse[t] = oi
```

Computed on the host, shipped as a constant, used on the board to free intermediate
tensors as it goes. This is compile-time liveness analysis, and it is what keeps a
103-op graph inside the board's memory. The detect heads are given `lastuse =
len(g.ops)` so they survive past the last op — a one-line detail that would
otherwise free the outputs before they are read.

**The output is a `.h` of C constants, not a file to parse.** The board does no JSON
parsing and no graph loading; the graph is `#include`d and burned into the binary.
The header opens with `/* auto-generated by export_yolov8n_int8.py - do not edit */`,
which is the right warning to have on a 103-op generated file.

---

## 5. `export_minilm_int8.py` — a transformer forward pass with no transformer

### The unusual choice

`encode()` implements MiniLM's forward pass **by hand, from the `state_dict`**:

```python
def encode(sd, ids, mask):
    """Manual MiniLM forward from the state_dict. No HF model needed."""
```

No `AutoModel.from_pretrained`. Embeddings, LayerNorm, 6 × (QKV → scaled dot-product
attention → output projection → residual+LN → FFN with GELU → residual+LN), then
mean-pool over the mask and L2-normalise. Indexed by string keys like
`"encoder.encoder.layer.3.attention.self.query.weight"`.

This looks like extra work and is the opposite. **The board has to implement exactly
this forward pass in C.** Writing it once in ~35 readable lines of PyTorch tensor ops
produces a specification that a C port can be checked against line by line. If the
export had called the HF model, the C port would have had to be validated against a
black box, and every discrepancy would have started with "where in the library does
this happen?"

It also makes the ops list in the manifest honest, because it was derived from code
someone wrote rather than from documentation:

```python
"board_ops":       ["tokenize", "embedding lookup", "LayerNorm", "softmax",
                    "GELU", "residual", "mean-pool", "L2-norm", "sigmoid"],
"accelerator_ops": ["Q/K/V/O projections", "FFN1", "FFN2",
                    "QK^T", "A*V", "head.0", "head.3"],
```

That split — what the CPU does, what the 8×8 array does — is the deployment plan in
two lines. Note that `QK^T` and `A·V` are on the accelerator: attention is GEMM too,
even though its operands are activations rather than weights.

### The scoring head

```python
f = torch.cat([t_emb, o_emb, (t_emb - o_emb).abs(), t_emb * o_emb], -1)   # 4*384 = 1536
h = torch.relu(f @ head0.T + head0.b)                                      # 1536 -> 384
return torch.sigmoid((h @ head3.T + head3.b).squeeze(-1))                  # 384 -> 1
```

The fused feature `[t, o, |t−o|, t⊙o]` is the standard sentence-pair interaction
form. Both projections are GEMMs on the accelerator; only ReLU and sigmoid stay on
the CPU.

`head.0` / `head.3` — the gap in numbering is not an error. The trained head is a
`Sequential` whose indices 1 and 2 are ReLU and Dropout, neither of which has
parameters and neither of which exists at inference. Keeping the original names
means the `state_dict` keys match the checkpoint exactly, which matters more than
tidy numbering.

### Mixed precision in one package

INT8 for anything large, FP32 for anything small and sensitive:

```python
for k, v in sd.items():
    if (".bias" in k) or ("LayerNorm" in k):
        fp32[k] = ...
```

LayerNorm's γ and β and all biases stay float. They are a negligible fraction of
23,088,004 bytes and they are exactly the parameters where an INT8 rounding error
propagates into every subsequent token. The GEMM weights — where the size is —
are INT8.

### `MAX_TOKENS = 8`

The training-time `inference.py` used 32. Eight covers `"task: serve wine"` (~6
WordPiece tokens) with room to spare, and it shrinks every attention matrix from
32×32 to 8×8 — which, on an 8×8 array, is exactly one tile. The reduction is
recorded in the manifest as `"note_max_length_original": 32` rather than silently
applied. That is the right way to ship a deployment-time simplification: make it a
recorded field, not a constant someone has to rediscover.

---

## 6. `make_layout_header.py` and `dump_vocab.py` — the small scripts with the sharp edges

### `make_layout_header.py`

Turns the manifest into `#define`s so the board indexes a 23 MB blob with
compile-time constants and parses no JSON. Two lines of policy in its docstring are
worth keeping: *"No JSON parsing on the board"* and *"The .bin file is NOT modified
or re-exported"* — the header is a **view** of the package, so regenerating it can
never invalidate the binary the board already has.

The `cfloat()` helper carries its own bug report, and it is a good one:

```python
def cfloat(v):
    """Emit a VALID C float literal.

    The previous version wrote f"{v:.12g}f", which turns a scale of exactly
    1.0 into the token `1f`. That is legal C#, not legal C -- gcc rejects it
    with "invalid suffix 'f' on integer constant", so every FP32 side table
    in the generated header (all LayerNorm params and all biases, SCALE = 1.0)
    made the header uncompilable. Force a decimal point.
    """
```

`%.12g` drops the decimal point for exact integers. Every FP32 side table has
`scale = 1.0`. So the naive formatter broke *every* FP32 entry at once — which is
the good case, because it failed loudly at compile time. A formatter that broke one
entry in a thousand would have shipped.

### `dump_vocab.py`

Writes the 30,522 WordPiece tokens, one per line, ordered by id.

The obvious implementation — `tok.save_vocabulary('.')` — is a **silent no-op in
transformers 5.x**. It returns without error and writes nothing. So the script goes
through `get_vocab()` and sorts explicitly, then asserts the property the board's
tokenizer actually depends on:

```python
ids = [i for _, i in ordered]
assert ids == list(range(len(ids))), "vocab ids are not contiguous"
```

The board indexes this file by line number. If the ids were not contiguous from
zero, every line after the first gap would map to the wrong token, the prompt would
tokenise into nonsense, and the scores would be plausible-looking garbage. One
`assert` converts that into a build-time failure.

The script finishes by tokenising the demo prompt and printing ids, mask and tokens
— a three-line smoke test that the file it just wrote is usable.

---

## 7. `build.sh` — the flag that makes the golden reference trustworthy

```sh
# Both builds MUST use -ffp-contract=off. RISC-V contracts a*b+c into a single
# FMA and x86 does not; without this flag the board and the host simulation
# disagree in the last bits and the fpcheck line in the run header will differ.
```

This is the project's best single example of a bug that could only be found by
distrusting your own tools.

`a*b + c` compiled for RISC-V becomes one fused multiply-add with a single rounding.
The same source on x86 becomes a multiply and an add with two roundings. Both are
correct C — the standard *permits* contraction. The consequence is that the host
"golden reference" and the board were computing subtly different functions, so the
reference was **wrong in the same direction as nothing else**, and every comparison
against it carried an unexplained residual.

`-ffp-contract=off` on **both** builds is the fix, and the file makes the pass
condition concrete:

```sh
# A build is correct when the host and the board print the same fpcheck line:
#   fpcheck: 34800000 3efe65ea be54d74a
```

Three hex words printed by both binaries at startup. Not an approximate agreement,
not a tolerance — bit-identical or the build is wrong. Note it is checked *at
startup*, before any real work, so a mis-built binary is caught in the first line of
output rather than in a discrepancy 200 seconds later.

The same file also shows the two-target reality of the project: the board build runs
inside the `buildroot-2004_new` container at `--user $(id -u):$(id -g)` (so output
files are not root-owned), and the host build is plain `gcc` of **the same source
file**, with `--sim` selecting the software emulation of the accelerator. One source,
two targets, one fingerprint.

## `verify.sh` — the first thing that runs on the board

```sh
# Step one of every deployment. Run it in /tmp/team43 straight after untarring.
# A payload file that arrived short will otherwise decode into a picture whose
# bottom is garbage, and nothing else in the pipeline will complain.
```

46 MB of model packages over SCP to an embedded board. A truncated transfer produces
a *plausible* failure — a half-decoded image, a weight tensor that is mostly right —
and nothing downstream checks. The script tries `sha256sum`, falls back to `md5sum`,
and falls back again to a presence check if the board has neither, then prints
`PAYLOAD OK` with a count or `PAYLOAD BAD` with the specific files to re-transfer.

Pair it with `team43_memcheck` (`CODE_GUIDE_PROBES.md` §1): between them, two
commands and about ten seconds rule out the two failure modes — bad transfer, bad
DRAM — that otherwise consume a fourteen-minute query before revealing themselves.

---

## 8. `eval_coco_tasks.py` — measuring against the paper's own metric

**402 lines. The file that turns "it works" into a number someone else can check.**

### The framing

The metric is AP@0.5 per task against the preferred-object annotations, averaged
over the 14 tasks, from Sawatzky, Souri, Grund & Gall, *"What Object Should I Use? —
Task Driven Object Detection"*, CVPR 2019, Tables 2 and 3.

Three box sources exist so that **each of our numbers lands in the right column of
their table** — the script says exactly that:

| flag | boxes | comparable to |
|---|---|---|
| `--boxes gt` | ground-truth COCO objects | their *gt bbox* column |
| `--boxes faster` | the paper's own `detections_faster.json` | their *Faster-RCNN detections* column |
| `--boxes yolov8n` | our YOLOv8n at `--imgsz` | the full lightweight pipeline |
| `--boxes detfile` | any COCO detection file | our board's `--detect-json` output |

The first two need no images and no torch — JSON only. That means the ranking half
of the system can be scored on a machine with nothing installed, and it isolates
ranker quality from detector quality by construction.

And crucially:

> *"Our probability is MiniLM(task, class_name), read from a precomputed 14×80 table
> produced by the DEPLOYED INT8 package — so this evaluates exactly the numbers the
> board computes, not a separate float model."*

The evaluation scores the artefact that ships. A great many published INT8 results
do not.

### The self-check on the metric itself

```python
ap.add_argument('--method', choices=['minilm','pickbest'], default='minilm',
                help="pickbest reproduces the paper's pick-best-class baseline "
                     "(published 0.386 on gt boxes) as a check on the AP code")
```

`--method pickbest` re-implements the paper's simplest baseline — per-class
preference rate on the *train* split, then keep the highest-rated class present in
each image — for which the paper publishes **0.386 on gt boxes**. If our AP code
reproduces 0.386, the AP code is right. If it does not, every other number the
script prints is suspect *before* any of them is interpreted.

This is the same discipline as the probes: a control that can fail, run before the
measurement that matters. Anyone can implement AP@0.5; getting the 101-point
interpolation, the monotone precision envelope and the greedy one-to-one matching
all right is another thing, and `pickbest` is how that was established rather than
assumed. The paper's full Table 2 is printed after every run for reference.

### The ranking ablation

```python
ap.add_argument('--rank', choices=['auto', 'product', 'minilm', 'det'], ...)
```

- `product` — MiniLM × detection score, the paper's rule
- `minilm` — MiniLM alone, what the deployed pipeline uses by default
- `det` — **detection score alone**, described in the help as *"a control that shows
  how much of the score is the language model at all"*

That third option is the honest one. It answers "would a detector that just picks
its most confident object do as well?" — and a task-aware selector that cannot beat
it is not doing anything. Shipping the control alongside the result is what makes
the result mean something.

### `top1_selection` — the metric that matches the demo

mAP@0.5 scores a *ranked list*. The demo marks **one box**. Those are different
questions, and the script measures both:

| quantity | meaning |
|---|---|
| **box hit** | the marked box overlaps a preferred object at IoU ≥ 0.5 — the user sees the right thing circled |
| **class hit** | the marked box is of a preferred *class*, any instance |
| **no candidate** | the detector returned nothing at all |
| **oracle** | a preferred object was *somewhere* on the candidate list |

The class/box split *"separates a ranker that reasoned correctly but was handed a
bad box from a ranker that chose the wrong kind of object entirely."* Those failures
look identical in a single accuracy number and have completely different fixes.

The oracle is the strongest idea in the file, and the code says why:

```python
# oracle: was the right answer even on the list? An upper bound on what
# any ranker could score given these boxes. The gap between orac and hit
# is the ranker's loss; the gap between orac and 1.0 is the detector's.
```

which the summary then prints as a direct attribution:

```
-> attribution: detector loses X, ranker loses Y, of the Z total miss
```

Two components, separately actionable, from one run. Without it, a disappointing
top-1 number is a mystery; with it, it is a work item assigned to a specific half of
the system.

One more line of care — the denominator:

> *"Denominator is images that actually contain a preferred object; where there is
> none the question has no right answer and scoring it is meaningless."*

Including unanswerable images would depress every number by a constant and make
comparisons against the paper invalid.

### Two small robustness details

```python
except ValueError:
    bad += 1          # a run killed mid-write leaves one torn record
```

Board runs get interrupted. `load_dets` accepts either a JSON array or
newline-delimited records, skips the torn final line, and **prints the count** — it
tolerates the damage without hiding it.

```python
if d.get('category_id') == 0:      # placeholder: image had no detections
    continue
```

The pipeline emits a placeholder record for an image with no detections. That
distinguishes *"processed, found nothing"* from *"never processed"* — which is the
difference between a real zero and a missing data point, and it feeds the
`no_candidate` column above.

---

## 9. `make_image_list.py` and the shell drivers

### `make_image_list.py`

> *"The union of the fourteen task test splits is 9,505 distinct val2014 images (900
> per task, mostly disjoint). Running the detector once over the union and caching
> the result lets all fourteen tasks be scored from one pass, and lets the
> confidence threshold be swept afterwards for free."*

Detection is the expensive part and it does not depend on the task. Deduplicating
across tasks turns 14 × 900 detector passes into 9,505, and caching the detections
as JSON turns every subsequent threshold or ranking experiment into a re-score of
existing data. The `--per-task N` cap produces the smaller subsamples
(`images_200.txt`) used for the timed sweeps.

`eval_coco_tasks.py --images` then restricts every task to that same id list, *"so
two detection runs over the same subsample are compared on equal ground"* — which is
what makes a 128 px run and a 320 px run comparable at all.

Missing files are counted and reported rather than skipped silently.

### `sweep.sh` — resumability

```sh
[ -s "det_int8_$3.json" ] && { echo "skip $3"; return; }
```

Five configurations (192/320/128/192/256 px at conf 0.05 or 0.01), each writing its
own detection file, each skipped if a non-empty one already exists, each with
`--resume` passed to the pipeline and `date` stamps around it. A sweep like this
runs for hours; making it idempotent means an interruption costs the current
configuration rather than the run.

### `sim14.sh` — the qualitative evidence set

One image and one task prompt per COCO task, all fourteen, `--sim --imgsz 320`,
greping only the `SELECTED :` line and prefixing it with the task number. It is the
fastest way to see *what the system chose*, as opposed to how it scored:

```
task 10  ...  serve wine       -> wine glass
task  6  ...  get lemon out of tea
task 13  ...  extinguish fire
```

The prompts are the COCO-Tasks task definitions — `"step on something to reach top
of a shelf"`, `"get potatoes out of fire"`, `"pound carpet"` — not invented demo
phrases. Several are deliberately oblique: the right answer to *"pound carpet"* is
not a carpet, and no amount of object detection alone gets there. `--out
sim14/taskN.ppm` writes the annotated image per task, so the set doubles as the
figure source.

---

## 10. What holds it together

Reading the host toolchain as a whole, four principles recur:

1. **Do it once, offline, and only if it is input-independent.** Every export script
   states its own legality argument in its docstring, next to the code.
2. **Generate, do not parse.** The board reads no JSON and loads no graph. Op lists,
   tensor offsets, scales and class names all arrive as `#define`s, produced by
   `export_yolov8n_int8.py` and `make_layout_header.py` and marked *do not edit*.
3. **Write the reference implementation in the language that makes it readable, then
   port it.** `yolo_flatten.run_numpy` and `export_minilm_int8.encode` exist so the
   C has something to be right *against*. Four independent implementations agreeing
   (README §10.1) is downstream of this choice.
4. **Every measurement ships with a control that can fail.** `--method pickbest` for
   the AP code, `--rank det` for the ranker, the `fpcheck` line for the build, the
   contiguity assert for the vocab, the FNV hash for the im2col reordering. The
   controls are as much a part of the deliverable as the numbers.
