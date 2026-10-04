# Team 43 firmware — a reading guide

> This is the short version: where to start and in what order. For the stage-by-stage
> walkthrough — what each function does, why it is written that way, and the measured
> numbers behind every default — see `../CODE_GUIDE_FIRMWARE.md`. The probes have
> their own guide in `../CODE_GUIDE_PROBES.md`.

Everything the board runs is one C file, `src/team43_pipeline.c` (1,955 lines).
It takes a JPEG and an English sentence and writes back the same picture with one
object outlined. No host does any part of the inference.

The other files are either generated (the model graphs) or single-purpose
(the JPEG decoder, the bitmap font). The probes in `probes/` are standalone
programs that answer one hardware question each; they are not part of the
pipeline but two of them produced findings the pipeline now depends on.

```
src/        team43_pipeline.c        the whole firmware
            jpeg.h                   baseline JPEG decoder, integer IDCT
            font8x8.h                8x8 bitmap font for the result label
generated/  team43_yolo_graph.h      YOLOv8s as a flat op list  (export tool output)
            team43_minilm_layout.h   MiniLM tensor offsets in the package
probes/     team43_kdepth.c          can one invocation accumulate all of K?
            team43_regstress.c       can several registers be set in one write?
            team43_regbatch.c        the first, inadequate version of the above
build.sh    the two build commands, board and host
```

## Read it in this order

**1. The accelerator interface — lines 63 to 330.**
Start here. Everything else is ordinary C; this is the part that talks to the
FPGA. `rw`/`rr` write and read one 32-bit register through `/dev/accelerator_test`
(counts are in *words*, not bytes — a common trip-up). `dwr`/`drd`/`dw_bulk` move
blocks to and from the DDR scratch windows. Then read `gemm_run` at line 254: it
is the heart of the system, the loop that turns a matrix multiply into a sequence
of accelerator invocations. Read `sim_tile` at 234 alongside it — that is the same
arithmetic in plain C, and the two must agree exactly.

**2. One convolution — `op_conv`, line 529.**
This is the pattern every layer follows: quantise the input tensor to INT8, gather
patches into `packX` (im2col), call `gemm_run`, then dequantise and apply the
activation. The four timers around it (`g_amax_us`, `g_qcvt_us`, `g_deq_us`,
`g_im2col_us`) exist because each of those phases turned out to cost more than
expected at least once.

**3. The rest of the graph — lines 667 to 800.**
`op_maxpool`, `op_upsample`, `op_concat`, `op_slice`, `op_add`, then
`decode_detect` at 750, which turns the network's raw output into boxes and runs
non-maximum suppression. `main` walks `YOP_TYPE[]` from the generated header and
dispatches to these.

**4. The language side — lines 1040 to 1320.**
`encode_task` (1228) is a six-layer transformer written out longhand: attention,
feed-forward, layer norm, mean pooling. `score_classes` (1296) fuses the task
embedding with a precomputed per-class embedding and runs a small head. Note that
the class side is a *lookup*, not a forward pass — only the prompt goes through
the transformer, which is why MiniLM costs ~17 s against detection's ~250 s.

**5. Everything else.**
`read_image` (1318) sniffs JPEG vs PPM. The drawing helpers (1342–1392) mark the
result. `main` (1462) is argument parsing and orchestration, plus the batch
detection mode used for accuracy evaluation.

## Data layouts worth memorising

**Accelerator tiles.** The array consumes one beat per k: 8 bytes of X (one value
per row) and 8 bytes of W (one per column). So within a 64-byte tile, byte index
is `kk*8 + rr` for X and `kk*8 + cc` for W, and results come back as
`cc*8 + rr`. Get this wrong and the output is plausible but silently transposed.

**Register map**, byte offsets into the device file:

| | | | |
|---|---|---|---|
| 0x00 START | 0x04 K done | 0x08 K addr | 0x0C K bytes |
| 0x10 K tuser | 0x14 X done | 0x18 X addr | 0x1C X bytes |
| 0x20 X tuser | 0x24 B done | 0x28 B addr | 0x2C B bytes |
| 0x30 B tuser | 0x34 Y done | 0x38 Y addr | 0x3C Y bytes |

**DDR windows.** K at `0xBF000000`, X at `0xBF200000`, Y at `0xBF400000`, 2 MB
each. They were 1 MB until YOLOv8s arrived with a 1,179,648-byte weight block;
`main` now asserts the fit at startup rather than letting it corrupt silently.

## Five things that cost days to learn

These are all commented in place, but they are the parts you would otherwise
"clean up" and break.

1. **The k0 loop does not have to exist.** The array ends an accumulation when the
   DMA asserts last, and the transfer length is just a register — so K per
   invocation is a software choice. Collapsing it cut 320 px from 713 s to 218 s.
   `probes/team43_kdepth.c` is the experiment that proved it.

2. **Multi-word register writes do not work.** One `pwrite` covering several
   register words sets at most the first. `probes/team43_regstress.c` proved it
   under a real GEMM pattern after a naive first probe said the opposite.
   Five single-word writes per tile is the floor.

3. **`Y_DONE` must be cleared before every START.** It used to be only read, so
   after the first tile the "poll" was really one syscall of delay — correct by
   accident. That accident fails once K is large: exact to K=256, wrong from
   K=512.

4. **Both builds must use `-ffp-contract=off`.** RISC-V fuses `a*b+c` into a
   single FMA and x86 does not, so the same source produces different numbers on
   the two. The `fpcheck` line in the header prints the raw bits of an
   FMA-sensitive expression; if host and board print the same line, they round
   identically.

5. **Cost is memory accesses, not arithmetic.** A tile's arithmetic is ~0.16 µs
   against ~485 µs per accelerator invocation, and ~1,800 µs of wall clock per
   *output tile* before the k0 loop was collapsed. This is why widening the array
   was analysed and rejected, and why every optimisation that helped removed
   *accesses* rather than *operations*.

## Building

Board (RISC-V, static) and host (x86, with `--sim` to emulate the accelerator)
must both be built with `-ffp-contract=off`. See `build.sh`. The folder builds
the shipping configuration as it stands — no header swapping needed.

`generated/team43_yolo_graph.h` is **YOLOv8s**, the detector we ship. YOLOv8n was
the earlier one; it is structurally identical (the same 103 ops and 63 weights,
because v8n and v8s share a depth multiplier of 0.33 and differ only in width)
and is kept with the fallback build in `app/TEAM43_FROZEN/yolov8n/`. Nothing in
the firmware knows which model it is holding — the graph header supplies every
shape, which is why swapping detectors needed no code change beyond sizing two
buffers from the graph instead of from a constant.

Run `./team43_pipeline --help` for the full flag list. The ones that matter:
`--imgsz` (detector canvas), `--conf` (detection floor, default 0.02),
`--rank-by` (default `product` = MiniLM × detection score), `--kdepth`
(1 restores the old one-tile-per-invocation path), `--sim` (no FPGA needed).
