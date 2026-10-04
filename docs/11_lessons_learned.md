# 11 — Lessons learned (and advice for future teams)

This journey was about learning, facing challenges, and redoing the same things in different ways until we got them right. These are the lessons that cost us the most time — written so that the next team can learn them in minutes instead of days.

---

## Part A — Technical lessons

### 1. Measure the interface, not just the arithmetic

Our 8×8 array finishes one tile in **about 0.5 µs**. The measured wall-clock per output tile was **about 1,800 µs**. Each register access is a Linux system call costing 65–85 µs, and **cost is per call, not per byte** (a 1 KB write cost only about 4–7 times as much as a 4-byte write in our measurements).

So every real speed-up came from **removing calls**: uploading operands once (4.37×), letting one call accumulate a whole dot product (3.3×). A bigger array would have made almost no difference. *Before you widen your hardware, find out what it is waiting for.*

### 2. Read your own RTL like a stranger

We assumed one accelerator call handled 8 numbers of a dot product. Reading the RTL again showed the array ends an accumulation only when the DMA says "last" — and that length is just a register. **"K per call" was a software choice.** Changing it took a query from 713 s to 218 s with bit-identical output. The answer was in code we already had.

### 3. Beware "correct by accident"

Two bugs gave correct results for months:

- **`Y_DONE` was never cleared.** The "wait for done" loop was really one system call of delay. It worked only because calls were short. When calls got longer it gave wrong answers (max error 127,452 at K = 512).
- **FMA rounding.** RISC-V GCC fuses `a*b+c` into one instruction; x86 does not. Our PC "golden reference" quietly stopped matching the board in the last bits.

Neither would be found by comparing the system with itself. *Always compare against something independent*, and print a fingerprint (our `fpcheck` line) that proves both sides compute the same way. Use `-ffp-contract=off` on every build that must agree.

### 4. Build a software twin of your hardware

The board program's `--sim` mode emulates the accelerator exactly. Because of it, we debugged on laptops, used precious board slots only for real hardware questions, and could check every board result on a PC. It also let us run thousands of accuracy evaluations without the board.

### 5. One probe per question

Instead of running a 9-minute pipeline to test an idea, we wrote small programs that each answer one question in seconds: *Can one call accumulate all of K? Can several registers be written in one call? Is this board's memory healthy?* ([`firmware/probes/`](../firmware/probes/README.md)). One probe even gave a wrong answer (it used constant addresses) — the next, stricter probe caught it. **Make probes try hard to prove themselves wrong.**

### 6. Verify before silicon

Our accelerator reached the board **without a single RTL change**, because in Stage 3A it had already met the same CPU, bus and memory in a full-system simulation that checked itself (self-checking firmware + independent golden model + automatic waveform analysis). Time spent on verification was paid back many times.

### 7. Hash everything that crosses a boundary

- A ROM image that the tool silently regenerated would have made our Stage 3A design look broken on the judges' PC. A hash check caught it.
- Three of fourteen JPEGs arrived truncated at 94 KB/s. The pipeline ran happily on them. A `verify.sh` with checksums is now step one of every deployment.

### 8. Choose models for the platform, not for the leaderboard

YOLOv8x at 640 px would take about **25 hours per image** at our measured rate. YOLOv8n/YOLOv8s at 128–256 px take minutes. Our biggest accuracy jump came from a slightly bigger detector at a slightly bigger canvas (YOLOv8s @ 256 px), not from a cleverer ranker. Find where your errors live: in our system the **detector lost 72 points** of top-1 and the **ranker only 8**.

### 9. A text-only ranker is a hardware-friendly ranker

Because the ranker never sees pixels, the 80 class-name vectors can be computed once offline — legally and exactly. Only the task sentence (≤ 8 tokens) is encoded on the board. Look for parts of your model that do not depend on the run-time input.

### 10. Keep weak detections

A detector demo uses a 0.25 confidence floor; we used **0.02**. Every box you throw away is a candidate the ranker never sees. Multiplying detection confidence by suitability (the paper's rule) keeps weak boxes from winning unfairly.

### 11. Watch the hardware's mood

The same binary on the same board sometimes ran 4–20× slower after a fresh FPGA programming, with identical outputs — a DRAM/cache latency problem, not a logic bug. A 5-second memory check at the start of each session told us which state we were in. Small working sets (e.g. blocked im2col) make the program less sensitive to it.

## Part B — Competition lessons

1. **Read every e-mail from the organisers carefully** and keep a calendar of deadlines and slots. Deadlines moved several times.
2. **Upload limits are real.** EasyChair took one file of at most 100 MB. Have a Google Drive plan ready.
3. **Reproducibility is part of the score.** The judges ran our scripts on their machines. Assume a different OS, a different tool version and different paths.
4. **Do not modify organiser files** if you can avoid it — add wrappers and scripts beside them.
5. **Be honest in reports.** We listed failures (e.g. task 9, "open parcel": the board picks a fork, ground truth says knife) and things that were predicted but not yet measured. Clear limits made the rest more believable.
6. **Plan the demo around the slow parts.** A live query that takes minutes needs a story told while it runs, and a recorded result as back-up.
7. **Sort out travel early** if you might be invited abroad. Visas and flights at a week's notice are stressful and expensive.
8. **Ask for help.** Mentors and organisers were generous with time — at the Q&A sessions, by e-mail and during the hands-on days.

## Part C — "Five things that cost days to learn"

From our [technical report §17](TECHNICAL_REPORT.md#17-five-things-that-cost-days-to-learn):

1. The k0 loop does not have to exist — K per call is a software choice.
2. Multi-word register writes do not work — five single-word writes per tile is the floor.
3. `Y_DONE` must be cleared before every START.
4. Both builds must use `-ffp-contract=off`.
5. Cost is memory accesses, not arithmetic.
