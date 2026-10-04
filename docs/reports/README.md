# docs/reports/

Documents we wrote **during** the contest, kept as they were (bylines replaced by the team name; paths refer to our working folders at the time).

| File | Stage | What it is |
|---|---|---|
| [`stage1_background_research.md`](stage1_background_research.md) | 1 | Background research: VEGA AS1061, Genesys 2, COCO-Tasks, YOLOv8n, MobileCLIP |
| [`stage1_systolic_array_design_note.pdf`](stage1_systolic_array_design_note.pdf) | 1 | Matrix-multiplier design note: full systolic vs tiled, Cannon's algorithm |
| [`stage2a_report_text.md`](stage2a_report_text.md) | 2A | Text of our two-page Stage 2A report |
| [`stage3b_implementation_report.md`](stage3b_implementation_report.md) | 3B | **Stage 3B implementation report** (18 Aug 2026): pipeline, accelerator, standalone boundary, milestones, submission answers, performance, accuracy, limitations, why YOLOv8n, path to YOLOv8x |
| [`stage3b_quick_start.md`](stage3b_quick_start.md) | 3B | Step-by-step: program the FPGA, boot Linux, deploy, run, retrieve (no passwords) |
| [`ACCURACY_FINDINGS.html`](ACCURACY_FINDINGS.html) | finals prep | "Where the accuracy went" (23 Aug) — detector vs ranker loss, resolution sweep, confidence floor, array sizing. Some numbers were later superseded (see [`evaluation/README.md`](../../evaluation/README.md)) |
| [`FINAL_PLAN.html`](FINAL_PLAN.html) | finals prep | "The eight-day plan" (24 Aug): board access to 27 Aug, Bengaluru on 29 Aug, presentation 1 Sept; phases, gates, risk register, talk outline |

Open the `.html` files in a browser (download, then open) — GitHub shows them as source.

**Notes on the Stage 3B report.** Some timings in it (e.g. 538.8 s per query, 0.063 mAP at 128 px) are from 17–18 August and were superseded in the following week (see [`TECHNICAL_REPORT.md`](../TECHNICAL_REPORT.md)). It also mentions a 1 MB DDR scratch window; this was later enlarged to 2 MB for YOLOv8s.
