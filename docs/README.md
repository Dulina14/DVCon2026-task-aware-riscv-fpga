# Documentation

Read in order for the full story, or jump to what you need.

## The story

| # | Document | In one line |
|---|---|---|
| 01 | [The challenge](01_the_challenge.md) | The contest, the problem, the reference paper, the platform and the rules |
| 02 | [Journey timeline](02_journey_timeline.md) | All stages with dates, from roll-out to the award |
| 03 | [Stage 1 — Proposal](03_stage1_proposal.md) | First idea, profiling, systolic-array study |
| 04 | [Stage 2A — Software pipeline](04_stage2a_software_pipeline.md) | YOLO + fine-tuned MiniLM in Python, Colab training, results |
| 05 | [Stage 2B — Accelerator](05_stage2b_accelerator.md) | AXI-stream systolic array and its standalone testbench |
| 06 | [Stage 3A — SoC integration](06_stage3a_soc_integration.md) | Inside the VEGA SoC: self-checking simulation and synthesis |
| 07 | [Stage 3B — Remote hardware](07_stage3b_remote_hardware.md) | AnyDesk remote lab, board bring-up, milestones A–D, optimisation week |
| 08 | [Finals in Bengaluru](08_finals_bengaluru.md) | Hands-on with physical boards, demo, talk, award, photos |

## The engineering

| # | Document | In one line |
|---|---|---|
| 09 | [System architecture](09_system_architecture.md) | What runs where, data path, register and memory maps |
| 10 | [Reproduce it](10_reproduce.md) | Six levels, from "PC only" to "real board" |
| 11 | [Lessons learned](11_lessons_learned.md) | What cost us days, advice for future teams |
| 12 | [References](12_references.md) | Papers, datasets, IP, tools |
| — | [Glossary](glossary.md) | Every technical term in plain language |
| — | [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) | Our complete finals technical report (measurements, verification, optimisation history, limitations) |

## Code guides (line-level)

| Guide | Covers |
|---|---|
| [CODE_GUIDE_RTL.md](code_guides/CODE_GUIDE_RTL.md) | Every RTL module, and the things in it that look like cruft and are not |
| [CODE_GUIDE_FIRMWARE.md](code_guides/CODE_GUIDE_FIRMWARE.md) | `team43_pipeline.c` stage by stage |
| [CODE_GUIDE_PROBES.md](code_guides/CODE_GUIDE_PROBES.md) | The hardware probes, including the one that gave the wrong answer |
| [CODE_GUIDE_TOOLCHAIN.md](code_guides/CODE_GUIDE_TOOLCHAIN.md) | Host export and evaluation scripts, and why each is legal under the standalone rule |

## Reports and notes written during the contest

See [reports/README.md](reports/README.md), [stage2b/README.md](stage2b/README.md) and [archive/stage3a_2x2/README.md](archive/stage3a_2x2/README.md).
