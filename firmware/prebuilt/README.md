# firmware/prebuilt/

Static 64-bit RISC-V Linux executables built on 26 August 2026 in the organisers' `buildroot-2004_new` container (`-std=gnu99 -O2 -ffp-contract=off -static`), from the source in [`../frozen_aug26/`](../frozen_aug26/).

| Binary | Detector header compiled in | Use with |
|---|---|---|
| `team43_pipeline_n` | YOLOv8n | `models/team43_yolo_int8.bin` |
| `team43_pipeline_s` | YOLOv8s | `models/team43_yolo_s_int8.bin` |

`team43_pipeline_s` produced the 14-task board evidence in [`results/`](../../results/README.md). Checksums: [`SHA256SUMS.txt`](SHA256SUMS.txt).

They run only on the VEGA Linux board (they open `/dev/accelerator_test` unless `--sim` is given).
