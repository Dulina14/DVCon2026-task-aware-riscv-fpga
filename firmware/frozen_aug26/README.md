# firmware/frozen_aug26/

The source of the frozen build (25–26 August 2026) that produced the 14-task board evidence and the binaries in `../prebuilt/`. `SHA256SUMS.txt` is the record we made at the time (it also lists the frozen YOLOv8n binary and header, which are not duplicated here).

The final `../src/team43_pipeline.c` (29 August) differs only by blocked im2col (`--im2col-block`, default 64), which produces bit-identical output.
