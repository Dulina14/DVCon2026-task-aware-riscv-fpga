#!/bin/sh
# Both builds MUST use -ffp-contract=off. RISC-V contracts a*b+c into a single
# FMA and x86 does not; without this flag the board and the host simulation
# disagree in the last bits and the fpcheck line in the run header will differ.

# ---- board: RISC-V, static, inside the buildroot container -----------------
docker run --rm -t -v "$PWD":/workspace -w /workspace --user $(id -u):$(id -g) \
  buildroot-2004_new bash -c \
  'buildroot_dvcon/buildroot/output/host/bin/riscv64-buildroot-linux-gnu-gcc \
     -std=gnu99 -O2 -ffp-contract=off -static -Wall -Wextra \
     -o team43_pipeline team43_pipeline.c -lm -lrt'

# ---- host: x86, --sim emulates the accelerator in software ----------------
gcc -std=gnu99 -O2 -ffp-contract=off -Wall -Wextra \
    -o pipeline_x86 team43_pipeline.c -lm

# ---- probes (board only) --------------------------------------------------
# riscv64-...-gcc -std=gnu99 -O2 -static -o team43_kdepth    probes/team43_kdepth.c    -lrt
# riscv64-...-gcc -std=gnu99 -O2 -static -o team43_regstress probes/team43_regstress.c -lrt

# A build is correct when the host and the board print the same fpcheck line:
#   fpcheck: 34800000 3efe65ea be54d74a
