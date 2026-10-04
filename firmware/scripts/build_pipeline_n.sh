#!/bin/sh

# Rebuild into a new file so the validated binary remains untouched.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUTPUT=${1:-"$ROOT/binaries/team43_pipeline_n_rebuilt"}

if [ -n "${RISCV_GCC:-}" ]; then
    CC=$RISCV_GCC
else
    CC=""
    for candidate in \
        "$ROOT/../buildroot_dvcon/buildroot/output/host/bin/riscv64-buildroot-linux-gnu-gcc" \
        "$ROOT/../../buildroot_dvcon/buildroot/output/host/bin/riscv64-buildroot-linux-gnu-gcc" \
        "$PWD/buildroot_dvcon/buildroot/output/host/bin/riscv64-buildroot-linux-gnu-gcc"
    do
        if [ -x "$candidate" ]; then
            CC=$candidate
            break
        fi
    done
fi

if [ -z "$CC" ] || [ ! -x "$CC" ]; then
    echo "ERROR: RISC-V Buildroot compiler was not found." >&2
    echo "Set RISCV_GCC to its complete path, for example:" >&2
    echo "  RISCV_GCC=/path/to/riscv64-buildroot-linux-gnu-gcc ./scripts/build_pipeline_n.sh" >&2
    exit 1
fi

"$CC" \
    -std=gnu99 -O2 -ffp-contract=off -static -Wall -Wextra \
    -I"$ROOT/headers" \
    -o "$OUTPUT" \
    "$ROOT/source/team43_pipeline.c" \
    -lm -lrt

echo "Built: $OUTPUT"
sha256sum "$OUTPUT"
