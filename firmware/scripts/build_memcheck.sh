#!/bin/sh

set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUTPUT=${1:-"$ROOT/diagnostics/team43_memcheck_rebuilt"}

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
    echo "ERROR: set RISCV_GCC to the RISC-V Buildroot compiler path." >&2
    exit 1
fi

"$CC" -std=gnu99 -O2 -static -Wall -Wextra \
    -o "$OUTPUT" "$ROOT/diagnostics/team43_memcheck.c" -lm -lrt

echo "Built: $OUTPUT"
sha256sum "$OUTPUT"
