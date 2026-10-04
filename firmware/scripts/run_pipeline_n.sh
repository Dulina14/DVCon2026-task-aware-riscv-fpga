#!/bin/sh

set -eu

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 IMAGE TASK [IMGSZ] [OUTPUT.ppm]" >&2
    echo "Example: $0 image.ppm \"serve wine\" 256 result.ppm" >&2
    exit 2
fi

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
IMAGE=$1
TASK=$2
IMGSZ=${3:-256}
OUTPUT=${4:-"$ROOT/result.ppm"}

chmod 755 "$ROOT/binaries/team43_pipeline_n" 2>/dev/null || true

"$ROOT/binaries/team43_pipeline_n" \
    --image "$IMAGE" \
    --task "$TASK" \
    --imgsz "$IMGSZ" \
    --rect \
    --yolo "$ROOT/model_packages/team43_yolo_int8.bin" \
    --minilm "$ROOT/model_packages/team43_minilm_int8.bin" \
    --vocab "$ROOT/model_packages/vocab.txt" \
    --packed "$ROOT/model_packages/team43_minilm_packed.bin" \
    --out "$OUTPUT"
