#!/bin/sh
M=$HOME/mnt/2026-08-15
D=$HOME/mnt/HDL/final_phase/eval
cd $D
for cfg in "128 0.01 128c01 chunk_" "192 0.01 192c01 chunk_" "320 0.05 320c05 c320_"; do
  set -- $cfg; SZ=$1; CF=$2; TAG=$3; PFX=$4
  for c in ${PFX}*; do
    STAMP=".done_${TAG}_${c}"
    [ -f "$STAMP" ] && continue
    ./pipeline_x86 --sim --imgsz $SZ --conf $CF \
       --yolo $M/team43_yolo_int8.bin --minilm $M/team43_minilm_int8.bin \
       --vocab $M/vocab.txt --detect-list $c \
       --detect-json det7_$TAG.json --resume --quiet >/dev/null 2>&1 \
       && touch "$STAMP" && echo "ok $TAG $c" && exit 0
  done
done
echo ALLDONE
