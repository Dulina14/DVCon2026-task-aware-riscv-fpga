#!/bin/sh
M=$HOME/mnt/2026-08-15
D=$HOME/mnt/HDL/final_phase/eval
cd $D
START=$(date +%s)
for cfg in "tab " "exa --exact-float"; do
  set -- $cfg; TAG=$1; shift; FLAG="$*"
  for c in c320_*; do
    STAMP=".d2_${TAG}_${c}"
    [ -f "$STAMP" ] && continue
    ./pipeline_new --sim --imgsz 192 --conf 0.05 $FLAG \
       --yolo $M/team43_yolo_int8.bin --minilm $M/team43_minilm_int8.bin \
       --vocab $M/vocab.txt --detect-list $c \
       --detect-json d2_$TAG.json --resume --quiet >/dev/null 2>&1 \
       && touch "$STAMP" && echo "ok $TAG $c"
    NOW=$(date +%s); [ $((NOW-START)) -ge 25 ] && exit 0
  done
done
echo ALLDONE
