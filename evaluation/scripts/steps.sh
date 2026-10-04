#!/bin/sh
M=$HOME/mnt/2026-08-15
Y=$HOME/mnt/HDL/final_phase/yolov8s/team43_yolo_s_int8.bin
D=$HOME/mnt/HDL/final_phase/eval_s
cd $D
one() {
  ./pipeline_s --sim --imgsz 192 --conf 0.05 \
     --yolo $Y --minilm $M/team43_minilm_int8.bin --vocab $M/vocab.txt \
     --detect-list $1 --detect-json ds_$1.json --quiet >/dev/null 2>&1 && touch ".ds_$1"
}
n=0
for c in p20_*; do
  [ -f ".ds_$c" ] && continue
  one $c & n=$((n+1))
  [ $n -ge 2 ] && break
done
wait
ls .ds_p20_* 2>/dev/null | wc -l
