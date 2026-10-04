#!/bin/sh
M=$HOME/mnt/2026-08-15
D=$HOME/mnt/HDL/final_phase/eval
cd $D
run() {  # size conf tag
  [ -s "det_int8_$3.json" ] && { echo "skip $3"; return; }
  echo "=== $3 start $(date)"
  ./pipeline_x86 --sim --imgsz $1 --conf $2 \
     --yolo $M/team43_yolo_int8.bin --minilm $M/team43_minilm_int8.bin \
     --vocab $M/vocab.txt --detect-list images_200.txt \
     --detect-json det_int8_$3.json --resume --quiet 2>&1 | tail -3
  echo "=== $3 done $(date)  $(wc -l < det_int8_$3.json) lines"
}
run 192 0.05 192c05
run 320 0.05 320c05
run 128 0.01 128c01
run 192 0.01 192c01
run 256 0.05 256c05
echo ALLDONE
