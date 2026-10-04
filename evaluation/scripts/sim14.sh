#!/bin/sh
M=$HOME/mnt/2026-08-15
I=$HOME/mnt/HDL/final_phase/phase3_images
D=$HOME/mnt/HDL/final_phase/eval
cd $D; mkdir -p sim14
r(){ ./pipeline_new --image $I/COCO_val2014_$2.jpg --task "$3" --sim --imgsz 320 \
      --yolo $M/team43_yolo_int8.bin --minilm $M/team43_minilm_int8.bin --vocab $M/vocab.txt \
      --out sim14/task$1.ppm 2>&1 | grep -E "SELECTED :" | sed "s/^/task $1  /"; }
r 1 000000262605 "step on something to reach top of a shelf"
r 2 000000393836 "sit comfortably"
r 3 000000132329 "place flowers"
r 4 000000394608 "get potatoes out of fire"
r 5 000000528200 "water plant"
r 6 000000132415 "get lemon out of tea"
r 7 000000265789 "dig hole"
r 8 000000001647 "open bottle of beer"
r 9 000000272880 "open parcel"
r 10 000000277015 "serve wine"
r 11 000000393682 "pour sugar"
r 12 000000402528 "smear butter"
r 13 000000012860 "extinguish fire"
r 14 000000526425 "pound carpet"
