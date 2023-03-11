#!/bin/bash
for var in 1
do
    python3 Scripts/train.py \
    --device cuda:2 \
    --position_offset \
    --name "VALA" \
    --lang_emb "CLIP" \
    --variations $var \
    --val_variations $var \
    --val_number 40 \
    --train_iters 400 
done
