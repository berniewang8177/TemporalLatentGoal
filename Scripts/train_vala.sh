#!/bin/bash
let TRAIN=1000
let GAP=50
let VAL_NUM="$TRAIN / $GAP"

for var in 2
do
    python3 Scripts/train.py \
    --log_to_wandb \
    --device cuda:1 \
    --name "VALA" \
    --lang_emb "CLIP" \
    --variations $var \
    --val_variations $var \
    --train_iters $TRAIN \
    --val_number $VAL_NUM \
    --save_model \
    --film_first
done  