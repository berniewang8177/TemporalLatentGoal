#!/bin/bash
let TRAIN=1000
let GAP=50
let VAL_NUM="$TRAIN / $GAP"

for var in 2
do
    python3 Scripts/train.py \
    --log_to_wandb \
    --device cuda:2 \
    --name "LAVA" \
    --lang_emb "CLIP" \
    --variations $var \
    --val_variations $var \
    --train_iters $TRAIN \
    --val_number $VAL_NUM \
    --save_model \
    --film_first
done
#     --log_to_wandb \
# python3 Scripts/train.py \
# --device cuda:1 \
# --name "LAVA" \
# --lang_emb "CLIP" \
# --variations "1 " \
# --position_offset \
# --train_iters 100 \
# --val_variations "7 " \
# --save_model \


# --load_model \
# --load_name "train_1_variation_7.pth" \
# --log_to_wandb