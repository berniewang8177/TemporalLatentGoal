#!/bin/bash
# 0:400 1:1000 2:1000
for var_setup in 0:400
do
    var=$(echo $var_setup | cut -d ":" -f 1)
    train_num=$(echo $var_setup | cut -d ":" -f 2)
    let TRAIN=$train_num
    let GAP=50
    let VAL_NUM="$TRAIN / $GAP"

    python3 Scripts2/train.py \
    --log_to_wandb \
    --save_model \
    --lr 0.0005 \
    --device cuda:1 \
    --name "VALA" \
    --lang_emb "W2V" \
    --variations $var \
    --val_variations $var \
    --train_iters $TRAIN \
    --val_number $VAL_NUM 
done

# --log_to_wandb \