#!/bin/bash


# 0:600 1:1000 2:1000
for var_setup in 2:1000
do
    var=$(echo $var_setup | cut -d ":" -f 1)
    train_num=$(echo $var_setup | cut -d ":" -f 2)
    let TRAIN=$train_num
    let GAP=50
    let VAL_NUM="$TRAIN / $GAP"

    python3 Scripts2/train.py \
    --log_to_wandb \
    --lr 0.0005 \
    --device cuda:3 \
    --name "LAVA" \
    --lang_emb "CLIP" \
    --variations $var \
    --val_variations $var \
    --train_iters $TRAIN \
    --val_number $VAL_NUM 
done
# --log_to_wandb \

# --load_model \
# --load_name "train_1_variation_7.pth" \
# --log_to_wandb

# for x in 0:600 1:100
# do
#     first=$(echo $x | cut -d ":" -f 1)
#     second=$(echo $x | cut -d ":" -f 2)
#     echo $first $second
# done