#!/bin/bash
# "0:0:400" "1:1 7:1000" "2:2 5:1000"
for var_setup in "1:1:1000"
do
    var=$(echo $var_setup | cut -d ":" -f 1)
    var_val=$(echo $var_setup | cut -d ":" -f 2) 
    train_num=$(echo $var_setup | cut -d ":" -f 3)

    let TRAIN=$train_num
    let GAP=50
    let VAL_NUM="$TRAIN / $GAP"

    python3 Scripts2/train.py \
    --log_to_wandb \
    --oracle_goal \
    --lr 0.0005 \
    --device cuda:2 \
    --name "VALA" \
    --lang_emb "W2V" \
    --variations $var \
    --val_variations "$var_val" \
    --train_iters $TRAIN \
    --save_model \
    --val_number $VAL_NUM 
done

# --log_to_wandb \