#!/bin/bash
# "0:0:400" "1:1 7:1000" "2:2 5:1000"
WARM=3
for var_setup in "1 10:1 4 10:2000"
do
    var=$(echo $var_setup | cut -d ":" -f 1)
    var_val=$(echo $var_setup | cut -d ":" -f 2) 
    train_num=$(echo $var_setup | cut -d ":" -f 3)
    
    let TRAIN=$train_num
    let GAP=50
    let VAL_NUM="$TRAIN / $GAP"
    let WARMUP="$TRAIN / $WARM"

    python3 Scripts2/train.py \
    --accumulate_grad_batches 2\
    --lr 0.0001 \
    --device cuda:1 \
    --name "VALA" \
    --lang_emb "W2V" \
    --variations "$var" \
    --val_variations "$var_val" \
    --train_iters $TRAIN \
    --val_number $VAL_NUM \
    --warmup "$WARMUP"
done

# --log_to_wandb \