#!/bin/bash
for var in 1
do
    python3 Scripts/train.py \
    --device cuda:1 \
    --position_offset \
    --name "LAVA" \
    --lang_emb "CLIP" \
    --variations 1 \
    --val_variations "7 " \
    --log_to_wandb \
    --save_model \
    --train_iters 200 &\
    python3 Scripts/train.py \
    --device cuda:2 \
    --position_offset \
    --name LAVA \
    --variations 2 \
    --val_variations "5 11 14 17" \
    --cross_decode \
    --log_to_wandb \
    --save_model \
    --train_iters 200
done

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