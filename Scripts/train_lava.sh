#!/bin/bash
for var in 1
do
    python3 Scripts/train.py \
    --device cuda:1 \
    --position_offset \
    --name "LAVA" \
    --lang_emb "CLIP" \
    --variations $var \
    --val_variations $var \
    --log_to_wandb \
    --train_iters 400
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