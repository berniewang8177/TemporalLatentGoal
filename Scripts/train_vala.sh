#!/bin/bash
# for var in 1
# do
#     for expert in 1
#     do
#         python3 Scripts/train.py \
#         --device cuda:1 \
#         --position_offset \
#         --name "VALA" \
#         --lang_emb "W2V" \
#         --variations "$var " \
#         --val_variations "7 " \
#         --expert_counts $expert \
#         --log_to_wandb \
#         --save_model \
#         --train_iters 200 &\
        # python3 Scripts/train.py \
        # --device cuda:2 \
        # --position_offset \
        # --name "VALA" \
        # --lang_emb "W2V" \
        # --variations "$var " \
        # --val_variations "5 11 14 17 " \
        # --cross_decode \
        # --expert_counts $expert\
        # --log_to_wandb \
        # --train_iters 200
    # done
# done

# python3 Scripts/train.py \
# --device cuda:1 \
# --position_offset \
# --name "VALA" \
# --lang_emb "CLIP" \
# --variations "1 " \
# --val_variations "7 " \
# --log_to_wandb \
# --save_model \
# --train_iters 200 &\
# python3 Scripts/train.py \
# --device cuda:3 \
# --position_offset \
# --name "VALA" \
# --lang_emb "CLIP" \
# --variations "2 " \
# --val_variations "5 11 14 17 " \
# --log_to_wandb \
# --save_model \
# --train_iters 200

# python3 Scripts/train.py \
# --device cuda:1 \
# --position_offset \
# --name "VALA" \
# --lang_emb "W2V" \
# --variations "1 " \
# --val_variations "7 " \
# --log_to_wandb \
# --save_model \
# --train_iters 200 &\
# python3 Scripts/train.py \
# --device cuda:3 \
# --position_offset \
# --name "VALA" \
# --lang_emb "W2V" \
# --variations "2 " \
# --val_variations "5 11 14 17 " \
# --log_to_wandb \
# --save_model \
# --train_iters 200

python3 Scripts/train.py \
--device cuda:1 \
--name "VALA" \
--lang_emb "CLIP" \
--variations "1 " \
--position_offset \
--train_iters 100 \
--val_variations "7 " \
--save_model \