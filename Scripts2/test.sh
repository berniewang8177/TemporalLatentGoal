#!/bin/bash

VARIATION=2

# xvfb-run python3 Scripts2/test.py \
# --device cuda:0 \
# --name "LAVA" \
# --lang_emb "CLIP" \
# --var_num $VARIATION \
# --load_model \
# --load_name "CLIP_LAVA_train_2_variation_2_no_film.pth"

xvfb-run python3 Scripts2/test.py \
--device cuda:0 \
--name "VALA" \
--lang_emb "W2V" \
--var_num $VARIATION \
--load_model \
--load_name "W2V_VALA_train_2_variation_2_no_film.pth"
