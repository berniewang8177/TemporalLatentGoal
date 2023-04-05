#!/bin/bash

# VARIATION=4

xvfb-run python3 Scripts2/test.py \
--device cuda:0 \
--oracle_goal \
--name "VALA" \
--lang_emb "W2V" \
--var_num "10 " \
--load_model \
--load_name "W2V_VALA_train_['4', '7']_variation_10_film.pth"

# xvfb-run python3 Scripts2/test_lowdim.py \
# --device cuda:0 \
# --name "VALA" \
# --lang_emb "W2V" \
# --ref_variations "7 4 " \
# --var_num "10 " \
# --failed_demo 10 \
# --load_model \
# --load_name "NearestNeighbor"
