#!/bin/bash

# VARIATION=4

# xvfb-run python3 Scripts2/test.py \
# --device cuda:0 \
# --oracle_goal \
# --name "VALA" \
# --lang_emb "W2V" \
# --var_num "4 " \
# --load_model \
# --load_name "W2V_VALA_train_1_variation_4_film.pth"

xvfb-run python3 Scripts2/test_lowdim.py \
--device cuda:0 \
--oracle_goal \
--name "VALA" \
--lang_emb "W2V" \
--ref_variations "4 7" \
--var_num "10 " \
--failed_demo 10 \
--load_model \
--load_name "NearestNeighbor"
