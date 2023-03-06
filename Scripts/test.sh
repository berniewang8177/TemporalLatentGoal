#!/bin/bash

xvfb-run python3 Scripts/test.py \
--device cuda:0 \
--position_offset \
--name "VALA" \
--lang_emb "CLIP" \
--var_num 1 \
--load_model \
--load_name "VALA_train_1_variation_7_film_once.pth"
