#!/bin/bash

xvfb-run python3 Scripts/test.py \
--device cuda:0 \
--film_first \
--position_offset \
--name "VALA" \
--lang_emb "CLIP" \
--var_num 2 \
--load_model \
--load_name "VALA_train_2_variation_2_film_once.pth"
