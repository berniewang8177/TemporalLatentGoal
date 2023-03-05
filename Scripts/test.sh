#!/bin/bash

python3 Scripts/test.py \
--device cuda:1 \
--position_offset \
--name "VALA" \
--lang_emb "CLIP" \
--var_num 7 \
--load_model \
--load_name "VALA_train_1_variation_7_film_once.pth"
