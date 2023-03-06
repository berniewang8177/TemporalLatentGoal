#!/bin/bash

xvfb-run python3 Scripts/test.py \
--device cuda:0 \
--position_offset \
--name "VALA" \
--lang_emb "CLIP" \
--var_num 0 \
--load_model \
--load_name "VALA_train_0_variation_0_cross_decode.pth"
