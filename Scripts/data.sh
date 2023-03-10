#!/bin/bash

TASK="push_buttons" 
test_dataset="test_datasets/"
TEST_VAR="5 7 11 14 17 "
train_dataset="train_datasets"
TRAIN_VAR="0 1 2 "

# test data pre-process

DATA_PATH="/home/yiqiw2/experiment/language_rl/$test_dataset/instructions/$TASK/variation"
SAVE_PATH="/home/yiqiw2/experiment/language_rl/$test_dataset/$TASK+"
# python3 Preprocess/lang_feature_clip.py --data_path $DATA_PATH --save_path $SAVE_PATH
python3 Preprocess/lang_feature_w2v.py --data_path $DATA_PATH --save_path $SAVE_PATH --variations "5 7 11 14 17 " 


# training data pre-process
DATA_PATH="/home/yiqiw2/experiment/language_rl/$train_dataset/instructions/$TASK/variation$var"
SAVE_PATH="/home/yiqiw2/experiment/language_rl/$train_dataset/$TASK+$var"
# python3 Preprocess/lang_feature_clip.py --data_path $DATA_PATH --save_path $SAVE_PATH
python3 Preprocess/lang_feature_w2v.py --data_path $DATA_PATH --save_path $SAVE_PATH --variations "0 1 2 "

