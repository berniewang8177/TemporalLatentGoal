#!/bin/bash

TASK="push_buttons" 
# test_dataset="test_datasets"
test_dataset="new_val_data"

VAL_VAR="10 "
train_dataset="new_train_data"
TRAIN_VAR="0 1 2 4"

# test data pre-process

DATA_PATH="/home/yiqiw2/experiment/language_rl/$test_dataset/instructions/$TASK/variation"
SAVE_PATH="/home/yiqiw2/experiment/language_rl/$test_dataset/$TASK+"
python3 Preprocess/lang_feature_clip.py --data_path $DATA_PATH --save_path $SAVE_PATH --variations "10 "
python3 Preprocess/lang_feature_w2v.py --data_path $DATA_PATH --save_path $SAVE_PATH --variations "10 " 


# # training data pre-process
DATA_PATH="/home/yiqiw2/experiment/language_rl/$train_dataset/instructions/$TASK/variation"
SAVE_PATH="/home/yiqiw2/experiment/language_rl/$train_dataset/$TASK+"
python3 Preprocess/lang_feature_clip.py --data_path $DATA_PATH --save_path $SAVE_PATH --variations "10 "
python3 Preprocess/lang_feature_w2v.py --data_path $DATA_PATH --save_path $SAVE_PATH --variations "10 "

