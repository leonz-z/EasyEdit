#!/bin/bash

DATE=$(date +%Y-%m-%d)

# Ensure log directory exists
if [ ! -d "examples/log" ]; then
    mkdir examples/log
fi
# if log/$DATE does not exist, create it
if [ ! -d "examples/log/$DATE" ]; then
    mkdir "examples/log/$DATE"
fi

source activate ke2torch23cu121
# 3090
export HUGGINGFACE_CACHE=/share/huggingface/ 
export CUDA_VISIBLE_DEVICES=1
MODEL=gpt-j-6b
hparams=$MODEL-0-5

DATA_TEST=ZsRE-test-all
DATA_TYPE=zsre
EDIT_METHOD=MEMIT
# DATA_TEST=recent_test
# DATA_TRAIN=recent_train
# DATA_TYPE=recent
NUM=1

nohup python examples/run_knowedit_llama2.py \
    --editing_method $EDIT_METHOD \
    --hparams_dir hparams/$EDIT_METHOD/$hparams \
    --data_dir dataset/ccks2024_know_edit/$DATA_TEST.json \
    --metrics_save_dir examples/output \
    --datatype $DATA_TYPE \
    > examples/log/$DATE/$EDIT_METHOD-$hparams-$DATA_TEST-$NUM.log 2>&1 &
    # --train_data_path dataset/ccks2024_know_edit/$DATA_TRAIN.json \
