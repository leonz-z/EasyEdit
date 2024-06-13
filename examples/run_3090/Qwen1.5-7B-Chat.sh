#!/bin/bash

DATE=2024-6-3

# Ensure log directory exists
if [ ! -d "examples/log" ]; then
    mkdir examples/log
fi
# if log/$DATE does not exist, create it
if [ ! -d "examples/log/$DATE" ]; then
    mkdir "examples/log/$DATE"
fi

# 3090
# export HUGGINGFACE_CACHE=/share/huggingface/ # 3090
# MODEL=Llama-2-7b-ms # 3090
# source activate EasyEdit
# A100
MODEL=Qwen1.5-7B-Chat
export HUGGINGFACE_CACHE=/home/bingxing2/public/models/Qwen/
module load compilers/cuda/11.8
module load cudnn/8.8.1.3_cuda11.x
module load compilers/gcc/12.2.0
source activate ke
# DATA_TEST=ZsRE-test-all
# DATA_TYPE=zsre
EDIT_METHOD=ROME
DATA_TEST=recent_test
DATA_TRAIN=recent_train
DATA_TYPE=recent
NUM=1

CUDA_VISIBLE_DEVICES=0 nohup python examples/run_knowedit_llama2.py \
    --editing_method $EDIT_METHOD \
    --hparams_dir hparams/$EDIT_METHOD/$MODEL \
    --data_dir dataset/ccks2024_know_edit/$DATA_TEST.json \
    --train_data_path dataset/ccks2024_know_edit/$DATA_TRAIN.json \
    --metrics_save_dir examples/output \
    --datatype $DATA_TYPE \
    > examples/log/$DATE/$EDIT_METHOD-$MODEL-$DATA_TEST-$NUM.log 2>&1 &
