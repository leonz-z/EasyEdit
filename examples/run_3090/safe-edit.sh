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
export CUDA_VISIBLE_DEVICES=3
MODEL=Llama-2-7b-ms

# A100
# MODEL=Llama-2-7b-chat-hf 
# export HUGGINGFACE_CACHE=/home/bingxing2/public/models/llama2/
# module load compilers/cuda/11.8
# module load cudnn/8.8.1.3_cuda11.x
# module load compilers/gcc/12.2.0
# source activate ke
EDIT_METHOD=DINM
DATA=SafeEdit_test
# DATA_TEST=recent_test
# DATA_TRAIN=recent_train
# DATA_TYPE=recent
NUM=1

nohup python examples/run_ccks_SafeEdit_gpt2-xl.py \
    --editing_method $EDIT_METHOD \
    --edited_llm $MODEL \
    --hparams_dir hparams/$EDIT_METHOD/$MODEL.yaml \
    --safety_classifier_dir /share/huggingface/DINM-Safety-Classifier \
    --data_dir dataset/ccks2024_know_edit/$DATA.json \
    --metrics_save_dir examples/safety_results_test \
    > examples/log/$DATE/$EDIT_METHOD-$MODEL-$DATA-$NUM.log 2>&1 &