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

source activate ke
# 3090
export HUGGINGFACE_CACHE=/share/huggingface/ 
export CUDA_VISIBLE_DEVICES=3
MODEL=Qwen-7B-Chat 

# A100
# MODEL=Llama-2-7b-chat-hf 
# export HUGGINGFACE_CACHE=/home/bingxing2/public/models/llama2/
# module load compilers/cuda/11.8
# module load cudnn/8.8.1.3_cuda11.x
# module load compilers/gcc/12.2.0
# source activate ke
# DATA_TEST=ZsRE-test-all
# DATA_TYPE=zsre
EDIT_METHOD=ROME
DATA_TEST=recent_test
DATA_TRAIN=recent_train
DATA_TYPE=recent
NUM=1

nohup python examples/run_knowedit_llama2.py \
    --editing_method $EDIT_METHOD \
    --hparams_dir hparams/$EDIT_METHOD/$MODEL \
    --data_dir LLMKnowledgeEditDataset/ccks2024_know_edit/$DATA_TEST.json \
    --train_data_path LLMKnowledgeEditDataset/ccks2024_know_edit/$DATA_TRAIN.json \
    --metrics_save_dir examples/output \
    --datatype $DATA_TYPE \
    > examples/log/$DATE/$EDIT_METHOD-$MODEL-$DATA_TEST-$NUM.log 2>&1 &
