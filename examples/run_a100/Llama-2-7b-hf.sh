#!/bin/bash

DATE=$(date +%Y-%m-%d)

# if log/$DATE does not exist, create it
if [ ! -d "examples/log/$DATE" ]; then
    mkdir -p "examples/log/$DATE"
fi

# A100
MODEL=Llama-2-7b-ms
# export HUGGINGFACE_CACHE=/home/bingxing2/public/models/llama2/ 
export HUGGINGFACE_CACHE=/home/bingxing2/home/scx7avs/lyc/huggingface/
export CUDA_VISIBLE_DEVICES=0
# module load compilers/cuda/11.8
# module load cudnn/8.8.1.3_cuda11.x
# module load compilers/gcc/12.2.0
# source activate ke
module load compilers/cuda/12.1
module load cudnn/8.9.5.29_cuda12.x
module load compilers/gcc/12.2.0
source activate ke2torch23cu121
# DATA_TEST=ZsRE-test-all
# DATA_TYPE=zsre
EDIT_METHOD=KN
DATA_TEST=recent_test
DATA_TRAIN=recent_train
DATA_TYPE=recent
NUM=1

 python examples/run_knowedit_llama2.py \
    --editing_method $EDIT_METHOD \
    --hparams_dir hparams/$EDIT_METHOD/$MODEL \
    --data_dir LLMKnowledgeEditDataset/ccks2024_know_edit/$DATA_TEST.json \
    --datatype $DATA_TYPE \
    --train_data_path LLMKnowledgeEditDataset/ccks2024_know_edit/$DATA_TRAIN.json \
    > examples/log/$DATE/$EDIT_METHOD-$MODEL-$DATA_TEST-$NUM.log 2>&1
