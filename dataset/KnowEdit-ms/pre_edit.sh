#!/bin/bash

DATE=$(date +%Y-%m-%d)
if [ ! -d "log/$DATE" ]; then
    mkdir -p "log/$DATE"
fi

# A100
# module load compilers/cuda/11.8
# module load cudnn/8.8.1.3_cuda11.x
# module load compilers/gcc/12.2.0
# source activate ke
module load compilers/cuda/12.1
module load cudnn/8.9.5.29_cuda12.x
module load compilers/gcc/12.2.0
source activate ke2torch23cu121


export CUDA_VISIBLE_DEVICES=1
# export HUGGINGFACE_CACHE=/home/bingxing2/public/models/llama3/ 
# model_name=Meta-Llama-3-8B-Instruct

model_name=gpt-j-6b
export HUGGINGFACE_CACHE=/home/bingxing2/home/scx7avs/lyc/huggingface/

data_path=../ccks2024_know_edit/ZsRE-test-all.json
dataset=zsre
prefix_prompt=3

python pre_edit.py \
    --model_name $model_name \
    --dataset $dataset \
    --data_path $data_path \
    --prefix_prompt $prefix_prompt \
    >log/$DATE/pre_edit-$model_name-$dataset-$prefix_prompt.log 2>&1
