#!/bin/bash

DATE=$(date +%Y-%m-%d)
if [ ! -d "log/$DATE" ]; then
    mkdir -p "log/$DATE"
fi

module load compilers/cuda/12.1
module load cudnn/8.9.5.29_cuda12.x
module load compilers/gcc/12.2.0
source activate ke2torch23cu121

export CUDA_VISIBLE_DEVICES=0
export HUGGINGFACE_CACHE=/home/bingxing2/public/models/llama3/


