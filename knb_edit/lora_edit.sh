#!/bin/bash
# -x paraai-n32-h-01-agent-[1-33],paraai-n32-h-01-agent-[48-56],paraai-n32-h-01-agent-[63-197]
# sbatch -x paraai-n32-h-01-agent-[1-33],paraai-n32-h-01-agent-[48-56],paraai-n32-h-01-agent-[63-197] --gpus=1 ./lora_edit.sh
DATE=$(date +%Y-%m-%d)
if [ ! -d "logs/$DATE" ]; then
    mkdir -p "logs/$DATE"
fi

module load compilers/cuda/12.1
module load cudnn/8.9.5.29_cuda12.x
module load compilers/gcc/12.2.0
source activate ke2torch23cu121

export CUDA_VISIBLE_DEVICES=0
export HUGGINGFACE_CACHE=/home/bingxing2/public/models/llama3/
# type ['orgin','abs','square']
# p  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# type=square
# p=0.6
# python lora_edit.py \
#     --type $type \
#     --p $p \
#     > logs/$DATE/llama3-zsre-326-$type-$p-down_proj-1.log 2>&1

type=abs
# for p in {0.2,0.1}
# for p in {0.4,0.3}
# for p in {0.6,0.5}
for p in {0.8,0.7}
do
    python lora_edit.py \
        --type $type \
        --p $p \
        > logs/$DATE/llama3-zsre-326-$type-$p-down_proj-2.log 2>&1
done