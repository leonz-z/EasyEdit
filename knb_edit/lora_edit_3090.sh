#!/bin/bash
DATE=$(date +%Y-%m-%d)
if [ ! -d "logs/$DATE" ]; then
    mkdir -p "logs/$DATE"
fi

source activate ke2torch23cu121
export CUDA_VISIBLE_DEVICES=0
export HUGGINGFACE_CACHE=/share/huggingface/
type=orgin
p=99.7
batch_size=2
num_steps=50
nohup python lora_edit.py \
    --type $type \
    --p $p \
    --batch_size 2 \
    --num_steps 50 \
    > logs/$DATE/llama3-zsre-326-$type-$p-$batch_size-$num_steps-down_proj-1.log 2>&1 &

# type=orgin
# for p in {0.2,0.1}
# for p in {0.4,0.3}
# for p in {0.6,0.5}
# for p in {0.8,0.7}
# do
#     python lora_edit.py \
#         --type $type \
#         --p $p \
#         > logs/$DATE/llama3-zsre-326-$type-$p-down_proj-2.log 2>&1
# done