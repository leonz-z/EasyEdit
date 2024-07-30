#!/bin/bash
DATE=$(date +%Y-%m-%d)
if [ ! -d "logs/$DATE" ]; then
    mkdir -p "logs/$DATE"
fi

source activate ke2torch23cu121
# export CUDA_VISIBLE_DEVICES=0
export HUGGINGFACE_CACHE=/share/huggingface/

type=orgin
ds_size=326
batch_size=2
num_steps=50

i=0
for p in {99.2,99.4,99.6,99.8}; do
    echo "Running $i-th job for p=$p"
    CUDA_VISIBLE_DEVICES=$i python lora_edit_3090.py \
    --type $type \
    --p $p \
    --batch_size $batch_size \
    --num_steps $num_steps \
    --ds_size $ds_size \
    > logs/$DATE/$ds_size-llama3-zsre-$type-$p-$batch_size-$num_steps-down-1.log 2>&1 &
    i=$((i+1))
done
wait
# p=60
# CUDA_VISIBLE_DEVICES=2 python lora_edit.py \
#     --type $type \
#     --p $p \
#     --batch_size $batch_size \
#     --num_steps $num_steps \
#     --ds_size $ds_size \
#     > logs/$DATE/$ds_size-llama3-zsre-$type-$p-$batch_size-$num_steps-down_proj-1.log 2>&1 &