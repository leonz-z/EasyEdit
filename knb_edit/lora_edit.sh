#!/bin/bash
#SBATCH --gpus=4
#SBATCH -x paraai-n32-h-01-agent-[1-33],paraai-n32-h-01-agent-[48-56],paraai-n32-h-01-agent-[63-197]
export PYTHONUNBUFFERED=1
DATE=$(date +%Y-%m-%d)
if [ ! -d "logs/$DATE" ]; then
    mkdir -p "logs/$DATE"
fi

module load compilers/cuda/12.1
module load cudnn/8.9.5.29_cuda12.x
module load compilers/gcc/12.2.0
source activate ke2torch23cu121

# export CUDA_VISIBLE_DEVICES=0
export HUGGINGFACE_CACHE=/home/bingxing2/public/models/llama3/

type=orgin
batch_size=50
num_steps=100
i=0
for p in {91.0,95.0,99.1,99.9}; do
    echo "Running $i-th job for p=$p"
    CUDA_VISIBLE_DEVICES=$i python lora_edit.py \
    --type $type \
    --p $p \
    --batch_size $batch_size \
    --num_steps $num_steps \
    > logs/$DATE/all-llama3-zsre-$type-$p-$batch_size-$num_steps-down_proj-1.log 2>&1 &
    i=$((i+1))
done
wait