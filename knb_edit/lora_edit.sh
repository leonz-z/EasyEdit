#!/bin/bash
#SBATCH --gpus=1
##SBATCH -x paraai-n32-h-01-agent-[1-33],paraai-n32-h-01-agent-[48-56],paraai-n32-h-01-agent-[63-197]
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
export HUGGINGFACE_CACHE=/home/bingxing2/public/models/llama2/

type=max
ds_size=all
data_type=counterfact
# max_99.85_60_50_no_prompts	0.999360	479.426572	0.502695	0.711985
batch_size=60
# batch_size=70
# batch_size=60
# batch_size=50
# batch_size=40
# batch_size=30
# batch_size=20
# batch_size=10
num_steps=50
model_name=Llama-2-7b-hf
i=0
p=99.85
echo "$i batch_size=$batch_size num_steps=$num_steps p=$p"
CUDA_VISIBLE_DEVICES=$i python lora_edit.py \
    --type $type \
    --p $p \
    --batch_size $batch_size \
    --num_steps $num_steps \
    --ds_size $ds_size \
    --data_type $data_type \
    --model_name $model_name \
    --data_dir ../dataset/KnowEdit-ms/benchmark_wiki_counterfact_test_cf.json \
    --no_prompts \
    > logs/$DATE/$ds_size-$model_name-$data_type-$type-$p-$batch_size-$num_steps-down-no-prompts-1.log 2>&1 &

# for p in {99.2,99.3,99.4,99.5}; do
# for p in {99.6,99.7,99.8,99.9}; do
# for p in {99,99.1,99.05,99.15}; do
# for p in {99.25,99.35,99.45,99.55}; do
# for p in {99.65,99.75,99.85,99.95}; do

# for p in {99,99.05,99.1,99.15}; do
# for p in {99.2,99.25,99.3,99.35}; do
# for p in {99.4,99.45,99.5,99.55}; do
# for p in {99.6,99.65,99.7,99.75}; do
# for p in {99.8,99.85,99.9,99.95}; do
#     echo "$i batch_size=$batch_size num_steps=$num_steps p=$p"
#     CUDA_VISIBLE_DEVICES=$i python lora_edit.py \
#     --type $type \
#     --p $p \
#     --batch_size $batch_size \
#     --num_steps $num_steps \
#     --ds_size $ds_size \
#     --data_type $data_type \
#     --model_name $model_name \
#     --data_dir ../dataset/KnowEdit-ms/benchmark_wiki_counterfact_test_cf.json \
#     --no_prompts \
#     > logs/$DATE/$ds_size-$model_name-$data_type-$type-$p-$batch_size-$num_steps-down-no-prompts-1.log 2>&1 &
#     i=$((i+1))
# done
# wait