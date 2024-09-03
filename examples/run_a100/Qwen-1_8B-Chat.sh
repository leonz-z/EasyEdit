#!/bin/bash
#SBATCH --gpus=1
#SBATCH -x paraai-n32-h-01-agent-[1-33],paraai-n32-h-01-agent-[48-56],paraai-n32-h-01-agent-[63-197]
export PYTHONUNBUFFERED=1

model=qwen1.8b.chat
DATE=$(date +%Y-%m-%d)_${model}
if [ ! -d "logs/$DATE" ]; then
    mkdir -p "logs/$DATE"
fi


module load compilers/cuda/12.1
module load cudnn/8.8.1.3_cuda12.x
module load compilers/gcc/11.3.0
source activate ke2torch23cu121
export HUGGINGFACE_CACHE=/home/bingxing2/home/scx7avs/lyc/huggingface/

method=FT
batch_size=10
num_steps=100
model_name=Qwen-1_8B-Chat
data_name=type7_277
layers=0,24
start_idx_end_idx=0,277
max_new_tokens_times=4
objective_optimization=target_new
python examples/run_CKnowEdit_qwen-1.8B.py \
    --objective_optimization $objective_optimization \
    --editing_method $method \
    --layers $layers \
    --start_idx_end_idx $start_idx_end_idx \
    --data_type CKnowEdit_$data_name \
    --metrics_save_dir ./ccks2024_output/$data_name \
    --data_dir ./dataset/ccks2024_know_edit/CKnowEditType/$data_name.json \
    --max_new_tokens_times $max_new_tokens_times \
    --is_post_metrics \
    --num_steps $num_steps \
    > logs/$DATE/$data_name-$objective_optimization-bs$batch_size-$num_steps-tt$max_new_tokens_times-$method-$layers-$start_idx_end_idx-Qwen-1_8B-Chat-$cnt.log 2>&1
    
