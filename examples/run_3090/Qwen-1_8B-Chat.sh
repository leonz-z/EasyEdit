DATE=$(date +"%Y-%m-%d")
mkdir -p logs/$DATE
source activate ke2torch23cu121
export HUGGINGFACE_CACHE=/share/huggingface/
export PYTHONUNBUFFERED=1
# export CUDA_VISIBLE_DEVICES=1

# MEMIT
method=MEMIT
start_idx_end_idx=0,133
data_type=type1_133
max_new_tokens_times=2

layers=0,12
cnt=1
gpu=0
CUDA_VISIBLE_DEVICES=$gpu python examples/run_CKnowEdit_qwen-1.8B.py \
    --editing_method $method \
    --layers $layers \
    --start_idx_end_idx $start_idx_end_idx \
    --data_type CKnowEdit_$data_type \
    --metrics_save_dir ./ccks2024_output/$data_type \
    --data_dir /share/dataset/CKnowEditType/$data_type.json \
    --max_new_tokens_times $max_new_tokens_times \
    > logs/$DATE/$gpu-$data_type-tt$max_new_tokens_times-$method-$layers-$start_idx_end_idx-Qwen-1_8B-Chat-$cnt.log 2>&1 &

layers=12,24
cnt=1
gpu=3
CUDA_VISIBLE_DEVICES=$gpu python examples/run_CKnowEdit_qwen-1.8B.py \
    --editing_method $method \
    --layers $layers \
    --start_idx_end_idx $start_idx_end_idx \
    --data_type CKnowEdit_$data_type \
    --metrics_save_dir ./ccks2024_output/$data_type \
    --data_dir /share/dataset/CKnowEditType/$data_type.json \
    --max_new_tokens_times $max_new_tokens_times \
    > logs/$DATE/$gpu-$data_type-tt$max_new_tokens_times-$method-$layers-$start_idx_end_idx-Qwen-1_8B-Chat-$cnt.log 2>&1 &
