DATE=$(date +"%Y-%m-%d")
mkdir -p logs/$DATE
source activate ke2torch23cu121
export HUGGINGFACE_CACHE=/share/huggingface/
export PYTHONUNBUFFERED=1
# export CUDA_VISIBLE_DEVICES=1

# KNB
method=KNB
num_steps=120
type=max 
cnt=1

i=0
data_type=type1_133

max_new_tokens_times=4
ff_attrs=mlp.c_proj
p=99.05
t_loss=1e-3
batch_size=10
for skip in $(seq 0 10 120); do
    start_idx_end_idx=$skip,$((skip+10))
    CUDA_VISIBLE_DEVICES=$i python examples/run_CKnowEdit_qwen-1.8B.py \
        --target_modules $ff_attrs \
        --t_loss $t_loss \
        --batch_size $batch_size \
        --num_steps $num_steps \
        --editing_method $method \
        --p $p \
        --data_type CKnowEdit_$data_type \
        --metrics_save_dir ./ccks2024_output/$data_type \
        --data_dir ./dataset/ccks2024_know_edit/CKnowEditType/$data_type.json \
        --start_idx_end_idx $start_idx_end_idx \
        --knb_dict_path /share/knb_dict/KNB-Qwen-1_8B-Chat-CKnowEdit/${data_type}_v4/$data_type-Qwen-1_8B-Chat-CKnowEdit-layer-0-24-$ff_attrs-knb_dict-$type-p$p.json \
        --is_post_metrics \
        --max_new_tokens_times $max_new_tokens_times \
        > logs/$DATE/$i-token$max_new_tokens_times-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-Qwen-1_8B-Chat-$cnt.log 2>&1 &
    wait
done

start_idx_end_idx=130,133
CUDA_VISIBLE_DEVICES=$i python examples/run_CKnowEdit_qwen-1.8B.py \
    --target_modules $ff_attrs \
    --t_loss $t_loss \
    --batch_size $batch_size \
    --num_steps $num_steps \
    --editing_method $method \
    --p $p \
    --data_type CKnowEdit_$data_type \
    --metrics_save_dir ./ccks2024_output/$data_type \
    --data_dir ./dataset/ccks2024_know_edit/CKnowEditType/$data_type.json \
    --start_idx_end_idx $start_idx_end_idx \
    --knb_dict_path /share/knb_dict/KNB-Qwen-1_8B-Chat-CKnowEdit/${data_type}_v4/$data_type-Qwen-1_8B-Chat-CKnowEdit-layer-0-24-$ff_attrs-knb_dict-$type-p$p.json \
    --is_post_metrics \
    --max_new_tokens_times $max_new_tokens_times \
    > logs/$DATE/$i-token$max_new_tokens_times-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-Qwen-1_8B-Chat-$cnt.log 2>&1 &
