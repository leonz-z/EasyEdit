DATE=$(date +"%Y-%m-%d")
mkdir -p logs/$DATE
source activate ke2torch23cu121
export HUGGINGFACE_CACHE=/share/huggingface/
export PYTHONUNBUFFERED=1
# export CUDA_VISIBLE_DEVICES=1

# KNB
method=KNB
num_steps=120
max_new_tokens_times=2
cnt=1

i=2
type=mean
data_type=type5_70
t_loss=1e-3
ff_attrs=mlp.w1
knb_type=$ff_attrs
for batch_size in {1,10}; do
    for p in {90,91,92,93,94,95,96,97,98,99}; do
        for skip in $(seq 0 10 60); do
            start_idx_end_idx=$skip,$((skip+10))
            echo $i-token$max_new_tokens_times-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-Qwen-1_8B-Chat-$cnt
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
                --knb_dict_path /share/knb_dict/KNB-Qwen-1_8B-Chat-CKnowEdit/${data_type}_v4/$data_type-Qwen-1_8B-Chat-CKnowEdit-layer-0-24-$knb_type-knb_dict-$type-p$p.json \
                --is_post_metrics \
                --max_new_tokens_times $max_new_tokens_times \
                > logs/$DATE/$i-token$max_new_tokens_times-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-Qwen-1_8B-Chat-$cnt.log 2>&1 &
            wait
        done
    done
done