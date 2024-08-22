DATE=$(date +"%Y-%m-%d")
mkdir -p logs/$DATE
source activate torch24py310cu118
export HUGGINGFACE_CACHE=/root/
export PYTHONUNBUFFERED=1
# export CUDA_VISIBLE_DEVICES=1

# KNB
method=KNB
num_steps=120
type=max 
cnt=1

i=7
data_type=type6_50

max_new_tokens_times=4
for batch_size in {1,10}; do
    for t_loss in {5e-2,1e-2,5e-3,1e-3}; do
        for p in {99.2,99.4,99.6,99.8}; do
            ff_attrs=mlp.c_proj
            for skip in $(seq 0 10 40); do
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
                    --knb_dict_path /root/KNB-Qwen-1_8B-Chat-CKnowEdit/${data_type}_v4/$data_type-Qwen-1_8B-Chat-CKnowEdit-layer-0-24-$ff_attrs-w1*w2-knb_dict-$type-p$p.json \
                    --is_post_metrics \
                    --max_new_tokens_times $max_new_tokens_times \
                    > logs/$DATE/$i-token$max_new_tokens_times-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-Qwen-1_8B-Chat-$cnt.log 2>&1 &
                wait
            done
        done
    done
done

# for batch_size in {1,10,20,30}; do
#     for t_loss in {5e-2,1e-2,5e-3,1e-3}; do
#         for p in {99.1,99.3,99.5,99.7,99.9}; do
#             for ff_attrs in {mlp.w1,mlp.w2,,attn.c_proj,attn.c_attn}; do
#                 for skip in $(seq 0 10 70); do
#                     start_idx_end_idx=$skip,$((skip+10))
#                     echo $i-token$max_new_tokens_times-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-Qwen-1_8B-Chat-$cnt
#                     CUDA_VISIBLE_DEVICES=$i python examples/run_CKnowEdit_qwen-1.8B.py \
#                         --target_modules $ff_attrs \
#                         --t_loss $t_loss \
#                         --batch_size $batch_size \
#                         --num_steps $num_steps \
#                         --editing_method $method \
#                         --p $p \
#                         --data_type CKnowEdit_$data_type \
#                         --metrics_save_dir ./ccks2024_output/$data_type \
#                         --data_dir ./dataset/ccks2024_know_edit/CKnowEditType/$data_type.json \
#                         --start_idx_end_idx $start_idx_end_idx \
#                         --knb_dict_path /root/KNB-Qwen-1_8B-Chat-CKnowEdit/${data_type}_v4/$data_type-Qwen-1_8B-Chat-CKnowEdit-layer-0-24-$ff_attrs-knb_dict-$type-p$p.json \
#                         --is_post_metrics \
#                         --max_new_tokens_times $max_new_tokens_times \
#                         > logs/$DATE/$i-token$max_new_tokens_times-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-Qwen-1_8B-Chat-$cnt.log 2>&1 &
#                     wait
#                 done
#             done
#         done
#     done
# done