DATE=$(date +"%Y-%m-%d")
mkdir -p logs/$DATE
source activate ke2torch23cu121
export HUGGINGFACE_CACHE=/share/huggingface/
export PYTHONUNBUFFERED=1
# export CUDA_VISIBLE_DEVICES=1

# KNB
method=KNB
# 当batch_size=1时,num_steps设置小一点可能效果好,防止过拟合
num_steps=100 # < t_loss阈值,跳出迭代训练
batch_size=1
type=mean # 当batch_size=1时,type=mean,max一样的
# type=orgin 

# num_steps=120
# batch_size=10
# type=max 
# type=mean
cnt=1


# p=95.0
# 当batch_size=1时,p设置小一点可能效果好
i=3
start_idx_end_idx=0,70
data_type=type5_70
t_loss=1e-2
ff_attrs=mlp.c_proj
max_new_tokens_times=2
for p in {99.05,99.35,99.65,99.95}; do
    echo $i-token$max_new_tokens_times-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-Qwen-1_8B-Chat-$cnt
    CUDA_VISIBLE_DEVICES=$i python examples/run_CKnowEdit_qwen-1.8B.py \
        --t_loss $t_loss \
        --batch_size $batch_size \
        --num_steps $num_steps \
        --editing_method $method \
        --p $p \
        --data_type CKnowEdit_$data_type \
        --metrics_save_dir ./ccks2024_output/$data_type \
        --data_dir /share/dataset/CKnowEditType/$data_type.json \
        --start_idx_end_idx $start_idx_end_idx \
        --knb_dict_path /share/knb_dict/Qwen-1_8B-Chat-CKnowEdit/$data_type/$data_type-Qwen-1_8B-Chat-CKnowEdit-layer-0-24-$ff_attrs-knb_dict-$type-bs$batch_size.json \
        --is_post_metrics \
        --max_new_tokens_times $max_new_tokens_times \
        > logs/$DATE/$i-token$max_new_tokens_times-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-Qwen-1_8B-Chat-$cnt.log 2>&1 &
    wait
done

# 0,70_type5_70-Qwen-1_8B-Chat-CKnowEdit-layer-0-24-mlp.c_proj-knb_dict-mean-bs1_100_p99.35_rsTrue_a1_pd0.1_bias_none_t_loss0.1_wd0.json
# t_loss=1e-1
# ff_attrs=mlp.c_proj
# p=99.35
# i=3
# for max_new_tokens_times in {4,6,8,10,15,20}; do
#     echo $i-$max_new_tokens_times-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-$batch_size-$num_steps-$data_type-$type-Qwen-1_8B-Chat-$cnt
#     CUDA_VISIBLE_DEVICES=$i python examples/run_CKnowEdit_qwen-1.8B.py \
#         --t_loss $t_loss \
#         --batch_size $batch_size \
#         --num_steps $num_steps \
#         --editing_method $method \
#         --p $p \
#         --data_type CKnowEdit_$data_type \
#         --metrics_save_dir ./ccks2024_output/$data_type \
#         --data_dir /share/dataset/CKnowEditType/$data_type.json \
#         --start_idx_end_idx $start_idx_end_idx \
#         --knb_dict_path /share/knb_dict/Qwen-1_8B-Chat-CKnowEdit/$data_type/$data_type-Qwen-1_8B-Chat-CKnowEdit-layer-0-24-$ff_attrs-knb_dict-$type-bs$batch_size.json \
#         --is_post_metrics \
#         --max_new_tokens_times $max_new_tokens_times \
#         > logs/$DATE/$i-$max_new_tokens_times-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-$batch_size-$num_steps-$data_type-$type-Qwen-1_8B-Chat-$cnt.log 2>&1 &
#     wait
# done

# ff_attrs=mlp.w1
# for p in {99.35,99.55,99.75,99.95}; do
#     echo $i-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-$batch_size-$num_steps-$data_type-$type-Qwen-1_8B-Chat-$cnt
#     CUDA_VISIBLE_DEVICES=$i python examples/run_CKnowEdit_qwen-1.8B.py \
#         --t_loss $t_loss \
#         --batch_size $batch_size \
#         --num_steps $num_steps \
#         --editing_method $method \
#         --p $p \
#         --data_type CKnowEdit_$data_type \
#         --metrics_save_dir ./ccks2024_output/$data_type \
#         --data_dir /share/dataset/CKnowEditType/$data_type.json \
#         --start_idx_end_idx $start_idx_end_idx \
#         --knb_dict_path /share/knb_dict/Qwen-1_8B-Chat-CKnowEdit/$data_type/$data_type-Qwen-1_8B-Chat-CKnowEdit-layer-0-24-$ff_attrs-knb_dict-$type-bs$batch_size.json \
#         --is_post_metrics \
#         > logs/$DATE/$i-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-$batch_size-$num_steps-$data_type-$type-Qwen-1_8B-Chat-$cnt.log 2>&1 &
#     # i=$((i+1))
#     wait
# done
# wait

# ff_attrs=attn.c_proj
# for p in {99.35,99.55,99.75,99.95}; do
#     echo $i-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-$batch_size-$num_steps-$data_type-$type-Qwen-1_8B-Chat-$cnt
#     CUDA_VISIBLE_DEVICES=$i python examples/run_CKnowEdit_qwen-1.8B.py \
#         --t_loss $t_loss \
#         --batch_size $batch_size \
#         --num_steps $num_steps \
#         --editing_method $method \
#         --p $p \
#         --data_type CKnowEdit_$data_type \
#         --metrics_save_dir ./ccks2024_output/$data_type \
#         --data_dir /share/dataset/CKnowEditType/$data_type.json \
#         --start_idx_end_idx $start_idx_end_idx \
#         --knb_dict_path /share/knb_dict/Qwen-1_8B-Chat-CKnowEdit/$data_type/$data_type-Qwen-1_8B-Chat-CKnowEdit-layer-0-24-$ff_attrs-knb_dict-$type-bs$batch_size.json \
#         --is_post_metrics \
#         > logs/$DATE/$i-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-$batch_size-$num_steps-$data_type-$type-Qwen-1_8B-Chat-$cnt.log 2>&1 &
#     # i=$((i+1))
#     wait
# done

# i=1
# for p in {92.5,97.5,99.25}; do
#     echo $i-$p-$start_idx_end_idx-$method-$batch_size-$num_steps-$data_type-$type-Qwen-1_8B-Chat-$cnt
#     CUDA_VISIBLE_DEVICES=$i python examples/run_CKnowEdit_qwen-1.8B.py \
#         --batch_size $batch_size \
#         --num_steps $num_steps \
#         --editing_method $method \
#         --p $p \
#         --data_type CKnowEdit_$data_type \
#         --metrics_save_dir ./ccks2024_output/$data_type \
#         --data_dir /share/dataset/CKnowEditType/$data_type.json \
#         --start_idx_end_idx $start_idx_end_idx \
#         --knb_dict_path /share/knb_dict/Qwen-1_8B-Chat-CKnowEdit/$data_type-Qwen-1_8B-Chat-CKnowEdit-0-24-mlp.c_proj-knb_dict-$type-bs$batch_size.json \
#         --is_post_metrics \
#         > logs/$DATE/$i-$p-$start_idx_end_idx-$method-$batch_size-$num_steps-$data_type-Qwen-1_8B-Chat-$cnt.log 2>&1 &
#     i=$((i+1))
# done
# wait



# p=99.75 
# i=2
# echo $i-$p-$start_idx_end_idx-$method-$batch_size-$num_steps-$data_type-$type-Qwen-1_8B-Chat-$cnt
# CUDA_VISIBLE_DEVICES=$i python examples/run_CKnowEdit_qwen-1.8B.py \
#     --batch_size $batch_size \
#     --num_steps $num_steps \
#     --editing_method $method \
#     --p $p \
#     --data_type CKnowEdit_$data_type \
#     --metrics_save_dir ./ccks2024_output/$data_type \
#     --data_dir /share/dataset/CKnowEditType/$data_type.json \
#     --start_idx_end_idx $start_idx_end_idx \
#     --knb_dict_path /share/knb_dict/Qwen-1_8B-Chat-CKnowEdit/$data_type-Qwen-1_8B-Chat-CKnowEdit-0-24-mlp.c_proj-knb_dict-$type-bs$batch_size.json \
#     --is_post_metrics \
#     > logs/$DATE/$i-$p-$start_idx_end_idx-$method-$batch_size-$num_steps-$data_type-Qwen-1_8B-Chat-$cnt.log 2>&1 &

# # p=90.0
# p=99.55
# i=3
# echo $i-$p-$start_idx_end_idx-$method-$batch_size-$num_steps-$data_type-$type-Qwen-1_8B-Chat-$cnt
# CUDA_VISIBLE_DEVICES=$i python examples/run_CKnowEdit_qwen-1.8B.py \
#     --batch_size $batch_size \
#     --num_steps $num_steps \
#     --editing_method $method \
#     --p $p \
#     --data_type CKnowEdit_$data_type \
#     --metrics_save_dir ./ccks2024_output/$data_type \
#     --data_dir ./dataset/ccks2024_know_edit/CKnowEditType/$data_type.json \
#     --start_idx_end_idx $start_idx_end_idx \
#     --knb_dict_path /share/knb_dict/Qwen-1_8B-Chat-CKnowEdit/$data_type-Qwen-1_8B-Chat-CKnowEdit-0-24-mlp.c_proj-knb_dict-$type-bs$batch_size.json \
#     --is_post_metrics \
#     > logs/$DATE/$i-$p-$start_idx_end_idx-$method-$batch_size-$num_steps-$data_type-Qwen-1_8B-Chat-$cnt.log 2>&1 &

DATE=$(date +"%Y-%m-%d")
mkdir -p logs/$DATE
source activate ke2torch23cu121
export HUGGINGFACE_CACHE=/share/huggingface/
export PYTHONUNBUFFERED=1
# export CUDA_VISIBLE_DEVICES=1

# KNB
method=KNB
num_steps=60
start_idx_end_idx=0,80
data_type=type2_80
type=max 
# type=mean
cnt=1

batch_size=10
p=99.05
i=0
echo $i-$p-$start_idx_end_idx-$method-$batch_size-$num_steps-$data_type-$type-Qwen-1_8B-Chat-$cnt
CUDA_VISIBLE_DEVICES=$i python examples/run_CKnowEdit_qwen-1.8B.py \
    --batch_size $batch_size \
    --num_steps $num_steps \
    --editing_method $method \
    --p $p \
    --data_type CKnowEdit_$data_type \
    --metrics_save_dir ./ccks2024_output/$data_type \
    --data_dir ./dataset/ccks2024_know_edit/CKnowEditType/$data_type.json \
    --start_idx_end_idx $start_idx_end_idx \
    --knb_dict_path ../knb_dict/Qwen-1_8B-Chat-CKnowEdit/$data_type-Qwen-1_8B-Chat-CKnowEdit-0-24-mlp.c_proj-knb_dict-$type-bs$batch_size.json \
    --is_post_metrics \
    > logs/$DATE/$i-$p-$start_idx_end_idx-$method-$batch_size-$num_steps-$data_type-Qwen-1_8B-Chat-$cnt.log 2>&1 &

batch_size=10
p=99.95
i=1
echo $i-$p-$start_idx_end_idx-$method-$batch_size-$num_steps-$data_type-$type-Qwen-1_8B-Chat-$cnt
CUDA_VISIBLE_DEVICES=$i python examples/run_CKnowEdit_qwen-1.8B.py \
    --batch_size $batch_size \
    --num_steps $num_steps \
    --editing_method $method \
    --p $p \
    --data_type CKnowEdit_$data_type \
    --metrics_save_dir ./ccks2024_output/$data_type \
    --data_dir ./dataset/ccks2024_know_edit/CKnowEditType/$data_type.json \
    --start_idx_end_idx $start_idx_end_idx \
    --knb_dict_path ../knb_dict/Qwen-1_8B-Chat-CKnowEdit/$data_type-Qwen-1_8B-Chat-CKnowEdit-0-24-mlp.c_proj-knb_dict-$type-bs$batch_size.json \
    --is_post_metrics \
    > logs/$DATE/$i-$p-$start_idx_end_idx-$method-$batch_size-$num_steps-$data_type-Qwen-1_8B-Chat-$cnt.log 2>&1 &

batch_size=20
p=99.05
i=2

echo $i-$p-$start_idx_end_idx-$method-$batch_size-$num_steps-$data_type-$type-Qwen-1_8B-Chat-$cnt
CUDA_VISIBLE_DEVICES=$i python examples/run_CKnowEdit_qwen-1.8B.py \
    --batch_size $batch_size \
    --num_steps $num_steps \
    --editing_method $method \
    --p $p \
    --data_type CKnowEdit_$data_type \
    --metrics_save_dir ./ccks2024_output/$data_type \
    --data_dir ./dataset/ccks2024_know_edit/CKnowEditType/$data_type.json \
    --start_idx_end_idx $start_idx_end_idx \
    --knb_dict_path ../knb_dict/Qwen-1_8B-Chat-CKnowEdit/$data_type-Qwen-1_8B-Chat-CKnowEdit-0-24-mlp.c_proj-knb_dict-$type-bs$batch_size.json \
    --is_post_metrics \
    > logs/$DATE/$i-$p-$start_idx_end_idx-$method-$batch_size-$num_steps-$data_type-Qwen-1_8B-Chat-$cnt.log 2>&1 &

batch_size=20
p=99.95
i=3
echo $i-$p-$start_idx_end_idx-$method-$batch_size-$num_steps-$data_type-$type-Qwen-1_8B-Chat-$cnt
CUDA_VISIBLE_DEVICES=$i python examples/run_CKnowEdit_qwen-1.8B.py \
    --batch_size $batch_size \
    --num_steps $num_steps \
    --editing_method $method \
    --p $p \
    --data_type CKnowEdit_$data_type \
    --metrics_save_dir ./ccks2024_output/$data_type \
    --data_dir ./dataset/ccks2024_know_edit/CKnowEditType/$data_type.json \
    --start_idx_end_idx $start_idx_end_idx \
    --knb_dict_path ../knb_dict/Qwen-1_8B-Chat-CKnowEdit/$data_type-Qwen-1_8B-Chat-CKnowEdit-0-24-mlp.c_proj-knb_dict-$type-bs$batch_size.json \
    --is_post_metrics \
    > logs/$DATE/$i-$p-$start_idx_end_idx-$method-$batch_size-$num_steps-$data_type-Qwen-1_8B-Chat-$cnt.log 2>&1 &


# # LoRA
# method=LoRA
# batch_size=35
# num_steps=100
# target_modules=all
# layers=0,24
# lora_type=adalora
# cnt=use_rslora:true,bias:lora_only,r8,a32,tr8,ir16


# data_batch=35
# skip=0
# for i in $(seq 0 2); do
#     idx1=$((i * data_batch + skip))
#     idx2=$((idx1 + data_batch))
#     start_idx_end_idx=$idx1,$idx2
#     echo $i,$start_idx_end_idx,$batch_size,$num_steps,$target_modules,$layers,$lora_type,$cnt
#     CUDA_VISIBLE_DEVICES=$i python examples/run_CKnowEdit_qwen-1.8B.py \
#         --batch_size $batch_size \
#         --num_steps $num_steps \
#         --start_idx_end_idx $start_idx_end_idx \
#         --target_modules $target_modules \
#         --layers $layers \
#         --editing_method $method \
#         --lora_type $lora_type \
#         --is_post_metrics \
#         --metrics_save_dir ./ccks2024_output/task1_133 \
#         > logs/$DATE/$method-$lora_type-$start_idx_end_idx-$target_modules-$layers-$batch_size-$num_steps-Qwen-1_8B-Chat-$cnt.log 2>&1 &
# done

# i=3
# start_idx_end_idx=105,133
# echo $i,$start_idx_end_idx,$batch_size,$num_steps,$target_modules,$layers,$lora_type,$cnt
# CUDA_VISIBLE_DEVICES=$i python examples/run_CKnowEdit_qwen-1.8B.py \
    # --batch_size $batch_size \
    # --num_steps $num_steps \
    # --start_idx_end_idx $start_idx_end_idx \
    # --target_modules $target_modules \
    # --layers $layers \
    # --editing_method $method \
    # --lora_type $lora_type \
    # --is_post_metrics \
    # --metrics_save_dir ./ccks2024_output/task1_133 \
    # > logs/$DATE/$method-$lora_type-$start_idx_end_idx-$target_modules-$layers-$batch_size-$num_steps-Qwen-1_8B-Chat-$cnt.log 2>&1 &

# method=IKE
# cnt=1
# python examples/run_CKnowEdit_qwen-1.8B.py \
#     --editing_method $method \
#     > logs/$DATE/$method-Qwen-1_8B-Chat-$cnt.log 2>&1 &

# # MEMIT
# method=MEMIT
# layers=0,6
# # layers=6,12
# # layers=12,18
# # start_idx_end_idx=400,500
# # start_idx_end_idx=500,600
# start_idx_end_idx=600,700
# # start_idx_end_idx=0,400
# cnt=1
# python examples/run_CKnowEdit_qwen-1.8B.py \
#     --editing_method $method \
#     --layers $layers \
#     --start_idx_end_idx $start_idx_end_idx \
#     > logs/$DATE/$method-$layers-$start_idx_end_idx-Qwen-1_8B-Chat-$cnt.log 2>&1 &
#     # > logs/$DATE/$method-$layers-Qwen-1_8B-Chat-$cnt.log 2>&1 &

# # PMET
# method=PMET
# # layers=0,6
# # layers=6,12
# layers=12,18
# # start_idx_end_idx=400,500
# # start_idx_end_idx=500,600
# # start_idx_end_idx=600,700
# # start_idx_end_idx=0,400
# cnt=1
# python examples/run_CKnowEdit_qwen-1.8B.py \
#     --editing_method $method \
#     --layers $layers \
#     > logs/$DATE/$method-$layers-Qwen-1_8B-Chat-$cnt.log 2>&1 &
#     # --start_idx_end_idx $start_idx_end_idx \
#     # > logs/$DATE/$method-$layers-$start_idx_end_idx-Qwen-1_8B-Chat-$cnt.log 2>&1 &

# target_r (`int`): The target average rank of incremental matrix.
# init_r (`int`): The initial rank for each incremental matrix.
# tinit (`int`): The steps of initial fine-tuning warmup.
# tfinal (`int`): The step of final fine-tuning.
# deltaT (`int`): The time internval between two budget allocations.
# beta1 (`float`): The hyperparameter of EMA for sensitivity smoothing.
# beta2 (`float`): The hyperparameter of EMA for undertainty quantification.
# orth_reg_weight (`float`): The coefficient of orthogonal regularization.
# total_step (`int`): The total training steps that should be specified before training.
# rank_pattern (`list`): The allocated rank for each weight matrix by RankAllocator.