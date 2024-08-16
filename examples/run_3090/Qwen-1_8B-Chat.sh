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