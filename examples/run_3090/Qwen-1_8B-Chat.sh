DATE=$(date +"%Y-%m-%d")
mkdir -p logs/$DATE
source activate ke2torch23cu121
export HUGGINGFACE_CACHE=/share/huggingface/
export PYTHONUNBUFFERED=1
# export CUDA_VISIBLE_DEVICES=1

# KNB
method=KNB
# 当batch_size=1时,num_steps设置小一点可能效果好,防止过拟合
num_steps=40
batch_size=1
type=orgin 

# num_steps=120
# batch_size=20
# type=max 
# type=mean
start_idx_end_idx=0,80
data_type=type2_80
cnt=1


# p=95.0
# 当batch_size=1时,p设置小一点可能效果好
p=99.95 
i=2
echo $i-$p-$start_idx_end_idx-$method-$batch_size-$num_steps-$data_type-$type-Qwen-1_8B-Chat-$cnt
CUDA_VISIBLE_DEVICES=$i python examples/run_CKnowEdit_qwen-1.8B.py \
    --batch_size $batch_size \
    --num_steps $num_steps \
    --editing_method $method \
    --p $p \
    --data_type CKnowEdit_$data_type \
    --metrics_save_dir ./ccks2024_output/$data_type \
    --data_dir /share/dataset/CKnowEditType/$data_type.json \
    --start_idx_end_idx $start_idx_end_idx \
    --knb_dict_path /share/knb_dict/Qwen-1_8B-Chat-CKnowEdit/$data_type-Qwen-1_8B-Chat-CKnowEdit-0-24-mlp.c_proj-knb_dict-$type-bs$batch_size.json \
    --is_post_metrics \
    > logs/$DATE/$i-$p-$start_idx_end_idx-$method-$batch_size-$num_steps-$data_type-Qwen-1_8B-Chat-$cnt.log 2>&1 &

# p=90.0
p=99.05
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