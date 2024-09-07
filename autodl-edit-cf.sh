DATE=$(date +"%Y-%m-%d")
mkdir -p logs/$DATE
source activate base
export HUGGINGFACE_CACHE=/root/autodl-fs/
export PYTHONUNBUFFERED=1
# export CUDA_VISIBLE_DEVICES=1

# # model
# model=gpt-j-6b
# ff_attrs=mlp.fc_out

# model=llama3-8b
model=llama2-7b
# model=qwen1.5-7b
# model=qwen2-7b
ff_attrs=mlp.down_proj

# model=qwen-7b
# model=gpt2
# model=gpt-neo-xxx
# ff_attrs=mlp.c_proj

# KNB
method=KNB
type=max 
p=99.9
t_loss=1e-1
batch_size=20
num_steps=100

# dataset 
# zsre数据部分字段有缺失
data_dir=./dataset/KnowEdit-ms/benchmark_wiki_counterfact_test_cf.json
data_type=counterfact
start_idx_end_idx=0,885

# knb dict
next_token=answer_next_token # argmax_next_token
knb_dict_path=/root/autodl-fs/knb-dict-2024/$model/$data_type/${next_token}_target_new/$ff_attrs/bs$batch_size-p$p-$type.json \
# knb_dict_path=/root/knb_dict/test.json

# 5张gpu
gpus=5
cnt=1
batch=$((885/$gpus))

type=mean 
p=99.9
# t_loss
for t_loss in 1e-1 2e-1 3e-1 4e-1 5e-1 6e-1 7e-1 8e-1 9e-1 5e-2 1e-2 5e-3 1e-3; do
    for gpu in $(seq 0 $(($gpus-1))); do
        start_idx_end_idx=$(($gpu*$batch)),$(($gpu*$batch+$batch))
        echo $gpu-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-$model-$cnt

        CUDA_VISIBLE_DEVICES=$gpu python examples/run_knowedit.py \
            --target_modules $ff_attrs \
            --t_loss $t_loss \
            --batch_size $batch_size \
            --num_steps $num_steps \
            --editing_method $method \
            --p $p \
            --data_type $data_type \
            --data_dir $data_dir \
            --start_idx_end_idx $start_idx_end_idx \
            --knb_dict_path $knb_dict_path \
            --hparams_dir ./hparams/KNB/$model.yaml \
            --pre_file ./pre_edit/${model}_${data_type}_pre_edit.json \
            > logs/$DATE/$gpu-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-$model-$cnt.log 2>&1 &
    done
    wait
done
wait

type=max 
p=99.9
# t_loss
for t_loss in 1e-1 2e-1 3e-1 4e-1 5e-1 6e-1 7e-1 8e-1 9e-1 5e-2 1e-2 5e-3 1e-3; do
    for gpu in $(seq 0 $(($gpus-1))); do
        start_idx_end_idx=$(($gpu*$batch)),$(($gpu*$batch+$batch))
        echo $gpu-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-$model-$cnt

        CUDA_VISIBLE_DEVICES=$gpu python examples/run_knowedit.py \
            --target_modules $ff_attrs \
            --t_loss $t_loss \
            --batch_size $batch_size \
            --num_steps $num_steps \
            --editing_method $method \
            --p $p \
            --data_type $data_type \
            --data_dir $data_dir \
            --start_idx_end_idx $start_idx_end_idx \
            --knb_dict_path $knb_dict_path \
            --hparams_dir ./hparams/KNB/$model.yaml \
            --pre_file ./pre_edit/${model}_${data_type}_pre_edit.json \
            > logs/$DATE/$gpu-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-$model-$cnt.log 2>&1 &
    done
    wait
done
wait

type=mean 
p=90
# t_loss
for t_loss in 1e-1 2e-1 3e-1 4e-1 5e-1 6e-1 7e-1 8e-1 9e-1 5e-2 1e-2 5e-3 1e-3; do
    for gpu in $(seq 0 $(($gpus-1))); do
        start_idx_end_idx=$(($gpu*$batch)),$(($gpu*$batch+$batch))
        echo $gpu-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-$model-$cnt

        CUDA_VISIBLE_DEVICES=$gpu python examples/run_knowedit.py \
            --target_modules $ff_attrs \
            --t_loss $t_loss \
            --batch_size $batch_size \
            --num_steps $num_steps \
            --editing_method $method \
            --p $p \
            --data_type $data_type \
            --data_dir $data_dir \
            --start_idx_end_idx $start_idx_end_idx \
            --knb_dict_path $knb_dict_path \
            --hparams_dir ./hparams/KNB/$model.yaml \
            --pre_file ./pre_edit/${model}_${data_type}_pre_edit.json \
            > logs/$DATE/$gpu-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-$model-$cnt.log 2>&1 &
    done
    wait
done
wait

type=max 
p=90
# t_loss
for t_loss in 1e-1 2e-1 3e-1 4e-1 5e-1 6e-1 7e-1 8e-1 9e-1 5e-2 1e-2 5e-3 1e-3; do
    for gpu in $(seq 0 $(($gpus-1))); do
        start_idx_end_idx=$(($gpu*$batch)),$(($gpu*$batch+$batch))
        echo $gpu-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-$model-$cnt

        CUDA_VISIBLE_DEVICES=$gpu python examples/run_knowedit.py \
            --target_modules $ff_attrs \
            --t_loss $t_loss \
            --batch_size $batch_size \
            --num_steps $num_steps \
            --editing_method $method \
            --p $p \
            --data_type $data_type \
            --data_dir $data_dir \
            --start_idx_end_idx $start_idx_end_idx \
            --knb_dict_path $knb_dict_path \
            --hparams_dir ./hparams/KNB/$model.yaml \
            --pre_file ./pre_edit/${model}_${data_type}_pre_edit.json \
            > logs/$DATE/$gpu-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-$model-$cnt.log 2>&1 &
    done
    wait
done
wait