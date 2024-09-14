DATE=$(date +"%Y-%m-%d")
mkdir -p /root/autodl-fs/knb-logs/$DATE
source activate base
# export HUGGINGFACE_CACHE=/root/autodl-fs/
export HUGGINGFACE_CACHE=/root/
export PYTHONUNBUFFERED=1
# export CUDA_VISIBLE_DEVICES=1

# KNB
method=KNB
knb_layer=this_layer # last_layer this_layer
type=max # mean max
num_steps=100

# dataset 
data_dir=./dataset/KnowEdit-ms/benchmark_wiki_counterfact_test_cf.json
data_type=counterfact
# start_idx_end_idx=0,885
# data_dir=./dataset/KnowEdit-ms/benchmark_ZsRE_ZsRE-test-all.json
# data_type=zsre
# start_idx_end_idx=0,1304

# 5张gpu
gpus=5
cnt=1
batch=$((885/$gpus))

# model=llama2-7b-chat
# batch_size=20
# ff_attrs=mlp.up_proj
# p=96
# t_loss=0.4
# for n in 100 200 300 400 500 600 700 800 885; do
#     for gpu in $(seq 0 $(($gpus-1))); do
#         start_idx_end_idx=$(($gpu*$batch)),$(($gpu*$batch+$batch))
#         knb_dict_path=/root/knb-dict-n/$model/$data_type/$knb_layer/n$n-p$p-$type.json
#         echo $gpu-n$n-p$p-$ff_attrs-l$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-$model-$knb_layer-$cnt
#         CUDA_VISIBLE_DEVICES=$gpu python examples/run_knowedit.py \
#             --knb_layer $knb_layer \
#             --target_modules $ff_attrs \
#             --t_loss $t_loss \
#             --batch_size $batch_size \
#             --num_steps $num_steps \
#             --editing_method $method \
#             --p $p \
#             --data_type $data_type \
#             --data_dir $data_dir \
#             --start_idx_end_idx $start_idx_end_idx \
#             --knb_dict_path $knb_dict_path \
#             --hparams_dir ./hparams/KNB/$model.yaml \
#             --pre_file ./pre_edit/${model}_${data_type}_pre_edit.json \
#             > /root/autodl-fs/knb-logs/$DATE/$gpu-n$n-p$p-$ff_attrs-l$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-$model-$knb_layer-$cnt.log 2>&1 &
# 	done
# 	wait
# done
# wait

# model=llama2-7b-chat
# batch_size=20
# p=96
# t_loss=0.4
# n=200
# for ff_attrs in mlp.up_proj mlp.down_proj mlp.gate_proj self_attn.q_proj self_attn.k_proj self_attn.v_proj self_attn.o_proj; do
#     for gpu in $(seq 0 $(($gpus-1))); do
#         start_idx_end_idx=$(($gpu*$batch)),$(($gpu*$batch+$batch))
#         knb_dict_path=/root/knb-dict-n/$model/$data_type/$knb_layer/n$n-p$p-$type.json
#         echo $gpu-n$n-p$p-$ff_attrs-l$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-$model-$knb_layer-$cnt
#         CUDA_VISIBLE_DEVICES=$gpu python examples/run_knowedit.py \
#             --knb_layer $knb_layer \
#             --target_modules $ff_attrs \
#             --t_loss $t_loss \
#             --batch_size $batch_size \
#             --num_steps $num_steps \
#             --editing_method $method \
#             --p $p \
#             --data_type $data_type \
#             --data_dir $data_dir \
#             --start_idx_end_idx $start_idx_end_idx \
#             --knb_dict_path $knb_dict_path \
#             --hparams_dir ./hparams/KNB/$model.yaml \
#             --pre_file ./pre_edit/${model}_${data_type}_pre_edit.json \
#             > /root/autodl-fs/knb-logs/$DATE/$gpu-n$n-p$p-$ff_attrs-l$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-$model-$knb_layer-$cnt.log 2>&1 &
# 	done
# 	wait
# done

model=llama2-7b-chat
batch_size=1
p=96
t_loss=0.4
n=200
for ff_attrs in mlp.up_proj,mlp.down_proj,mlp.gate_proj,self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj mlp.up_proj,mlp.down_proj,mlp.gate_proj self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj mlp.up_proj,mlp.down_proj self_attn.q_proj,self_attn.k_proj self_attn.v_proj,self_attn.o_proj; do
    for gpu in $(seq 0 $(($gpus-1))); do
        start_idx_end_idx=$(($gpu*$batch)),$(($gpu*$batch+$batch))
        knb_dict_path=/root/knb-dict-n/$model/$data_type/$knb_layer/n$n-p$p-$type.json
        echo $gpu-n$n-p$p-$ff_attrs-l$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-$model-$knb_layer-$cnt
        CUDA_VISIBLE_DEVICES=$gpu python examples/run_knowedit.py \
            --knb_layer $knb_layer \
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
            > /root/autodl-fs/knb-logs/$DATE/$gpu-n$n-p$p-$ff_attrs-l$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-$model-$knb_layer-$cnt.log 2>&1 &
	done
	wait
done

wait
model=llama2-7b-chat
ff_attrs=mlp.up_proj
p=96
t_loss=0.4
n=200
for batch_size in 30 10 1; do
    for gpu in $(seq 0 $(($gpus-1))); do
        start_idx_end_idx=$(($gpu*$batch)),$(($gpu*$batch+$batch))
        knb_dict_path=/root/knb-dict-n/$model/$data_type/$knb_layer/n$n-p$p-$type.json
        echo $gpu-n$n-p$p-$ff_attrs-l$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-$model-$knb_layer-$cnt
        CUDA_VISIBLE_DEVICES=$gpu python examples/run_knowedit.py \
            --knb_layer $knb_layer \
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
            > /root/autodl-fs/knb-logs/$DATE/$gpu-n$n-p$p-$ff_attrs-l$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-$model-$knb_layer-$cnt.log 2>&1 &
	done
	wait
done
wait
