DATE=$(date +"%Y-%m-%d")
mkdir -p logs/$DATE
source activate base
export HUGGINGFACE_CACHE=/root/autodl-fs/
export PYTHONUNBUFFERED=1
# export CUDA_VISIBLE_DEVICES=1

# KNB
method=KNB
knb_layer=this_layer # last_layer，this_layer
type=max # mean
num_steps=100

# dataset 
# zsre数据部分字段有缺失
data_dir=./dataset/KnowEdit-ms/benchmark_wiki_counterfact_test_cf.json
data_type=counterfact
# start_idx_end_idx=0,885

# 5张gpu
gpus=5
cnt=1
batch=$((885/$gpus))

model=llama2-7b
batch_size=20 
# batch_size in 20 10 8 6 4
for t_loss in 0.1 0.4 0.7; do
  for p in 99 99.3 99.6 99.9; do
    for ff_attrs in self_attn.v_proj self_attn.o_proj mlp.up_proj mlp.down_proj self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj mlp.up_proj,mlp.gate_proj,mlp.down_proj; do
      for gpu in $(seq 0 $(($gpus-1))); do
        start_idx_end_idx=$(($gpu*$batch)),$(($gpu*$batch+$batch))
        knb_dict_path=/root/autodl-fs/knb-dict-bs/$model/$data_type/$knb_layer/bs$batch_size-p$p-$type.json
        echo $gpu-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-$model-$knb_layer-$cnt
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
            > logs/$DATE/$gpu-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-$model-$knb_layer-$cnt.log 2>&1 &
      done
      wait
    done
  done
done
