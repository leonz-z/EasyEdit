# model=llama2-7b-chat
# 知识定位的数据条数：n = 200 400 600 800
# knb dict path ：knb-dict-bs/llama2-7b-chat/counterfact/this_layer/n{n}-p{p}-max.json
# batch_size=10（需要查看log观察是否爆显存）
# t_loss=0.4
# p=90 92 94 96 98
# ff_attrs=mlp.up_proj mlp.down_proj mlp.up_proj,mlp.gate_proj,mlp.down_proj （mlp）
DATE=$(date +"%Y-%m-%d")
mkdir -p logs/$DATE
source activate base
export HUGGINGFACE_CACHE=/root/autodl-fs/
export PYTHONUNBUFFERED=1
# export CUDA_VISIBLE_DEVICES=1

# KNB
method=KNB
knb_layer=this_layer # last_layer，this_layer
type=max             # mean
num_steps=100

# dataset
# zsre数据部分字段有缺失
data_dir=./dataset/KnowEdit-ms/benchmark_wiki_counterfact_test_cf.json
data_type=counterfact
# start_idx_end_idx=0,885

# 5张gpu
gpus=5
cnt=1
batch=$((885 / $gpus))

model=llama2-7b-chat
batch_size=20
t_loss=0.4
start_idx_end_idx=0,885
# for p in 90 92 94 96 98; do
p=96
# for n in 200 400 600 800; do
n=200
for ff_attrs in mlp.gate_proj; do
  for gpu in $(seq 0 $(($gpus - 1))); do
    start_idx_end_idx=$(($gpu * $batch)),$(($gpu * $batch + $batch))
    knb_dict_path=/root/autodl-fs/knb-dict-bs/llama2-7b-chat/counterfact/this_layer/n$n-p$p-max.json
    # echo $knb_dict_path
    echo $gpu-$p-$n-$ff_attrs-$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-$model-$knb_layer-$cnt
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
      >logs/$DATE/$gpu-$p-$n-$ff_attrs-$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-$model-$knb_layer-$cnt.log 2>&1 &
  done
  wait
done
