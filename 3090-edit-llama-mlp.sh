DATE=$(date +"%Y-%m-%d")
mkdir -p logs/$DATE
source activate ccks2024
export HUGGINGFACE_CACHE=/share/huggingface/
export PYTHONUNBUFFERED=1
# export CUDA_VISIBLE_DEVICES=1

# KNB
method=KNB
knb_layer=this_layer # last_layer，this_layer
type=max             # mean
num_steps=100

# dataset
# zsre数据部分字段有缺失
# data_dir=./dataset/KnowEdit-ms/benchmark_wiki_counterfact_test_cf.json
# data_dir = ./dataset/KnowEdit-ms/benchmark_wiki_recent_recent_test.json
# data_type=recent
data_dir=./dataset/KnowEdit-ms/benchmark_wiki_recent_recent_test.json
data_type=recent
# start_idx_end_idx=0,885

# 5张gpu
gpus=3
cnt=1
batch=$((1266 / $gpus))

model=llama3-8b-instruct
t_loss=0.4
start_idx_end_idx=0,1266
# n=200
# n=200
# mlp.up_proj mlp.down_proj mlp.gate_proj self_attn.q_proj self_attn.k_proj self_attn.v_proj self_attn.o_proj;
for ff_attrs in mlp.down_proj; do
  # ff_attrs=mlp.gate_proj
  for batch_size in 10; do
    for n in 300 400; do
      for p in 95; do
        # for batch_size in 1; do
        # for n in 100; do
        #   for p in 90 91 92 93 94 95; do
        # for batch_size in 1; do
        # for n in 200 300 400; do
        #   for p in 90 91 92 93 94 95 96 97 98 99; do
        for gpu in 1 2 3; do
          # if [ $gpu -eq 0]; then
          #   start_idx_end_idx=$(($gpu * $batch)),$(($gpu * $batch + $batch))
          # else
          #   start_idx_end_idx=$((($gpu - 1) * $batch)),$((($gpu - 1) * $batch + $batch))
          # fi
          start_idx_end_idx=$((($gpu - 1) * $batch)),$((($gpu - 1) * $batch + $batch))
          knb_dict_path=/home/yantao/llm2024/knb-dict-n/llama3-8b-instruct/recent/this_layer/n$n-p$p-max.json
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
            --hparams_dir /home/yantao/llm2024/EasyEdit/hparams/KNB/llama3-8b-instruct.yaml \
            --pre_file ./pre_edit/${model}_${data_type}_pre_edit.json \
            >logs/$DATE/$gpu-$p-$n-$ff_attrs-$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-$model-$knb_layer-$cnt.log 2>&1 &
        done
        wait
      done
      rm ./pre_edit/${model}_${data_type}_pre_edit.json
    done
  done
done
