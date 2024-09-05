DATE=$(date +"%Y-%m-%d")
mkdir -p logs/$DATE
source activate base
export HUGGINGFACE_CACHE=/root/autodl-fs/
export PYTHONUNBUFFERED=1
# export CUDA_VISIBLE_DEVICES=1

# KNB
method=KNB
data_type=counterfact
num_steps=100
cnt=1

type=max 
p=99
t_loss=1e-1
batch_size=50

model=qwen1.5-7b
ff_attrs=mlp.up_proj
next_token=answer_next_token # argmax_next_token
# knb_dict_path=/root/knb-dict-2024/$model/$data_type/${next_token}_target_new/$ff_attrs/bs$batch_size-p$p-$type.json \
knb_dict_path=/root/knb_dict/test.json

i=1
# start_idx_end_idx=0,885
start_idx_end_idx=0,5

CUDA_VISIBLE_DEVICES=$i nohup python examples/run_knowedit.py \
    --target_modules $ff_attrs \
    --t_loss $t_loss \
    --batch_size $batch_size \
    --num_steps $num_steps \
    --editing_method $method \
    --p $p \
    --data_type $data_type \
    --metrics_save_dir ./knb_output/$data_type \
    --data_dir ./dataset/KnowEdit-ms/benchmark_wiki_counterfact_test_cf.json \
    --start_idx_end_idx $start_idx_end_idx \
    --knb_dict_path $knb_dict_path \
    --hparams_dir ./hparams/KNB/$model.yaml \
    --pre_file ./pre_edit/${model}_pre_edit_$start_idx_end_idx.json \
    > logs/$DATE/$i-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-$model-$cnt.log 2>&1 &