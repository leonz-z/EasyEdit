DATE=$(date +"%Y-%m-%d")
mkdir -p logs/$DATE
source activate ke2torch23cu121
export HUGGINGFACE_CACHE=/share/huggingface/
export PYTHONUNBUFFERED=1
# export CUDA_VISIBLE_DEVICES=1

# KNB
method=KNB
data_type=counterfact
num_steps=100
cnt=1

type=max 
p=99.9
t_loss=1e-1
batch_size=30

model=gpt-j-6b
ff_attrs=mlp.fc_out
next_token=answer_next_token # argmax_next_token
knb_dict_path=../knb-dict-2024/$model/$data_type/${next_token}_target_new/$ff_attrs/bs$batch_size-p$p-$type.json

i=0
start_idx_end_idx=0,885
CUDA_VISIBLE_DEVICES=$i nohup python examples/run_knowedit.py \
    --target_modules $ff_attrs \
    --t_loss $t_loss \
    --batch_size $batch_size \
    --num_steps $num_steps \
    --editing_method $method \
    --p $p \
    --data_type $data_type \
    --data_dir ./dataset/KnowEdit-ms/benchmark_wiki_counterfact_test_cf.json \
    --start_idx_end_idx $start_idx_end_idx \
    --knb_dict_path $knb_dict_path \
    --hparams_dir ./hparams/KNB/$model.yaml \
    --pre_file ./pre_edit/${model}_pre_edit_$start_idx_end_idx.json \
    > logs/$DATE/$i-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-$model-$cnt.log 2>&1 &