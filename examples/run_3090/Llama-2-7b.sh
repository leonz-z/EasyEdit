DATE=$(date +"%Y-%m-%d")
mkdir -p logs/$DATE
source activate ke2torch23cu121
export HUGGINGFACE_CACHE=/share/huggingface/
export PYTHONUNBUFFERED=1
# export CUDA_VISIBLE_DEVICES=1

# KNB
method=KNB
num_steps=120
type=max 
model=Llama-2-7b
cnt=1

i=0
data_type=counterfact

ff_attrs=mlp.down_proj
p=99.85
t_loss=1e-1
batch_size=1

start_idx_end_idx=0,400
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
    --knb_dict_path ../knb_dict/all-llama2-counterfact/all-Llama-2-7b-hf-counterfact-knb_dict-orgin-max-99.85.json \
    --hparams_dir ./hparams/KNB/Llama-2-7b-ms.yaml \
    --pre_file ./pre_edit/Llama-2-7b-hf_counterfact_pre_edit_all.json \
    > logs/$DATE/$i-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-Llama-2-7b-$cnt.log 2>&1 &

i=1
start_idx_end_idx=400,885
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
    --knb_dict_path ../knb_dict/all-llama2-counterfact/all-Llama-2-7b-hf-counterfact-knb_dict-orgin-max-99.85.json \
    --hparams_dir ./hparams/KNB/Llama-2-7b-ms.yaml \
    --pre_file ./pre_edit/Llama-2-7b-hf_counterfact_pre_edit_all.json \
    > logs/$DATE/$i-$p-$ff_attrs-$t_loss-$start_idx_end_idx-$method-bs$batch_size-epoch$num_steps-$data_type-$type-Llama-2-7b-$cnt.log 2>&1 &
