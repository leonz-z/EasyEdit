DATE=2024-6-3
# if log/$DATE does not exist, create it
if [ ! -d "log/$DATE" ]; then
    mkdir log/$DATE
fi
# 3090
# export HUGGINGFACE_CACHE=/share/huggingface/ # 3090
# MODEL=Llama-2-7b-ms # 3090

# A100
MODEL=Llama-2-7b-hf # A100
export HUGGINGFACE_CACHE=/home/bingxing2/public/models/llama2 # A100
module load compilers/cuda/11.8
module load cudnn/8.8.1.3_cuda11.x
module load compilers/gcc/12.2.0
conda activate ke
# DATA_TEST=ZsRE-test-all
# DATA_TYPE=zsre
EDIT_METHOD=ROME
DATA_TEST=recent_test
DATA_TRAIN=recent_train
DATA_TYPE=recent
NUM=1

CUDA_VISIBLE_DEVICES=0 nohup python run_knowedit_llama2.py \
    --editing_method $EDIT_METHOD \
    --hparams_dir ../hparams/$EDIT_METHOD/$MODEL \
    --data_dir ../LLMKnowledgeEditDataset/ccks2024_know_edit/$DATA_TEST.json \
    --train_data_path ../LLMKnowledgeEditDataset/ccks2024_know_edit/$DATA_TRAIN.json \
    --datatype $DATA_TYPE \
    > log/$DATE/$EDIT_METHOD-$MODEL-$DATA_TEST-$NUM.log 2>&1 &
