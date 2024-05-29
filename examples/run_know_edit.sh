DATE=2024年5月29日
EDIT_METHOD=ROME
MODEL=Qwen-7B
# DATA=ZsRE-test-all
DATA_TEST=recent_test
DATA_TRAIN=recent_train
DATA_TYPE=recent
NUM=1

CUDA_VISIBLE_DEVICES=2 nohup python run_knowedit_llama2.py \
    --editing_method $EDIT_METHOD \
    --hparams_dir ../hparams/$EDIT_METHOD/$MODEL \
    --data_dir ../dataset/round1_dataset/$DATA.json \
    --datatype $DATA_TYPE \
    > log/$DATE/$EDIT_METHOD-$MODEL-$DATA-$NUM.log 2>&1 &
