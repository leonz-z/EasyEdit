DATE=2024年5月30日
EDIT_METHOD=KN
MODEL=gpt2-xl
# DATA_TEST=ZsRE-test-all
# DATA_TYPE=zsre
DATA_TEST=recent_test
DATA_TRAIN=recent_train
DATA_TYPE=recent
NUM=1

CUDA_VISIBLE_DEVICES=1 nohup python run_knowedit_llama2.py \
    --editing_method $EDIT_METHOD \
    --hparams_dir ../hparams/$EDIT_METHOD/$MODEL \
    --data_dir ../dataset/round1_dataset/$DATA_TEST.json \
    --datatype $DATA_TYPE \
    > log/$DATE/$EDIT_METHOD-$MODEL-$DATA_TEST-$NUM.log 2>&1 &
