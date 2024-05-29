# MEND未适配到safe edit?
DATE=2024年5月29日
EDIT_METHOD=MEND
MODEL=Qwen-7B
DATA=SafeEdit_test
NUM=1

CUDA_VISIBLE_DEVICES=1 nohup python run_safety_editing.py \
    --editing_method $EDIT_METHOD \
    --edited_model $MODEL \
    --hparams_dir ../hparams/$EDIT_METHOD/$MODEL.yaml \
    --safety_classifier_dir /share/huggingface/DINM-Safety-Classifier \
    --data_dir ../dataset/round1_dataset/$DATA.json \
    --metrics_save_dir ./safety_results_test \
    > log/$DATE/$EDIT_METHOD-$MODEL-$DATA-$NUM.log 2>&1 &