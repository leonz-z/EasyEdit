DATE=$(date +"%Y-%m-%d")
mkdir -p logs/$DATE
source activate ke2torch23cu121
export CUDA_VISIBLE_DEVICES=0
export HUGGINGFACE_CACHE=/share/huggingface/
python examples/run_CKnowEdit_qwen-1.8B.py \
    --editing_method LoRA \
    > logs/$DATE/LoRA_Qwen-1_8B-Chat-1.log 2>&1 &

# target_r (`int`): The target average rank of incremental matrix.
# init_r (`int`): The initial rank for each incremental matrix.
# tinit (`int`): The steps of initial fine-tuning warmup.
# tfinal (`int`): The step of final fine-tuning.
# deltaT (`int`): The time internval between two budget allocations.
# beta1 (`float`): The hyperparameter of EMA for sensitivity smoothing.
# beta2 (`float`): The hyperparameter of EMA for undertainty quantification.
# orth_reg_weight (`float`): The coefficient of orthogonal regularization.
# total_step (`int`): The total training steps that should be specified before training.
# rank_pattern (`list`): The allocated rank for each weight matrix by RankAllocator.