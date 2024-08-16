#!/bin/bash
#SBATCH --gpus=1
export PYTHONUNBUFFERED=1
DATE=$(date +%Y-%m-%d)
if [ ! -d "logs/$DATE" ]; then
    mkdir -p "logs/$DATE"
fi

module load compilers/cuda/12.1
module load cudnn/8.9.5.29_cuda12.x
module load compilers/gcc/12.2.0
source activate ke2torch23cu121

python knb_edit.py > logs/$DATE/knb-edit-llama2-max-bs60-steps50-1.log 2>&1