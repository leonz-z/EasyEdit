# module load compilers/cuda/12.1
# module load cudnn/8.9.5.29_cuda12.x
# module load compilers/gcc/12.2.0
# source activate ke2torch23cu121
# export HUGGINGFACE_CACHE=/home/bingxing2/home/scx7avs/lyc/huggingface/
export CUDA_VISIBLE_DEVICES=0

source activate ke2torch23cu121
export HUGGINGFACE_CACHE=/share/huggingface/

python test_answer.py