#!/bin/sh

# export PYTHONHASHSEED=0
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch17

python ../train_teacher.py \
    --dataset $DATASET \
    --save_path $SAVE_PATH \
    --num_gpus 1 \
    --misinformation $MISINFORMATION \