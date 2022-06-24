#!/bin/sh

# export PYTHONHASHSEED=0
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch17

python ../train_surrogate.py \
    --transfer_data $TRANSFER_DATA \
    --eval_data $EVAL_DATA \
    --save_path $SAVE_PATH \
    --num_gpus 1 \
    --early_stopping_epoch $EARLY_STOPPING_EPOCH \