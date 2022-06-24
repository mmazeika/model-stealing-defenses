#!/bin/sh

# export PYTHONHASHSEED=0
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch17

python ../get_queries.py \
    --transfer_data $TRANSFER_DATA \
    --eval_data $EVAL_DATA \
    --defense $DEFENSE \
    --epsilons "${EPSILONS}" \
    --save_path $SAVE_PATH \
    --eval_perturbations