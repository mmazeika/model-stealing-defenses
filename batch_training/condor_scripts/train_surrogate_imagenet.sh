#!/bin/sh

# export PYTHONHASHSEED=0
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch19_a40

python ../train_surrogate_imagenet.py \
    --dataset imagenet \
    --save_path your-path-here/model-stealing-defenses/batch_training/outputs/trained_models/imagenet_surrogate_80epochs.pt \
    --num_gpus 4 \
    --early_stopping_epoch 80 \
