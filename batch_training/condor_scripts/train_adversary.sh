#!/bin/sh

# export PYTHONHASHSEED=0
source ~/anaconda3/etc/profile.d/conda.sh

if [ "$(uname -n)" == "vision-14.cs.illinois.edu" ] || [ "$(uname -n)" == "vision-15.cs.illinois.edu" ] || \
   [ "$(uname -n)" == "vision-17.cs.illinois.edu" ] || [ "$(uname -n)" == "vision-19.cs.illinois.edu" ] || \
   [ "$(uname -n)" == "vision-20.cs.illinois.edu" ]; then
  conda activate pytorch_a40_py37 # if on K40m, use pytorch built on a K40m machine
elif [ "$(uname -n)" == "vision-21.cs.illinois.edu" ]; then
  conda activate pytorch_a40_py37 # if on A40, use cuda 11.1
else
  conda activate pytorch_a40_py37 # actually pytorch 1.10 rn; not sure about cuda version though
fi

python ../train_adversary.py \
    --transfer_data $TRANSFER_DATA \
    --eval_data $EVAL_DATA \
    --load_path $LOAD_PATH \
    --epsilon $EPSILON \
    --save_path $SAVE_PATH \
    --num_gpus 1 \
    --oracle $ORACLE \
