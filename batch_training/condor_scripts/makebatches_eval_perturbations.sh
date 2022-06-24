#!/bin/sh

cd your-path-here/model-stealing-defenses/batch_training/condor_scripts


# ==================== generating perturbed posteriors ==================== #
for eval_data in cifar10 cifar100 cub200
do
  if [ "$eval_data" = "cub200" ]; then
    # transfer_data=imagenet_cub200
    transfer_data=caltech256
  fi
  if [ "$eval_data" = "cifar10" ]; then
    # transfer_data=imagenet_cifar10
    transfer_data=cifar100
  fi
  if [ "$eval_data" = "cifar100" ]; then
    # transfer_data=imagenet_cifar100
    transfer_data=cifar10
  fi

  for defense in ALL-ONES_0_s1 ALL-ONES_10_s1 ALL-ONES_20_s1 ALL-ONES_30_s1 ALL-ONES_40_s1  # add other defenses as needed
  do
    epsilons="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
    if [ "$defense" = "None" ]; then
      epsilons="0.0"
    fi
    if [ "$defense" = "ReverseSigmoid" ]; then
      epsilons="0.0 0.0025 0.005 0.01 0.02 0.04 0.08 0.16 0.32 0.64 1.28"
    fi
    if [ "$defense" = "AdaptiveMisinformation" ]; then
      epsilons="0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0"  # tau parameters (named epsilon for convenience)
    fi

    exp_name="${transfer_data}_to_${eval_data}_${defense}_val"
    save_path="your-path-here/model-stealing-defenses/batch_training/outputs/generated_perturbations/${transfer_data}_to_${eval_data}_${defense}_val.pkl"
    condor_submit get_queries_val.sub -batch-name $exp_name "EXP_NAME=${exp_name}" "TRANSFER_DATA=${transfer_data}" "EVAL_DATA=${eval_data}" \
      "EPSILONS=${epsilons}" "DEFENSE=${defense}" "SAVE_PATH=${save_path}"
  done
done