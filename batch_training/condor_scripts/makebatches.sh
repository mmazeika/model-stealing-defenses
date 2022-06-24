#!/bin/sh

cd your-path-here/model-stealing-defenses/batch_training/condor_scripts

# ==================== training teachers ==================== #
misinformation=0
for dataset in cifar10 cifar100 cub200
do
  exp_name="${dataset}"
  save_path="your-path-here/model-stealing-defenses/batch_training/outputs/trained_models/${dataset}.pt"
  condor_submit train_teacher.sub -batch-name $exp_name "EXP_NAME=${exp_name}" "DATASET=${dataset}" "SAVE_PATH=${save_path}" "MISINFORMATION=${misinformation}"
done

misinformation=1
for dataset in cifar10 cifar100 cub200
do
  exp_name="${dataset}_misinformation"
  save_path="your-path-here/model-stealing-defenses/batch_training/outputs/trained_models/${dataset}_misinformation.pt"
  condor_submit train_teacher.sub -batch-name $exp_name "EXP_NAME=${exp_name}" "DATASET=${dataset}" "SAVE_PATH=${save_path}" "MISINFORMATION=${misinformation}"
done

# wait for teachers to finish training
while [ ! -f your-path-here/model-stealing-defenses/batch_training/outputs/trained_models/cifar10_1_teacher.pt ]; do sleep 10; done
while [ ! -f your-path-here/model-stealing-defenses/batch_training/outputs/trained_models/cifar100_1_teacher.pt ]; do sleep 10; done

echo "Finished training teachers"

# ==================== training surrogates ==================== #
for eval_data in cifar10 cifar100 cub200
do
  if [ "$eval_data" = "cub200" ]; then
    transfer_data=imagenet_cub200
    # transfer_data=caltech256
  fi
  if [ "$eval_data" = "cifar10" ]; then
    transfer_data=imagenet_cifar10
    # transfer_data=cifar100
  fi
  if [ "$eval_data" = "cifar100" ]; then
    transfer_data=imagenet_cifar100
    # transfer_data=cifar10
  fi
  if [ "$eval_data" = "mnist" ]; then
    transfer_data=fashionmnist
  fi
  if [ "$eval_data" = "fashionmnist" ]; then
    transfer_data=mnist
  fi

  for idx in 0 1 2 3 4 5
  do
    ese=$((10*idx))
    exp_name="${transfer_data}_to_${eval_data}_${ese}epochs"
    save_path="your-path-here/model-stealing-defenses/batch_training/outputs/trained_models/${transfer_data}_to_${eval_data}_surrogate_${ese}epochs.pt"
    condor_submit train_surrogate.sub -batch-name $exp_name "EXP_NAME=${exp_name}" "TRANSFER_DATA=${transfer_data}" "EVAL_DATA=${eval_data}" "SAVE_PATH=${save_path}" "EARLY_STOPPING_EPOCH=${ese}"
  done
done

# wait for surrogates to finish training
while [ ! -f your-path-here/model-stealing-defenses/batch_training/outputs/trained_models/cifar10_1_to_cifar10_surrogate_40epochs.pt ]; do sleep 10; done
while [ ! -f your-path-here/model-stealing-defenses/batch_training/outputs/trained_models/cifar100_1_to_cifar100_surrogate_40epochs.pt ]; do sleep 10; done

echo "Finished training surrogates"

# ==================== generating perturbed posteriors ==================== #
for eval_data in cifar10 cifar100 cub200
do
  if [ "$eval_data" = "cub200" ]; then
    transfer_data=imagenet_cub200
    # transfer_data=caltech256
  fi
  if [ "$eval_data" = "cifar10" ]; then
    # transfer_data=imagenet_cifar10
    transfer_data=cifar100
  fi
  if [ "$eval_data" = "cifar100" ]; then
    # transfer_data=imagenet_cifar100
    transfer_data=cifar10
  fi
  if [ "$eval_data" = "mnist" ]; then
    transfer_data=fashionmnist
  fi
  if [ "$eval_data" = "fashionmnist" ]; then
    transfer_data=mnist
  fi

  for defense in ALL-ONES_10 MIN-IP_10 MAD AdaptiveMisinformation ReverseSigmoid Random None # add other defenses as needed (ALL-ONES_10 is the main GRAD^2 setting)
  do
    epsilons="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
    if [ "$defense" = "None" ]; then
      epsilons="0.0"
    fi
    if [ "$defense" = "ReverseSigmoid" ]; then
      epsilons="0.0 0.0025 0.005 0.01 0.02 0.04 0.08 0.16 0.32 0.64 1.28" #  beta parameters (named epsilon for convenience)
    fi
    if [ "$defense" = "AdaptiveMisinformation" ]; then
      epsilons="0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0"  # tau parameters (named epsilon for convenience)
    fi

    exp_name="${transfer_data}_to_${eval_data}_${defense}"
    save_path="your-path-here/model-stealing-defenses/batch_training/outputs/generated_perturbations/${transfer_data}_to_${eval_data}_${defense}.pkl"
    condor_submit get_queries.sub -batch-name $exp_name "EXP_NAME=${exp_name}" "TRANSFER_DATA=${transfer_data}" "EVAL_DATA=${eval_data}" \
      "EPSILONS=${epsilons}" "DEFENSE=${defense}" "SAVE_PATH=${save_path}"
  done
done

# wait for MAD perturbations to finish
# while [ ! -f your-path-here/model-stealing-defenses/batch_training/outputs/generated_perturbations/cifar10_to_cifar10_MAX-NORM.pkl ]; do sleep 10; done
# while [ ! -f your-path-here/model-stealing-defenses/batch_training/outputs/generated_perturbations/cifar100_to_cifar100_MAX-NORM.pkl ]; do sleep 10; done
sleep 14400  # wait 4 hours
echo "Probably finished generating posteriors"

# ==================== training adversaries on perturbed posteriors ==================== #

oracle="None"  # Unused in the experiments in the paper; can be ignored (generates perturbations in each batch using the adversary's true parameters instead of a surrogate)

for eval_data in cifar10 cifar100 cub200
do
  if [ "$eval_data" = "cub200" ]; then
    transfer_data=imagenet_cub200
    # transfer_data=caltech256
  fi
  if [ "$eval_data" = "cifar10" ]; then
    # transfer_data=imagenet_cifar10
    transfer_data=cifar100
  fi
  if [ "$eval_data" = "cifar100" ]; then
    # transfer_data=imagenet_cifar100
    transfer_data=cifar10
  fi
  if [ "$eval_data" = "mnist" ]; then
    transfer_data=fashionmnist
  fi
  if [ "$eval_data" = "fashionmnist" ]; then
    transfer_data=mnist
  fi
    
  for defense in Random_vgg
  do
    load_path="your-path-here/model-stealing-defenses/batch_training/outputs/generated_perturbations/${transfer_data}_to_${eval_data}_${defense}.pkl"

    epsilons="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
    if [ "$defense" = "None" ]; then
      epsilons="0.0"
    fi
    if [ "$defense" = "None_argmax" ]; then
      epsilons="0.0"
    fi
    if [ "$defense" = "None_vgg" ]; then
      epsilons="0.0"
    fi
    if [ "$defense" = "ReverseSigmoid" ]; then
      epsilons="0.0 0.0025 0.005 0.01 0.02 0.04 0.08 0.16 0.32 0.64 1.28"
    fi
    if [ "$defense" = "AdaptiveMisinformation" ]; then
      epsilons="0.0"  # tau parameters (named epsilon for convenience)
    fi
    for epsilon in $epsilons
    do
      exp_name="${transfer_data}_to_${eval_data}_${defense}_eps${epsilon}"
      save_path="your-path-here/model-stealing-defenses/batch_training/outputs/trained_models/${transfer_data}_to_${eval_data}_${defense}_eps${epsilon}.pt"
      condor_submit train_adversary.sub -batch-name $exp_name "EXP_NAME=${exp_name}" "TRANSFER_DATA=${transfer_data}" "EVAL_DATA=${eval_data}" \
        "LOAD_PATH=${load_path}" "EPSILON=${epsilon}" "SAVE_PATH=${save_path}" "ORACLE=${oracle}"
    done
  done
done

# echo "Done submitting jobs!"