# Model Stealing Defenses with Gradient Redirection
This is the official repository for "[How to Steer Your Adversary: Targeted and Efficient Model Stealing Defenses with Gradient Redirection](https://arxiv.org/abs/2206.14157)" (ICML 2022)

## How To Use
First, clone the repository, then clone the [Outlier Exposure repository](https://github.com/hendrycks/outlier-exposure) into the model-stealing-defenses folder. This is used to provide a strong anomaly detector for the Adaptive Misinformation baseline on CIFAR experiments. Next, follow the instructions in batch\_training/condor\_scripts/data/README.md to setup the distribution-aware datasets. Optionally download and untar the outputs and condor\_outputs folders from [here](https://drive.google.com/drive/folders/1uzv_i2v8RedPYh6r06dWYBKFJ94zdxc-?usp=share_link), replacing the empty folders by the same name with the respective untarred folders. These contain perturbed posteriors and trained models, which can be used to replicate results from the paper, but this requires around 60GB of space.

The GRAD<sup>2</sup> method from the paper can be run using the models currently in the outputs folder. The functions for running GRAD<sup>2</sup> are in defenses.py, and example usage from the experiments in the paper is in get_queries.py and makebatches.sh.

To regenerate results from the paper, rerun the experiments in makebatches.sh in the specified order. The experiments were run on an HTCondor system, so the script would need to be adjusted for slurm. Results and figures can be generated in batch\_training/condor\_scripts/parse\_results.ipynb using either the regenerated results or the results in outputs.tar (see download link above).

## Citation

If you find this useful in your research, please consider citing:

    @article{mazeika2022defense,
      title={How to Steer Your Adversary: Targeted and Efficient Model Stealing Defenses with Gradient Redirection},
      author={Mazeika, Mantas and Li, Bo and Forsyth, David},
      journal={Proceedings of the International Conference on Machine Learning},
      year={2022}
    }
