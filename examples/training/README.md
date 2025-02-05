# Training Details

In addition to code to reproduce the evaluations of ProCyon, we also include the scripts and code needed to pretrain the model. 
Details are included below, along with pointers to example config files that are essential for pretraining.

ProCyon is a large multimodal foundation model, and thus requires a large amount of computational resources to train.
For example, the ProCyon-Split model was trained on 48 GPUs simultaneously while the ProCyon-Full model was trained on 32 GPUs.
We provide our codebase which scales to this number of GPUs through the use of the [`deepspeed`](https://github.com/deepspeedai/DeepSpeed) library.
We recommend using equivalent resources to train our model, as less GPUs will result in smaller batch sizes, which will harm gradient robustness for both generative language modeling and contrastive learning of the model.

The code provided here is a demonstration of what is needed to train, and it does not reflect the full process of training ProCyon, which involved careful observation of training curves and monitoring of performance throughout.
Our process is described in the methods of the [preprint](https://www.biorxiv.org/content/10.1101/2024.12.10.627665v1).

**If you would like to retrain our model, please reach out to [Owen](mailto:oqueen@stanford.edu) or [Robert](mailto:rcalef@mit.edu) to discuss the resources required.**

## Pretraining ProCyon-Full

ProCyon-Full training makes use of the config files:
1. Primary config: [`configs/llama3-full.yml`]()
2. Data config: [`configs/data_configs/all_datasets_pretrain_full.yml`](https://github.com/mims-harvard/ProCyon/blob/main/configs/data_configs/all_datasets_pretrain_full.yml)
3. Deepspeed config: [`configs/deepspeed/full_train_ds.json`](https://github.com/mims-harvard/ProCyon/blob/main/configs/deepspeed/full_train_ds.json)

An example SLURM file for launching the training job on the Kempner Institute cluster is included at `examples/training/procyon_full_pretrain.sh`.

## Fine-tuning for ProCyon-Bind
