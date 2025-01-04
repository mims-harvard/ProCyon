# Running ProCyon evaluations

TL;DR: refer to the example configuration files and run script [here](https://github.com/mims-harvard/ProCyon/blob/main/examples/evaluation/) to get started running model
evaluations. See below for a more thorough explanation of the evaluation framework and how to
configure runs.

The evaluation framework defined here is used for running systematic performance
evaluations of ProCyon models and other baselines across three task types (retrieval,
question-answering, and captioning). The core tenents of the framework are that
evaluations runs should be:
- Configurable: easy to add additional models or datasets to an evaluation run, enabling smooth transition from small testing runs to systematic benchmarking runs
- Reproducible: each run with the same config should produce the same results
- Rigorous: models are treated as black boxes, providing each with the exact same test sets and metrics are computed identically
- Extensible: evaluations use a standardized API for each task, allowing for integration of additonal models by implementing a wrapper class

## Configuration
An evaluation run is configured by providing three YAML files representing the configurations
of the three core parts of an evaluation run:

- Models
- Datasets
- Evaluation parameters

A single evaluation run will generate performance metrics for all specified models
across all specified datasets.

### Models
We have implemented multiple models within our framework, allowing for comparison
of ProCyon models, naive baselines, small models operating on common protein
representation learning methods, and third-party multi-modal protein models.

The available models can be seen [here](https://github.com/mims-harvard/ProCyon/blob/main/procyon/evaluate/framework/core.py#L68).

Models are specified in a YAML file, where each entry specifies a single model
to evaluate along with any additional model-specific configuration parameters. A
full example model config can be seen
[here](https://github.com/mims-harvard/ProCyon/blob/main/examples/evaluation/model_config.yml),
but here's a short example:
```
models:
        - model_name: ProCyon
          args:
                  checkpoint_dir: /path/to/model_checkpoints/ProCyon-Split
        - model_name: ProtST
          args:
                  max_prompt_len: 128
        - model_name: ESM3MLP
          args:
                  filter_zero_shot: True
                  num_steps: 2000
                  num_steps_no_validation: 300
                  validation_steps: 50
                  hidden_dim: 256
                  pos_weight: 1000
                  checkpoint_dir: /path/to/model_checkpoints/trained_mlps
        - model_name: GearNetMLP
```
the allowable `args` for each model come directly from the corresponding model's `__init__`
arguments.

## Datasets
The dataset config is an additional YAML file that specifies which datasets should be used for
evaluation. Currently, the supported datasets are those that are available in our ProCyon-Instruct
dataset. For more details on the structure of the dataset itself, please see the [HuggingFace dataset page](https://huggingface.co/datasets/mims-harvard/ProCyon-Instruct) for the dataset.

A full example config can be seen [here](https://github.com/mims-harvard/ProCyon/blob/main/examples/evaluation/dataset_config.yml), but here's a short example:
```
it_datasets:
  testing:
    # GO:
    - aaseq_type: protein
      text_type: go
      relations: [process, function, component]
      tasks: [retrieval, qa, caption]
      splits: [EVAL:pt_ft, EVAL:few_shot, EVAL:zero_shot]
      split_method: sample_aware_ontology_go_centric
    # Reactome
    - aaseq_type: protein
      text_type: reactome
      relations: [all]
      tasks: [retrieval, qa, caption]
      splits: [EVAL:pt_ft, EVAL:few_shot, EVAL:zero_shot]
      split_method: random_reactome_centric
    # DrugBank:
    - aaseq_type: protein
      text_type: drugbank
      relations: [drug_target, drug_enzyme, drug_carrier, drug_transporter]
      tasks: [retrieval, qa, caption]
      splits: [EVAL:pt_ft, EVAL:few_shot, EVAL:zero_shot]
      split_method: atc_aware_drugbank_centric
```
Each entry in the YAML file corresponds to a single dataset to evaluate on, and specifies the
following parameters:
- aaseq_type: what type of amino acid sequence these phenotypes are defined for. Typically `protein`, but can also be `domain`.
- text_type: what is the source database for the phenotypes in this dataset
- relations: some phenotype databases will have multiple types of phenotypes, e.g. Gene Ontology specifies Molecular Function, Cellular Component, and Biological Process. This parameter can be used to select a subset of relations, or the keyword `all` can be used to use all relations, or in the case when the dataset consists of a single relation type
- tasks: what type of tasks this dataset should be used for evaluating. Currently implemented tasks are `retrieval`, `qa`, and `caption`. Note that not all models are capable of performing all tasks, so one needs to ensure that the models specified in the model config are compatible with the set of specified tasks.
- splits: what dataset splits to use for evaluation. Options are some number of:
  - `CL_train`: training set used for training `ProCyon-Split`
  - `EVAL:pt_ft`: evaluation set using unseen protein-phenotype associations, but where the phenotype has been seen frequently during training
  - `EVAL:few_shot`: evaluation set using unseen protein-phenotype associations, but where the phenotype has been seen rarely during training
  - `EVAL:zero_shot`: evaluation set using unseen protein-phenotype associations, where the phenotype was never seen  during training. Note that not all models support prediction of zero-shot phenotypes.
- split_method: specifies the method used for generating the dataset splits. The current version of `ProCyon-Instruct` consists only of a single splitting method per dataset.

### Evaluation parameters
The final YAML file specifies parameters that affect how the evaluation run itself is performed.
These are parameters controlling things like how to compute metrics, what batch sizes to use for
inference, and where to save results. The arguments specified here are parsed directly into the
`EvalArgs` class specified [here](https://github.com/mims-harvard/ProCyon/blob/main/procyon/evaluate/framework/args.py#L6).

A full example can be found with the other example configs [here](https://github.com/mims-harvard/ProCyon/blob/main/examples/evaluation/eval_config.yml), but generally looks like keyword
arguments specified via YAML:

```
it_data_config_yml: /path/to/ProCyon/examples/evaluation/dataset_config.yml
models_config_yml: /path/to/ProCyon/examples/evaluation/model_config.yml
output_dir: /path/to/desired_output/

filter_training_pairs: True
separate_splits: True
keep_splits_union: False

retrieval_eval_all_aaseqs: True
retrieval_top_k_vals: [10, 20, 100]
```
Note that this is also where the paths to the model and dataset configs are specified.

## Running evaluations
Given the config files specified above, running evaluations is simple:
```
python /path/to/ProCyon/scripts/run_eval_framework.py \
        --from_yaml eval_args.yml
```
This will kick off the eval run, generating directories and results per task and model. The final
result directory will contain one TSV per task type, giving the performance of each model across
each dataset, with the exact metrics varying by task type.
For more details on how performance metrics are computed per task, please refer to the Methods
section of our publication.