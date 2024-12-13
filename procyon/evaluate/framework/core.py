import os
import sys

from collections import defaultdict
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from procyon.training.training_args_IT import (
    DataArgs,
    ModelArgs,
    update_data_args_data_dir,
)

from procyon.evaluate.framework.args import EvalArgs
from procyon.evaluate.framework.caption import run_caption_eval
from procyon.evaluate.framework.qa import run_qa_eval
from procyon.evaluate.framework.retrieval import run_retrieval_eval
from procyon.evaluate.framework.utils import (
    load_and_validate_model_args,
    load_datasets_for_eval,
    override_data_and_model_args,
    write_metrics,
)

# Model imports
from procyon.evaluate.framework.biotranslator import BioTranslatorRetrievalEval
from procyon.evaluate.framework.blast import BlastRetrievalEval
from procyon.evaluate.framework.knn import (
    BlastKnnRetrievalEval,
    BlastKnnQAEval,
    ESMKnnRetrievalEval,
    ESMKnnQAEval,
    ESM3KnnRetrievalEval,
    ESM3KnnQAEval,
    GearNetKnnRetrievalEval,
    GearNetKnnQAEval,
)
from procyon.evaluate.framework.mlp import (
    ESMMLPRetrievalEval,
    ESMMLPQAEval,
    ESM3MLPRetrievalEval,
    ESM3MLPQAEval,
    GearNetMLPRetrievalEval,
    GearNetMLPQAEval,
)

if sys.version_info[1] <= 10:
    from procyon.evaluate.framework.protst import ProtSTRetrievalEval

from procyon.evaluate.framework.procyon import (
    ProcyonCaptionEval,
    ProcyonRetrievalEval,
    ProcyonQAEval,
)
from procyon.evaluate.framework.random import (
    UniformRandomCaptionEval,
    WeightedRandomCaptionEval,
    MajorityRuleRandomCaptionEval,
    UniformRandomRetrievalEval,
    WeightedRandomRetrievalEval,
    MajorityRuleRandomRetrievalEval,
)

# from procyon.evaluate.framework.ProtLLMQA import ProtLLMQAEval

caption_models = {
    "ProCyon": ProcyonCaptionEval,
    "UniformRandom": UniformRandomCaptionEval,
    "WeightedRandom": WeightedRandomCaptionEval,
    "MajorityRule": MajorityRuleRandomCaptionEval,
}

retrieval_models = {
    "BLAST": BlastRetrievalEval,
    "BioTranslator": BioTranslatorRetrievalEval,
    "ProCyon": ProcyonRetrievalEval,
    "MajorityRule": MajorityRuleRandomRetrievalEval,
    "UniformRandom": UniformRandomRetrievalEval,
    "WeightedRandom": WeightedRandomRetrievalEval,
    "BlastKnn": BlastKnnRetrievalEval,
    "ESMKnn": ESMKnnRetrievalEval,
    "ESM3Knn": ESM3KnnRetrievalEval,
    "GearNetKnn": GearNetKnnRetrievalEval,
    "ESMMLP": ESMMLPRetrievalEval,
    "ESM3MLP": ESM3MLPRetrievalEval,
    "GearNetMLP": GearNetMLPRetrievalEval,
}

if sys.version_info[1] <= 10:
    retrieval_models["ProtST"] = ProtSTRetrievalEval

qa_models = {
    "ProCyon": ProcyonQAEval,
    "BlastKnn": BlastKnnQAEval,
    "ESMKnn": ESMKnnQAEval,
    "ESM3Knn": ESM3KnnQAEval,
    "GearNetKnn": GearNetKnnQAEval,
    "ESMMLP": ESMMLPQAEval,
    "ESM3MLP": ESM3MLPQAEval,
    "GearNetMLP": GearNetMLPQAEval,
    #    "ProtLLM": ProtLLMQAEval,
}

model_zoo = {
    "caption": caption_models,
    "retrieval": retrieval_models,
    "qa": qa_models,
}

task_evals = {
    "caption": run_caption_eval,
    "retrieval": run_retrieval_eval,
    "qa": run_qa_eval,
}


def run_evaluation(
    eval_args: EvalArgs,
    data_args: DataArgs,
    model_args: ModelArgs,
):
    """
    Primary entrypoint to running evaluation across tasks, models, and datasets.

    eval_args  - As defined in `./args.py`.
    data_args  - procyon.training.training_args_IT.DataArgs
    model_args - procyon.training.training_args_IT.ModelArgs
    """
    # Overall steps:
    #  1. Load all datasets from config (where each dataset may be
    #     overriding some subset of DataArgs).
    #  2. Run each dataset through given model for associated tasks.
    #  3. Output evaluation metrics per task and per dataset.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if we want to override ModelArgs using a ProCyon checkpoint
    if eval_args.model_args_from_checkpoint != "":
        checkpoint_dir = eval_args.model_args_from_checkpoint
        print(f"Loading ModelArgs from ProCyon checkpoint: {checkpoint_dir}")
        model_args = torch.load(os.path.join(checkpoint_dir, "model_args.pt"))
        model_args = ModelArgs(**asdict(model_args))

    if eval_args.data_args_from_checkpoint != "":
        checkpoint_dir = eval_args.data_args_from_checkpoint
        print(f"Loading DataArgs from ProCyon checkpoint: {checkpoint_dir}")
        loaded_data_args = torch.load(os.path.join(checkpoint_dir, "data_args.pt"))
        loaded_data_args = DataArgs(**asdict(loaded_data_args))

        update_data_args_data_dir(loaded_data_args)

        # Prefer to use the data config specified in data_args passed into this function
        # over one specified in the serialized data config.
        if data_args.it_data_config_yml is not None:
            loaded_data_args.it_data_config_yml = data_args.it_data_config_yml
        data_args = loaded_data_args

    # Check if we want to override any of the DataArgs or ModelArgs values parsed
    # from the model checkpoint.
    if eval_args.override_model_data_args_yml is not None:
        override_data_and_model_args(
            data_args, model_args, eval_args.override_model_data_args_yml
        )

    # Parse model specifications.
    models = load_and_validate_model_args(eval_args.models_config_yml)

    # Load datasets
    datasets, collators, dataset_eval_args = load_datasets_for_eval(
        data_args,
        model_args,
        eval_args.separate_splits,
        eval_args.keep_splits_union,
    )
    for task, train_datasets in datasets["train"].items():
        if len(train_datasets) != 0:
            print(
                f"Received training datasets for task {task}, will not be used for evaluation (check data config)"
            )
    # Package datasets and collators into data loaders.
    data_loaders = {}
    datasets = datasets["testing"]
    collators = collators["testing"]
    for task, task_datasets in datasets.items():
        task_loaders = {}
        for dataset_key, dataset in task_datasets.items():
            print(f"task: {task} dataset: {dataset_key} N: {len(dataset)}")
            task_loaders[dataset_key] = DataLoader(
                dataset,
                batch_size=eval_args.batch_size,
                collate_fn=collators[task][dataset_key],
                num_workers=eval_args.num_workers,
                pin_memory=True,
                drop_last=False,
            )
        data_loaders[task] = task_loaders

    # Begin evaluation
    all_results = {}
    for task, task_loaders in data_loaders.items():
        print(f"{task}: evaluating on {len(task_loaders)} datasets")
        task_results = defaultdict(dict)
        if task not in task_evals:
            raise ValueError(f"unknown task {task}")
        eval_func = task_evals[task]

        for model_name, args in models.items():
            if model_name.lower() == "protst":
                assert (
                    sys.version_info[1] <= 10
                ), "ProtST not compatible with Python >3.10, please change version"

            model = model_zoo[task][model_name](args, eval_args, model_args, device)

            model_results_dir = os.path.join(eval_args.output_dir, task, model_name)

            for dataset_key, data_loader in task_loaders.items():
                this_dataset_eval_args = dataset_eval_args[dataset_key]

                dataset_results_dir = os.path.join(model_results_dir, dataset_key)
                os.makedirs(dataset_results_dir, exist_ok=True)

                task_results[model_name][dataset_key] = eval_func(
                    model,
                    data_loader,
                    eval_args,
                    this_dataset_eval_args,
                    model_name,
                    dataset_key,
                    dataset_results_dir,
                )

                # NOTE: moved inside here, will intermediately write metrics for fault tolerance
                all_results[task] = task_results
                write_metrics(all_results, eval_args.output_dir)

        all_results[task] = task_results

    write_metrics(all_results, eval_args.output_dir)
    return all_results
