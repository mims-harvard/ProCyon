import os
import yaml

from collections import defaultdict
from dataclasses import fields
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import torch

import numpy as np
import pandas as pd

from esm.data import Alphabet
from scipy.stats import bootstrap
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from procyon.data.dataset import (
    AASeqDataset,
    AASeqTextUnifiedDataset,
    get_and_check_relation_id,
)
from procyon.data.it_data_config import (
    expand_datasets_on_splits,
    ITMultiDatasetConfig,
)
from procyon.training.training_args_IT import (
    DataArgs,
    ModelArgs
)

from procyon.data.dataset import get_and_check_relation_id

def sum_dicts(lhs: Dict, rhs: Dict):
    for k,v in rhs.items():
        lhs[k] += v

def move_inputs_to_device(
    data: Union[torch.Tensor, Any],
    device: torch.device,
) -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: move_inputs_to_device(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(move_inputs_to_device(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device=device)
    return data

def calc_bootstrap_bounds(
    metric_samples: Dict,
    num_bootstraps: int = 9999,
    ci: float = 0.95,
    batch_size: int = 10000,
    statistic: Callable = np.mean,
    seed: int = 42,
    paired: bool = False,
) -> Dict:
    if not paired:
        for k, v in metric_samples.items():
            metric_samples[k] = np.array(v)

    rng = np.random.default_rng(seed=seed)
    bounds = {}
    for metric_name, samples in metric_samples.items():
        if paired:
            x, y = zip(*samples)
            inputs = (x, y)
        else:
            inputs = (samples, )
        res = bootstrap(
            inputs,
            statistic=statistic,
            confidence_level=ci,
            n_resamples=num_bootstraps,
            batch=batch_size,
            random_state=rng,
            paired=paired,
        )
        bounds[f"{metric_name}_lb"] = res.confidence_interval.low
        bounds[f"{metric_name}_ub"] = res.confidence_interval.high

    return bounds

def compare_and_warn_model_args(
    model_args_a: ModelArgs,
    model_args_b: ModelArgs,
) -> None:
    """Compare two sets of ModelArgs to check for differences.

    ModelArgs also contains some parameters used for dataset loading and
    collating, so having a mismatch between what's specified for evaluation
    and what was specified when training a TxPLM model can create odd crashes
    and unexpected behavior. We want to warn if the two sets of ModelArgs
    mismatch, but also want to ignore some fields that are expected to be
    different.
    """
    # Fields to ignore as they get modified while loading pretrained model.
    # If you're seeing a lot of warnings, it's possible a new field needs to
    # be added here.
    ignore_fields = ["n_model_pieces", "model_splitting"]

    mismatches = []
    for field in fields(ModelArgs):
        if field.name in ignore_fields:
            continue
        # Ignore paths as they may have been updated to a different DATA_DIR
        if field.name.endswith("path"):
            continue

        field_a = getattr(model_args_a, field.name)
        field_b = getattr(model_args_b, field.name)
        if field_a != field_b:
            mismatches.append((field.name, field_a, field_b))

    if len(mismatches) != 0:
        mismatches_print = [f"{x[0]}: {x[1]} != {x[2]}" for x in mismatches]
        print(
            "Specified ModelArgs do not match those used in provided TxPLM checkpoint "
            "this may cause crashes or unexpected behavior. Use the EvalArgs.model_args_from_checkpoint "
            "command-line argument unless you're sure you want to do this, comment this out.\n"
            f"Mismatched fields: {' , '.join(mismatches_print)}"
        )

def override_data_and_model_args(
    data_args: DataArgs,
    model_args: ModelArgs,
    override_yml_path: str,
) -> None:
    """Parse a separate yaml for overriding a subset of DataArgs and ModelArgs params."""
    with open(override_yml_path, "r") as fh:
        raw_args = yaml.safe_load(fh)
    for name, val in raw_args.items():
        if hasattr(data_args, name):
            old_val = getattr(data_args, name)
            setattr(data_args, name, val)
            print(f"overriding DataArg: {name}: {old_val} -> {val}")
        if hasattr(model_args, name):
            old_val = getattr(model_args, name)
            setattr(model_args, name, val)
            print(f"overriding DataArg: {name}: {old_val} -> {val}")

def load_datasets_for_eval(
    data_args: DataArgs,
    model_args: ModelArgs,
    separate_splits: bool,
    keep_splits_union: bool,
) -> Tuple[Dict, Dict, Dict]:
    """Get eval datasets from arguments.

    Uses ITMultiDatasetConfig as specified in data_args to get datasets,
    collators, and per-dataset evaluation args.
    """
    protein_tokenizer = Alphabet.from_architecture(model_args.protein_tokenizer_name)

    text_tokenizer = AutoTokenizer.from_pretrained(
        data_args.data_dir + f"/model_weights/{model_args.text_encoder_fname}"
    )
    text_tokenizer.text_tokenizer_name = (
        model_args.text_tokenizer_name
    )  # used to determine how to preprocess sequences

    config = ITMultiDatasetConfig.load_from_yaml(data_args.it_data_config_yml)
    if separate_splits:
        config.testing_datasets = expand_datasets_on_splits(config.testing_datasets, keep_splits_union)

    it_datasets, it_collators = config.get_datasets_and_collators(
        data_args,
        model_args,
        protein_tokenizer,
        text_tokenizer,
        evaluation=True,
        deduplicate_dataset=True,
    )
    dataset_eval_args = config.get_eval_args_by_dataset()
    return it_datasets, it_collators, dataset_eval_args

def load_eval_data_loaders(
    data_args: DataArgs,
    model_args: ModelArgs,
    batch_size: int = 8,
    num_workers: int = 1,
) -> Dict[str, Dict[str, DataLoader]]:
    """Captioning specific helper function for getting data loaders.

    Captioning is becoming a mode where we have a lot of additional
    scripts for generating prompts, metrics, etc, so helpful to have
    a single chunk of code for getting the relevant data loaders
    given some config.
    """
    datasets, collators, _ = load_datasets_for_eval(
        data_args,
        model_args,
        separate_splits=True,
        keep_splits_union=False,
    )

    print(f"making data loaders: {datasets}")
    datasets = datasets["testing"]
    collators = collators["testing"]

    if len(datasets) == 0:
        raise ValueError("received zero datasets, are they marked as 'testing'?")

    data_loaders = defaultdict(dict)
    for task, task_datasets in datasets.items():
        for dataset_key, dataset in task_datasets.items():
            print(f"task: {task} dataset: {dataset_key} N: {len(dataset)}")
            data_loaders[task][dataset_key] = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=collators[task][dataset_key],
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
        )
    return data_loaders


def load_and_validate_model_args(
    path: str,
) -> Dict:
    """Load model specifications from YAML file."""
    with open(path, "r") as fh:
        raw_data = yaml.safe_load(fh)
    if "models" in raw_data:
        raw_data = raw_data["models"]

    model_args = {}
    for model_specs in raw_data:
        this_model_args = model_specs.get("args", {})
        assert isinstance(this_model_args, dict), f"expected dict, got {type(this_model_args)}"
        model_args[model_specs["model_name"]] = this_model_args
    return model_args

def write_metrics(
    metrics: Dict,
    output_dir: Optional[str],
):
    """Write metrics to an output directory.

    Writes one TSV per task to the specified directory. `metrics` is
    a dict of results as returned by `procyon.evaluate.framework.core.run_evaluation`.
    """
    if output_dir is None:
        output_dir = os.getcwd()
    for task, results in metrics.items():
        path = os.path.join(output_dir, f"{task}_metrics.tsv")
        task_metrics = []
        for model_name, model_metrics in results.items():
            df = (pd.DataFrame(model_metrics)
                .T
                .rename_axis(index="dataset")
                .reset_index()
                .assign(model=model_name))

            # Move model and dataset cols to front.
            df = df[["model", "dataset"] + [col for col in df.columns if col not in ["model", "dataset"]] ]
            task_metrics.append(df)

        task_metrics = pd.concat(task_metrics)
        with open(path, "w") as fh:
            task_metrics.to_csv(fh, index=False, sep="\t")

def get_train_relations_for_eval_dataset(eval_dataset, training_splits = ["CL_train"]):
    '''
    Gets relations in the train splits for a given eval dataset
        - At high level, extracts needed information from eval_dataset to get training relations,
            returns aaseq_text_relations but for the training data
    '''
    data_dir = eval_dataset.data_dir
    aaseq_type = eval_dataset.aaseq_type
    text_type = eval_dataset.text_type
    text_split_method = eval_dataset.text_split_method
    all_relations = pd.read_csv(os.path.join(data_dir,
                                                "integrated_data",
                                                "v1",
                                                f"{aaseq_type}_{text_type}",
                                                text_split_method,
                                                f"{aaseq_type}_{text_type}_relations_indexed.unified.csv"))

    # Filter down to train:
    # Filter relation type + train split
    if eval_dataset.relation_type != "all":
        # Filter dataframe by relation:
        if eval_dataset.text_type != 'go':
            valid_rel = get_and_check_relation_id(
                eval_dataset.data_dir, eval_dataset.relation_type
            )
            all_relations = all_relations.loc[lambda x: x.relation == valid_rel]
        else:
            all_relations = all_relations.loc[lambda x: x.text_type.str.lower() == eval_dataset.relation_type.lower()]

    # Now train:
    train_relations = (all_relations
                         .loc[lambda x: x.split.isin(training_splits)]
                         [["seq_id", "relation", "text_id"]])

    return train_relations

def get_dataset_alternate_splits(
    dataset: Union[AASeqTextUnifiedDataset, AASeqDataset],
    want_splits: List[str],
) -> Union[AASeqTextUnifiedDataset, AASeqDataset]:
    """Helper function for cloning a dataset but with different splits."""
    # First need to construct the corresponding train dataset for
    # the given query set.
    if isinstance(dataset, AASeqTextUnifiedDataset):
        new_dataset = AASeqTextUnifiedDataset(
            dataset.data_dir,
            dataset.aaseq_type,
            dataset.text_type,
            dataset.relation_type,
            splits_to_use=want_splits,
            text_split_method=dataset.text_split_method,
        )
    elif isinstance(dataset, AASeqDataset):
        new_dataset = AASeqDataset(
            dataset.data_dir,
            dataset.aaseq_type,
            dataset.relation_type,
            splits_to_use=want_splits,
            store_reverse_edges=False,
        )
    else:
        raise ValueError(f"unexpected dataset type: {type(dataset)}")
    return new_dataset

def extract_qa_data(
    data_loader: DataLoader,
) -> Tuple[List[Tuple[int, int]], List[str]]:
    """Extract QA relations and ground truth from data loader."""
    # Where the index of the actual text query is depends on whether or not
    # this is a dataset with context augmentation.
    no_context_aug = data_loader.collate_fn._get_input_contexts([], []) is None
    if no_context_aug:
        query_text_idx = -1
    else:
        query_text_idx = -2

    aaseq_text_pairs = []
    ground_truth = []
    for batch in tqdm(data_loader):
        seq_idxs = [x[-1] for x in batch["reference_indices"]["input"]["seq"]]
        text_idxs = [x[query_text_idx] for x in batch["reference_indices"]["input"]["text"]]

        aaseq_text_pairs.extend(zip(seq_idxs, text_idxs))

        ground_truth.extend(batch["target"]["text"])
    return aaseq_text_pairs, ground_truth

def optimal_qa_thresh_acc(
    yes_probs: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """Get best threshold for QA based on accuracy."""
    # Select best threshold based on "peeking" accuracy
    threshs = np.unique(yes_probs)
    accs = []
    for thresh in threshs:
        pred = np.where(yes_probs >= thresh, "yes", "no")
        accs.append(np.mean(pred == labels))
    best_acc_idx = np.argmax(np.array(accs))
    best_thresh = threshs[best_acc_idx]
    return best_thresh, accs[best_acc_idx]
