import os
from collections import defaultdict
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from procyon.data.data_utils import DATA_DIR
from procyon.data.dataset import (
    AASeqDataset,
    AASeqTextUnifiedDataset,
    ProteinEvalDataset,
)

from procyon.evaluate.framework.args import EvalArgs
from procyon.evaluate.framework.metrics import precision_recall_topk
from procyon.evaluate.framework.utils import (
    calc_bootstrap_bounds,
    get_train_relations_for_eval_dataset,
    sum_dicts,
)
from procyon.training.training_args_IT import ModelArgs


ALL_PROTEINS_FILE = os.path.join(
    DATA_DIR, "integrated_data/v1/protein/protein_info_filtered.pkl"
)

ALL_DOMAINS_FILE = os.path.join(
    DATA_DIR, "integrated_data/v1/domain/domain_info_filtered.pkl"
)


class AbstractRetrievalModel:
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        return self

    # Return a tensor of size num_queries x num_targets
    # where each value (i, j) is the score of target j
    # given query i.
    #
    # query_loader - DataLoader for queries (either text
    #                or protein)
    # target_loader - DataLoader for targets (proteins)
    # query_order: - List giving the expected order of
    #                queries in the returned tensor (i.e.
    #                row i of the returned tensor should
    #                correspond to query at query_order[i]).
    # target_order: - List giving the expected order of
    #                 targets in the returned tensor
    #                 (analogous to above but for columns and
    #                 targets).
    def get_predictions(
        self,
        query_loader: DataLoader,
        target_loader: DataLoader,
        query_order: List,
        target_order: List,
    ) -> torch.Tensor:
        raise Exception("not implemented")


def get_retrieval_target_set(
    query_dataset: Union[AASeqTextUnifiedDataset, AASeqDataset],
    dataset_eval_args: Dict,
    eval_args: EvalArgs,
    aaseq_type: str = "protein",
) -> pd.Series:
    if eval_args.retrieval_eval_all_aaseqs:
        if aaseq_type == "protein":
            target_ids = pd.read_pickle(ALL_PROTEINS_FILE).index.to_series()
        elif aaseq_type == "domain":
            target_ids = pd.read_pickle(ALL_DOMAINS_FILE).index.to_series()
        else:
            raise ValueError(f"unknown aaseq type: {aaseq_type}")
    else:
        # Can insert set minus here - i.e. getting all proteins not in train
        target_ids = pd.Series(query_dataset.unique_aaseq)

    if "target_subset" in dataset_eval_args:
        subset_path = dataset_eval_args["target_subset"]
        subset_ids = pd.read_pickle(subset_path).index.to_series()
        if not subset_ids.isin(target_ids).all():
            raise ValueError(
                f"dataset {query_dataset.name()}: some IDs from the specified target "
                "subset were not found in the superset of IDs. You may need to set "
                "retrieval_eval_all_proteins = True or check for incorrect IDs."
            )
        target_ids = subset_ids
    return target_ids


def get_retrieval_target_proteins_loader(
    targets: pd.Series,
    batch_size: int,
) -> DataLoader:

    protein_dataset = ProteinEvalDataset(targets)
    return DataLoader(
        protein_dataset,
        batch_size=batch_size,
        num_workers=2,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
    )


def prep_for_retrieval_eval(
    query_dataset: Union[AASeqTextUnifiedDataset, AASeqDataset],
    target_ids: pd.Series,
    filter_training: bool = True,
) -> Tuple[np.ndarray, List, List]:
    """Perform pre-processing for a retrieval task evaluation.

    Creates a np.ndarray of labels for a retrieval task, where the
    matrix is (num_queries x num_targets) and each value (i, j)
    is 1 if the target is associated with the query, and 0 otherwise.

    Also returns lists of query and target IDs in the same order as
    matrix axes.
    """
    # Need to handle AA seq <-> AA seq retrieval slightly differently
    # because underlying dataset class is different, and relations are
    # undirected.
    if isinstance(query_dataset, AASeqTextUnifiedDataset):
        is_ppi = False
    elif isinstance(query_dataset, AASeqDataset):
        is_ppi = True
    else:
        raise ValueError(f"unexpected dataset type: {type(query_dataset)}")

    # First construct label matrix. Matrix is num_queries x num_targets
    # where each value (i, j) is 1 if the target is associated with the
    # query, and 0 otherwise.
    num_targets = len(target_ids)

    if is_ppi:
        # Have to reverse it because of the notation below (target_id, _, query_id)
        relations = [
            (id2, rel, id1) for (id1, rel, id2) in query_dataset.aaseq_relations
        ]
        unique_queries = np.unique(
            [id1 for (id1, rel, id2) in query_dataset.aaseq_relations]
        ).tolist()

        num_queries = len(unique_queries)
    else:
        relations = query_dataset.true_relations
        unique_queries = query_dataset.unique_text.tolist()
        num_queries = len(query_dataset.unique_text)
        assert len(unique_queries) == len(np.unique(unique_queries))

    unique_targets = np.unique(target_ids).tolist()
    assert len(unique_targets) == len(target_ids)

    query_map = {query_id: i for i, query_id in enumerate(unique_queries)}
    target_map = {target_id: i for i, target_id in enumerate(unique_targets)}

    labels_mat = torch.zeros((num_queries, num_targets))
    # When using a subset of targets (specified via `dataset_eval_args["target_subset"]`
    # above), we want to ignore relations between a query and targets that aren't in
    # our subset. Just throwing these out could lead to unintended silent consequences,
    # so we want to warn if we're throwing out relations when constructing the labels.
    # To avoid spamming in the case of actually using a target subset as intended,
    # only warn once.
    warned = False

    # Relations (for AAseq <-> text) are stored as (seq_id, rel, text_id)
    # so in this case, the seq is the target and the text is the query.
    for target_id, _, query_id in relations:
        if target_id not in target_map:
            if not warned:
                print(
                    f"WARNING: in dataset {query_dataset.name()}, ignoring a relation due "
                    f"to target not being found (id={target_id}). In the case of using a "
                    "specified subset of targets, this is expected, otherwise may want to "
                    "check that all target IDs in dataset are in ALL_PROTEINS_FILE"
                )
                warned = True
            continue
        query_idx = query_map[query_id]
        target_idx = target_map[target_id]
        labels_mat[query_idx, target_idx] = 1

    # Perform adjustment for training leakage of pairs across GO terms and proteins in split
    if (not is_ppi) and filter_training:
        train_relations = get_train_relations_for_eval_dataset(query_dataset)

        # Further filter train_relations based on GO and aaseq presence in split:
        train_relations = train_relations.loc[
            train_relations.text_id.isin(unique_queries)
            & train_relations.seq_id.isin(unique_targets),
            :,
        ]

        for target_id, _, query_id in train_relations.itertuples(index=False):
            query_idx = query_map[query_id]
            target_idx = target_map[target_id]
            labels_mat[query_idx, target_idx] = float("nan")

    return labels_mat, unique_queries, unique_targets


def calc_and_plot_auroc_auprc(
    preds_mat: torch.Tensor,
    labels_mat: torch.Tensor,
    output_dir: Optional[str] = None,
    calc_per_query: bool = True,
) -> Tuple[float, float]:
    is_nan = torch.isnan(preds_mat)
    if is_nan.all().item():
        print(
            "Received all NaNs when calculating ROC and PRC "
            "(this may be expected for some models with "
            "downsampling, e.g. BLAST)"
        )
        return 0, 0
    if is_nan.any().item():
        print(
            "NaNs found when calculating ROC and PRC "
            "(this may be expected for some models, e.g. BLAST)"
        )
        fill_val = preds_mat[~is_nan].min().item() - 1
        preds_mat = preds_mat.clone()
        preds_mat[is_nan] = fill_val

    query_aurocs = []
    query_auprcs = []
    if calc_per_query:
        for idx in range(len(preds_mat)):
            query_preds = preds_mat[idx]
            query_labels = labels_mat[idx]
            pos_scores = query_preds[query_labels == 1]
            neg_scores = query_preds[query_labels == 0]

            labels = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
            scores = np.concatenate([pos_scores, neg_scores])
            query_aurocs.append(roc_auc_score(labels, scores))
            query_auprcs.append(average_precision_score(labels, scores))
        auroc = np.mean(query_aurocs)
        auprc = np.mean(query_auprcs)
    else:
        pos_scores = preds_mat[labels_mat == 1]
        neg_scores = preds_mat[labels_mat == 0]
        labels = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
        scores = np.concatenate([pos_scores, neg_scores])
        auroc = roc_auc_score(labels, scores)
        auprc = average_precision_score(labels, scores)

    if output_dir is not None:
        roc_path = os.path.join(output_dir, "roc.png")
        fpr, tpr, _ = roc_curve(labels, scores)
        ax = plt.gca()
        ax.step(fpr, tpr)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        plt.savefig(fname=roc_path)
        plt.close()

        prc_path = os.path.join(output_dir, "prc.png")
        precision, recall, _ = precision_recall_curve(labels, scores)
        ax = plt.gca()
        ax.step(recall, precision)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        plt.savefig(fname=prc_path)
        plt.close()

    return auroc, auprc, query_aurocs, query_auprcs


def calc_retrieval_metrics_single(
    preds_mat: torch.Tensor,
    labels_mat: torch.Tensor,
    eval_args: EvalArgs,
    output_dir: str,
) -> Tuple[Dict, Dict]:
    """Calculate retrieval metrics for a single set of predictions and labels."""
    metrics = {}
    samples_dict = {}
    for k in eval_args.retrieval_top_k_vals:
        if k > labels_mat.shape[1]:
            print(
                f"Number of candidate labels is {labels_mat.shape[1]} which is smaller than k={k}, so skipping this evaluation"
            )
            continue
        (precision, recall, fmax, per_query_precisions, per_query_recalls, fmaxes) = (
            precision_recall_topk(
                labels_mat,
                preds_mat,
                k,
                return_all_vals=True,
            )
        )

        metrics[f"precision_k{k}"] = precision
        metrics[f"recall_k{k}"] = recall

        samples_dict[f"precision_k{k}"] = per_query_precisions
        samples_dict[f"recall_k{k}"] = per_query_recalls

    metrics[f"Fmax"] = fmax
    samples_dict[f"Fmax"] = fmaxes

    (auroc, auprc, per_query_aurocs, per_query_auprcs) = calc_and_plot_auroc_auprc(
        preds_mat,
        labels_mat,
        output_dir,
        eval_args.retrieval_auroc_auprc_per_query,
    )
    metrics["auroc"] = auroc
    metrics["auprc"] = auprc

    samples_dict["auroc"] = per_query_aurocs
    samples_dict["auprc"] = per_query_auprcs

    return metrics, samples_dict


def calc_retrieval_metrics_class_balanced(
    preds_mat: torch.Tensor,
    labels_mat: torch.Tensor,
    eval_args: EvalArgs,
    output_dir: str,
) -> Dict:
    """Calculate retrievel metrics as an average over class-balanced samples."""
    assert isinstance(eval_args.retrieval_balanced_metrics_num_samples, int)
    num_samples_per_query = eval_args.retrieval_balanced_metrics_num_samples
    num_queries = len(preds_mat)
    rng = np.random.default_rng(eval_args.seed)

    all_metrics = defaultdict(float)
    all_samples = defaultdict(list)

    for sample_num in range(num_samples_per_query):
        all_preds = []
        all_labels = []

        no_pos_labels = []
        all_nan_preds = []
        num_queries_evaluated = 0
        for i in range(num_queries):
            # If full row of predictions is NaN, want to skip, this can happen if
            # a model just wants to completely reject a sample, e.g. non-zero shot models
            # like kNN or MLP baselines receive a text that wasn't in their training set.
            if preds_mat[i].isnan().all().item():
                all_nan_preds.append(i)
                continue
            # If a sample has no positive associations, then we also want to skip, as this
            # is a trivial/uninteresting prediction task. This can happen for multiple reasons
            # but one observed reason is relations existing in both a train and eval split,
            # resulting in the relations being filtered by filter_train_relations
            if labels_mat[i].nansum() == 0:
                no_pos_labels.append(i)
                continue
            pos_idxs = torch.argwhere(labels_mat[i] == 1).squeeze(-1)
            all_neg_idxs = torch.argwhere(labels_mat[i] == 0).squeeze(-1)

            num_negs_to_sample = (
                len(pos_idxs) * eval_args.retrieval_balanced_metrics_neg_per_pos
            )
            # If the number of negatives we want is more than what's available, just calculate
            # metrics once.
            if len(all_neg_idxs) <= num_negs_to_sample:
                neg_idxs = all_neg_idxs
            else:
                neg_idxs = rng.choice(
                    all_neg_idxs,
                    replace=False,
                    size=num_negs_to_sample,
                )

            want_idxs = torch.cat(
                (
                    torch.tensor(neg_idxs),
                    pos_idxs,
                )
            )
            all_preds.append(preds_mat[i, want_idxs])
            all_labels.append(labels_mat[i, want_idxs])
            num_queries_evaluated += 1

        if sample_num == 0 and len(no_pos_labels) != 0:
            print(
                f"retrieval eval: found {len(no_pos_labels)} queries with no positive labels: {no_pos_labels}"
            )

        if sample_num == 0 and len(all_nan_preds) != 0:
            print(
                f"retrieval eval: found {len(all_nan_preds)} queries with all NaN predictions: {all_nan_preds}"
            )

        sampled_preds = pad_sequence(
            all_preds, batch_first=True, padding_value=float("nan")
        )
        sampled_labels = pad_sequence(
            all_labels, batch_first=True, padding_value=float("nan")
        )

        # Only generate plots for final sample.
        if sample_num == num_samples_per_query - 1:
            out_dir = output_dir
        else:
            out_dir = None
        avg_metrics, samples = calc_retrieval_metrics_single(
            sampled_preds,
            sampled_labels,
            eval_args,
            out_dir,
        )
        sum_dicts(all_metrics, avg_metrics)
        sum_dicts(all_samples, samples)

    for k in all_metrics.keys():
        all_metrics[k] /= num_samples_per_query
    all_metrics["N"] = num_queries_evaluated

    return all_metrics, all_samples


def calc_retrieval_metrics(
    preds_mat: torch.Tensor,
    labels_mat: torch.Tensor,
    eval_args: EvalArgs,
    output_dir: str,
) -> Dict:
    """Calculate retrieval metrics for a full set of results, dispatching to appropriate methods."""
    if eval_args.retrieval_balanced_metrics_num_samples is None:
        raw_metrics, samples = calc_retrieval_metrics_single(
            preds_mat,
            labels_mat,
            eval_args,
            output_dir,
        )
    else:
        raw_metrics, samples = calc_retrieval_metrics_class_balanced(
            preds_mat,
            labels_mat,
            eval_args,
            output_dir,
        )
    raw_metrics.update(calc_bootstrap_bounds(samples))
    return raw_metrics


def run_retrieval_eval(
    model: AbstractRetrievalModel,
    data_loader: DataLoader,
    eval_args: EvalArgs,
    dataset_eval_args: Dict,
    model_name: str,
    dataset_key: str,
    output_dir: str,
) -> Dict:
    targets = get_retrieval_target_set(
        data_loader.dataset,
        dataset_eval_args,
        eval_args,
        data_loader.dataset.aaseq_type,
    )
    print(
        f"retrieval: evaluating model {model_name} on dataset {dataset_key} , num_queries={len(data_loader.dataset)}, num_targets={len(targets)}"
    )
    target_loader = get_retrieval_target_proteins_loader(
        targets,
        eval_args.batch_size,
    )
    labels, query_order, target_order = prep_for_retrieval_eval(
        data_loader.dataset,
        targets,
        filter_training=eval_args.filter_training_pairs,
    )
    preds_path = os.path.join(output_dir, "predictions.pkl")
    if os.path.exists(preds_path) and eval_args.use_cached_results:
        print(f"retrival: {model_name}: {dataset_key} loading cached predictions")
        with open(preds_path, "rb") as fh:
            predictions = torch.load(fh)
    else:
        predictions = model.get_predictions(
            data_loader,
            target_loader,
            query_order,
            target_order,
        )
        with open(preds_path, "wb") as fh:
            torch.save(predictions, fh)
    metrics = calc_retrieval_metrics(
        predictions,
        labels,
        eval_args,
        output_dir,
    )

    print(f"dataset: {dataset_key}")
    print("retrieval results:")
    print(metrics)

    return metrics
