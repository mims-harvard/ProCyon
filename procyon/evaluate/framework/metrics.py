from typing import Union

import torch
import numpy as np

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
)

FIXEDK = 25


def fmax_score(ys: np.ndarray, preds: np.ndarray, beta=1.0, pos_label=1):
    """
    Radivojac, P. et al. (2013). A Large-Scale Evaluation of Computational Protein Function Prediction. Nature Methods, 10(3), 221-227.
    """
    precision, recall, thresholds = precision_recall_curve(
        y_true=ys, probas_pred=preds, pos_label=pos_label
    )
    numerator = (1 + beta**2) * (precision * recall)
    denominator = (beta**2 * precision) + recall
    with np.errstate(divide="ignore"):
        fbeta = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=(denominator != 0),
        )
    return np.nanmax(fbeta), thresholds[np.argmax(fbeta)]


def precision_recall_at_k(
    y: np.ndarray, preds: np.ndarray, k: int, names: np.ndarray = None
):
    """
    Calculate recall@k, precision@k, and AP@k for binary classification.
    """
    assert preds.shape == y.shape
    assert k > 0

    # Sort the scores and the labels by the scores
    sorted_indices = np.argsort(preds.flatten())[::-1]
    sorted_preds = preds[sorted_indices]
    sorted_y = y[sorted_indices]
    if names is not None:
        sorted_names = names[sorted_indices]
    else:
        sorted_names = None

    # Get the scores of the k highest predictions
    topk_preds = sorted_preds[:k]
    topk_y = sorted_y[:k]

    # Calculate the recall@k and precision@k
    recall_k = np.sum(topk_y, axis=-1) / np.sum(y, axis=-1)
    precision_k = np.sum(topk_y, axis=-1) / k

    # Calculate the AP@k
    ap_k = average_precision_score(topk_y, topk_preds)

    if k > preds.shape[-1]:
        recall_k = np.nan
        precision_k = np.nan
        ap_k = np.nan

    return recall_k, precision_k, ap_k, (sorted_y, sorted_preds, sorted_names)


def precision_recall_topk(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    k: int,
    return_all_vals: bool = False,
):
    """
    Calculate precision and recall for top-k accuracy in multi-label classification.
    Parameters:
        y_true (array-like): True binary labels (shape: [n_samples, n_classes]).
        y_pred (array-like): Predicted probabilities for each class (shape: [n_samples, n_classes]).
        k (int): The value of k for top-k accuracy.
    Returns:
        precision (float): Precision for top-k accuracy.
        recall (float): Recall for top-k accuracy.
    """
    # Some of the following code doesn't work with numpy arrays, so just
    # standardize to torch.Tensor
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)

    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred)
    else:
        y_pred = y_pred.clone()

    # Make sure labels are actually binary.
    non_nan_labels = y_true[~torch.isnan(y_true)]
    if not np.isin(non_nan_labels, [0, 1]).all():
        raise ValueError(f"expected labels to be 0 or 1, got: {y_true}")

    num_samples, num_classes = y_true.shape
    if k > num_classes:
        print(
            f"Provided value of k greater than number of items ({k} > {num_classes}), "
            "padding with neg inf."
        )
        pad_len = k - num_classes
        y_pred = torch.cat(
            (y_pred, torch.full((num_samples, pad_len), -float("inf"))), dim=1
        )

    # Convert any NaN predictions to neg inf for better behavior during sorting.
    # Positions where label is NaN corresponds to pairs we want to ignore,
    # so we also set those predictions to NaN.
    y_pred[torch.isnan(y_true) | torch.isnan(y_pred)] = -float("inf")
    topk_vals, topk_idxs = y_pred.topk(k=k)

    precisions = []
    recalls = []
    fmaxes = []

    any_nan = False
    for i in range(num_samples):
        true_labels = y_true[i]
        preds = y_pred[i]

        sorted_indices = topk_idxs[i]
        topk_preds = topk_vals[i]
        is_neginf = torch.isneginf(topk_preds)
        if is_neginf.any().item():
            any_nan = True
            first_nan = torch.nonzero(is_neginf, as_tuple=True)[0][0].item()
            sorted_indices = sorted_indices[:first_nan]

        true_labels_k = true_labels[sorted_indices]

        # Calculate true positives, relevant items, and retrieved items
        query_true_pos = true_labels_k.nansum().item()
        query_relevant = true_labels.nansum().item()
        query_retrieved = len(sorted_indices)

        if query_retrieved > 0:
            precisions.append(query_true_pos / query_retrieved)
        else:
            precisions.append(0.0)

        if query_relevant > 0:
            recalls.append(query_true_pos / query_relevant)
        else:
            recalls.append(0.0)

        want_idxs = ~true_labels.isnan() & ~preds.isnan()
        want_labels = true_labels[want_idxs]
        want_preds = preds[want_idxs]

        fmaxes.append(fmax_score(want_labels, want_preds)[0])

    if any_nan:
        print(
            "NaNs found when calculating top-k precision/recall. Results truncated to number of non-NaN items "
            "(this may be expected for some models, e.g. BLAST)"
        )

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_fmax = np.mean(fmaxes)
    if return_all_vals:
        return avg_precision, avg_recall, avg_fmax, precisions, recalls, fmaxes
    else:
        return avg_precision, avg_recall
