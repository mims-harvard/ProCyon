import numpy as np

## Sklearn Metrics
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_curve, 
    precision_score, 
    recall_score, 
    average_precision_score, 
    roc_auc_score,
    f1_score,
    # fbeta_score,
)

FIXEDK = 25

def fmax_score(ys: np.ndarray, preds: np.ndarray, beta = 1.0, pos_label = 1):
    """
    Radivojac, P. et al. (2013). A Large-Scale Evaluation of Computational Protein Function Prediction. Nature Methods, 10(3), 221-227.

    TODO: Check this implementation
    """
    precision, recall, thresholds = precision_recall_curve(y_true = ys, probas_pred = preds, pos_label = pos_label)
    # precision += 1e-4
    # recall += 1e-4
    # f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    # return np.nanmax(f1), thresholds[np.argmax(f1)]
    numerator = (1 + beta**2) * (precision * recall)
    denominator = ((beta**2 * precision) + recall)
    with np.errstate(divide='ignore'):
        fbeta = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=(denominator!=0))
    return np.nanmax(fbeta), thresholds[np.argmax(fbeta)]

def precision_recall_at_k(y: np.ndarray, preds: np.ndarray, k: int, names: np.ndarray = None):
    """ 
    Calculate recall@k, precision@k, and AP@k for binary classification.

    TODO: Check this implementation
    """
    assert preds.shape == y.shape
    assert k > 0
    
    # Sort the scores and the labels by the scores
    sorted_indices = np.argsort(preds.flatten())[::-1]
    sorted_preds = preds[sorted_indices]
    sorted_y = y[sorted_indices]
    if names is not None:
        sorted_names = names[sorted_indices]
    else: sorted_names = None

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

