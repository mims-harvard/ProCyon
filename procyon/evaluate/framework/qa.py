import os
import pickle

from typing import (
    Dict,
    Tuple,
)

import torch

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score


from procyon.evaluate.framework.args import EvalArgs
from procyon.evaluate.framework.utils import calc_bootstrap_bounds
from procyon.training.training_args_IT import ModelArgs
from procyon.training.train_utils import (
    decompose_dataset_name,
    get_qa_metrics_from_preds
)

class AbstractQAModel:
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        return self

    def get_predictions(self,
        data_loader: DataLoader,
        aaseq_type: str = 'protein',
    ) -> Dict[str, torch.Tensor]:
        pass

def calc_qa_metrics(
    pred_tokens: torch.Tensor,
    y_tokens: torch.Tensor,
    yes_token: int,
    no_token: int,
) -> Dict:
    acc, f1 = get_qa_metrics_from_preds(
        pred_toks=pred_tokens,
        y_toks=y_tokens,
        yes_token=yes_token,
        no_token=no_token,
    )
    if isinstance(acc, torch.Tensor):
        acc = acc.item()

    acc_dict = {"acc": acc}
    correct = pred_tokens == y_tokens
    acc_dict.update(calc_bootstrap_bounds({"acc": correct}))

    f1_dict = {"f1": f1}
    def f1_func(preds, y_toks) -> float:
        return f1_score(y_toks, preds, average='macro')

    f1_dict.update(calc_bootstrap_bounds(
        {"f1": list(zip(pred_tokens.tolist(), y_tokens.tolist()))},
        statistic=f1_func,
        paired=True,
    ))
    acc_dict.update(f1_dict)
    return acc_dict

def run_qa_eval(
    model: AbstractQAModel,
    data_loader: DataLoader,
    eval_args: EvalArgs,
    dataset_eval_args: Dict,
    model_name: str,
    dataset_key: str,
    output_dir: str,
) -> Dict:
    print(f"QA: evaluating model {model_name} on dataset {dataset_key} , num_qs={len(data_loader.dataset)}")
    aaseq_type, _, _ = decompose_dataset_name(dataset_key)

    preds_path = os.path.join(output_dir, "results_dict.pkl")
    if os.path.exists(preds_path) and eval_args.use_cached_results:
        print(f"QA: {model_name}: {dataset_key} loading cached predictions")
        with open(preds_path, "rb") as fh:
            results_dict = pickle.load(fh)
    else:
        results_dict = model.get_predictions(
            data_loader,
            aaseq_type=aaseq_type,
        )
        with open(preds_path, "wb") as fh:
            pickle.dump(results_dict, fh)

    metrics = calc_qa_metrics(
        pred_tokens=results_dict["pred"],
        y_tokens=results_dict["y"],
        yes_token=model.yes_token,
        no_token=model.no_token,
    )

    print("Dataset", dataset_key)
    print("acc: {:.4f}".format(metrics["acc"]))
    print("f1: {:.4f}".format(metrics["f1"]))

    return metrics
