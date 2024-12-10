import gzip
import os

from collections import defaultdict
from typing import (
    Dict,
    List,
)

import evaluate
import torch
import pandas as pd

from procyon.evaluate.framework.args import EvalArgs
from procyon.training.training_args_IT import ModelArgs

from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

class AbstractCaptionModel:
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        return self

    def get_predictions(
        self,
        data_loader: DataLoader,
    ) -> pd.DataFrame:
        raise Exception("not implemented")

def truncate_for_bertscore(
    strs: List[str],
    tokenizer: PreTrainedTokenizer,
    max_len: int = 512,
) -> List[str]:
    """Round-trip through tokenizer to truncate."""
    tokenized = tokenizer(strs, max_length=max_len, truncation=True)
    return tokenizer.batch_decode(tokenized["input_ids"], skip_special_tokens=True)

def merge_reference_captions(
    preds: pd.DataFrame,
    data_loader: DataLoader,
) -> pd.DataFrame:
    merged = preds.merge(
        pd.DataFrame(data_loader.dataset.true_relations).drop(columns="relation"),
        on="seq_id",
    )

    collator = data_loader.collate_fn
    if collator.use_entity_compositions:
        reference_captions = collator._sample_batch_entity_descriptions(merged.text_id.to_list())
    else:
        reference_captions = collator.text_sequences[merged.text_id.to_list()].tolist()

    return merged.assign(reference_caption=reference_captions)

def calculate_bertscores(
    preds_w_refs: pd.DataFrame,
) -> pd.DataFrame:
    bertscorer = evaluate.load("bertscore")
    # Dummy pred to cache the relevant BERT scorer object
    _ = bertscorer.compute(
        predictions=["foo"],
        references=["bar"],
        lang="en-sci",
    )

    predictions_trunc = truncate_for_bertscore(
        preds_w_refs.generated_caption.to_list(),
        bertscorer.cached_bertscorer._tokenizer,
    )
    reference_trunc = truncate_for_bertscore(
        preds_w_refs.reference_caption.to_list(),
        bertscorer.cached_bertscorer._tokenizer,
    )
    results = bertscorer.compute(
        predictions=predictions_trunc,
        references=reference_trunc,
        rescale_with_baseline=True,
        lang="en-sci",
    )

    print(f"BERTScore hashcode: {results['hashcode']}")

    for metric_name in results.keys():
        if metric_name == "hashcode":
            continue
        preds_w_refs[f"bertscore_{metric_name}"] = results[metric_name]

    return preds_w_refs

def calculate_rouge(
    preds_w_refs: pd.DataFrame,
) -> pd.DataFrame:
    scorer = evaluate.load("rouge")

    results = scorer.compute(
        predictions=preds_w_refs.generated_caption,
        references=preds_w_refs.reference_caption,
        use_aggregator=False,
    )

    want_metrics = ["rouge1", "rouge2", "rougeL"]
    for metric_name in want_metrics:
        preds_w_refs[metric_name] = results[metric_name]
    return preds_w_refs

def calculate_bleu(
    preds_w_refs: pd.DataFrame,
) -> pd.DataFrame:
    scorer = evaluate.load("sacrebleu")

    # Have to compute one-by-one to get metrics per individual pair.
    bleu_scores = defaultdict(list)
    for gen, ref in preds_w_refs[["generated_caption", "reference_caption"]].itertuples(index=False):
        scores = scorer.compute(predictions=[gen], references=[ref])
        for i, val in enumerate(scores["precisions"]):
            bleu_scores[f"bleu{i+1}"].append(val/100)

    for metric_name, metric_vals in bleu_scores.items():
        preds_w_refs[metric_name] = metric_vals
    return preds_w_refs

def summarize_metrics(
    grouped_df: pd.core.groupby.generic.DataFrameGroupBy,
    metrics: List[str],
) -> pd.Series:
    all_cols = {}
    for metric in metrics:
        summary = grouped_df[metric].describe()
        all_cols[f"{metric}_max"] = summary["max"]
        all_cols[f"{metric}_mean"] = summary["mean"]
        all_cols["count"] = int(summary["count"])
    return pd.Series(all_cols)

def calc_caption_metrics(
    preds: pd.DataFrame,
    data_loader: DataLoader,
    output_dir: str,
) -> Dict:
    # Get and write dataframe with each generated caption shown alongside reference captions that
    # relate to the given aaseq.
    preds_w_ref = (merge_reference_captions(preds, data_loader)
                   .pipe(calculate_bertscores)
                   .pipe(calculate_rouge)
                   .pipe(calculate_bleu))
    with gzip.open(os.path.join(output_dir, "full_captions.tsv.gz"), "w") as fh:
        preds_w_ref.to_csv(fh, sep="\t", index=False)

    # Get and write dataframe with max and mean captioning metrics per aaseq, max and mean
    # calculated across all reference captions for that aaseq. Rationale for max is that
    # the one-to-many relation of aaseq to texts could mean that the model outputs a caption
    # that only captures/matches one reference caption (e.g. for Drugbank drug-carrier relations,
    # the protein may be a carrier for multiple drugs, and the model outputs a caption for just one
    # of those drugs).
    non_metric_cols =  ["seq_id", "text_id", "reference_caption", "generated_caption", "count"]
    metric_names = [x for x in preds_w_ref.columns if x not in non_metric_cols]
    summarized_scores = (preds_w_ref
                         .groupby("seq_id")
                         .apply(lambda x: summarize_metrics(x, metric_names))
                         .astype({"count": int})
                         .reset_index())
    with gzip.open(os.path.join(output_dir, "caption_scores_per_seq.tsv.gz"), "w") as fh:
        summarized_scores.to_csv(fh, sep="\t", index=False)

    # Metrics averaged over all aaseqs. Note that this is not weighted, i.e. aaseqs with many
    # relations are treated the same as those with few.
    metrics = (summarized_scores
               .drop(columns=["count"])
               .describe()
               .loc["mean"]
               .to_dict())
    return metrics


def run_caption_eval(
    model: AbstractCaptionModel,
    data_loader: DataLoader,
    eval_args: EvalArgs,
    dataset_eval_args: Dict,
    model_name: str,
    dataset_key: str,
    output_dir: str,
) -> Dict:
    print(f"caption: evaluating model {model_name} on dataset {dataset_key} , num_aaseqs={len(data_loader.dataset)}")
    preds = model.get_predictions(data_loader)

    metrics = calc_caption_metrics(preds, data_loader, output_dir)

    print(f"dataset: {dataset_key}")
    print("caption results:")
    print(metrics)

    return metrics