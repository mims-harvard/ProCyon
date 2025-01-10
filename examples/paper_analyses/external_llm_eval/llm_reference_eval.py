import os

import pandas as pd

from transformers.hf_argparser import HfArgumentParser

from procyon.data.data_utils import DATA_DIR
from procyon.evaluate.framework.caption import calc_caption_metrics
from procyon.evaluate.framework.utils import load_eval_data_loaders
from procyon.training.training_args_IT import (
    TrainArgs,
    DataArgs,
    ModelArgs,
    postprocess_args,
)

UNIPROT_IDS = pd.read_pickle(
    os.path.join(
        DATA_DIR,
        "integrated_data",
        "v1",
        "protein",
        "protein_info_filtered.pkl",
    )
)[["index", "protein_id"]]


def uniprot_id_to_index(uniprot_id):
    assert (
        UNIPROT_IDS["protein_id"] == uniprot_id
    ).sum() == 1, "ID {} not found in internal database".format(uniprot_id)
    i = UNIPROT_IDS["index"].loc[UNIPROT_IDS["protein_id"] == uniprot_id].item()
    return i


def index_to_uniprot_id(i):
    uniprot_id = UNIPROT_IDS["protein_id"].loc[UNIPROT_IDS["index"] == i].item()
    return uniprot_id


def load_generated_captions(
    base_path: str,
    aaseq_type: str,
    text_type: str,
    relation: str,
    suffix: str,
) -> pd.DataFrame:
    path = os.path.join(base_path, f"{aaseq_type}_{text_type}_{relation}{suffix}")
    preds = (
        pd.read_csv(path)
        .assign(
            seq_id=lambda x: x.protein_id.apply(uniprot_id_to_index),
        )
        .rename(columns={"response": "generated_caption"})
    )
    if not (preds.finish_reason == "stop").all():
        print(f"caption finish reasons: {preds.finish_reason.value_counts()}")
        # Sometimes response will be NA if LLM didn't return a response, e.g.
        # if it thought it should be filtered due to content (LLMs typically
        # are configured to avoid giving medical advice).
        preds = preds.fillna(value={"generated_caption": ""})

    return preds[["seq_id", "generated_caption"]]


def run_caption_eval(
    train_args: TrainArgs,
    data_args: DataArgs,
    model_args: ModelArgs,
    llm_captions_dir: str,
    outdir: str,
    suffix: str,
):
    data_args.use_caption = True
    data_args.use_entity_compositions = True

    model_args.use_aaseq_embeddings = True
    
    data_loaders = load_eval_data_loaders(
        data_args,
        model_args,
        train_args.caption_batch_size,
        train_args.num_workers,
    )
    data_loaders = data_loaders["caption"]

    all_metrics = []
    for dataset_key, data_loader in data_loaders.items():
        aaseq_type, text_type, relation = dataset_key.split("_", maxsplit=2)
        generated_captions = load_generated_captions(
            llm_captions_dir,
            aaseq_type,
            text_type,
            relation,
            suffix,
        )
        this_outdir = os.path.join(outdir, dataset_key)
        os.makedirs(this_outdir, exist_ok=True)
        metrics = calc_caption_metrics(
            generated_captions, data_loader, output_dir=this_outdir
        )
        metrics["dataset"] = dataset_key
        all_metrics.append(metrics)

    with open(os.path.join(outdir, "caption_metrics_summary.tsv"), "w") as fh:
        pd.DataFrame(all_metrics).to_csv(fh, sep="\t", index=False)


if __name__ == "__main__":
    parser = HfArgumentParser((TrainArgs, DataArgs, ModelArgs))
    parser.add_argument("--llm_captions_dir")
    parser.add_argument("--suffix", default=".caption.gpt_response.csv")
    parser.add_argument("--outdir")

    train_args, data_args, model_args, script_args = (
        parser.parse_args_into_dataclasses()
    )

    if train_args.from_yaml is not None:
        train_args, data_args, model_args = parser.parse_yaml_file(
            train_args.from_yaml,
        )
    if train_args.from_json is not None:
        train_args, data_args, model_args = parser.parse_json_file(
            train_args.from_json,
        )

    train_args, data_args, model_args = postprocess_args(
        train_args, data_args, model_args
    )

    run_caption_eval(
        train_args,
        data_args,
        model_args,
        script_args.llm_captions_dir,
        script_args.outdir,
        script_args.suffix,
    )
