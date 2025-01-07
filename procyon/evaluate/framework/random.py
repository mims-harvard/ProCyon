from typing import (
    Dict,
    List,
)

import torch
import tqdm
import pandas as pd
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from procyon.data.dataset import (
    AASeqTextUnifiedDataset,
    AASeqDataset,
    load_unified_aaseq_text_relations,
    load_unified_aaseq_relations,
)
from procyon.training.training_args_IT import ModelArgs

from procyon.evaluate.framework.args import EvalArgs
from procyon.evaluate.framework.caption import AbstractCaptionModel
from procyon.evaluate.framework.retrieval import AbstractRetrievalModel


class RandomCaptionEvalBase(AbstractCaptionModel):
    """Random baseline for captioning.

    Rather than generating captions in any sensible way, this
    baseline just randomly selects reference captions from the
    target dataset.
    """

    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        self.sample_from = model_config.get("sample_from", "full_dataset")
        assert self.sample_from in ["full_dataset", "split"]

        self.sample_method = model_config.get("sample_method", "uniform")
        assert self.sample_method in ["uniform", "weighted", "majority_rule"]

        self.max_len = eval_args.caption_max_len
        self.rng = np.random.default_rng(seed=eval_args.seed)

    def get_predictions(
        self,
        data_loader: DataLoader,
    ) -> pd.DataFrame:
        if not isinstance(data_loader.dataset, AASeqTextUnifiedDataset):
            raise ValueError(
                f"random model expected AASeqTextUnifiedDataset, got {type(data_loader.dataset)}"
            )
        aaseq_indices = []
        for model_inputs in tqdm(data_loader):
            aaseq_indices.extend(
                [x[-1] for x in model_inputs["reference_indices"]["input"]["seq"]]
            )

        if self.sample_from == "full_dataset":
            choices = np.arange(len(data_loader.collate_fn.text_sequences))
        else:
            choices = data_loader.dataset.all_texts

        if self.sample_method == "uniform":
            weights = None
        else:
            # sample_method is weighted or majority rule, either way
            # we need the relation counts.
            relations = load_unified_aaseq_text_relations(
                data_loader.dataset.data_dir,
                data_loader.dataset.aaseq_type,
                data_loader.dataset.text_type,
                data_loader.dataset.text_split_method,
                data_loader.dataset.relation_type,
            )

            # Subset to just the train set for generating predictions.
            counts = relations.text_id.value_counts()
            if self.sample_method == "weighted":
                if self.sample_from == "full_dataset" and len(counts) != len(choices):
                    print(
                        f"WARNING: random caption eval for dataset {data_loader.dataset.name()}: ",
                        "sampling from 'full_dataset' with weighted sampling, but "
                        "specified relations for weighting do not cover the full dataset. Falling "
                        "back to uniform sampling",
                    )
                    weights = None
                else:
                    counts = counts[choices]
                    weights = counts / counts.sum()
            else:
                # sample method is majority_rule
                counts = counts[choices]
                top_idx = counts.argmax()
                weights = np.zeroes_like(counts)
                weights[top_idx] = 1

        indices = self.rng.choice(
            choices,
            size=len(aaseq_indices),
            replace=True,
            p=weights,
        )
        generated_captions = [data_loader.collate_fn.text_sequences[i] for i in indices]
        # Truncate to max len.
        generated_captions = [
            " ".join(x.split()[: self.max_len]) for x in generated_captions
        ]

        return pd.DataFrame(
            {
                "seq_id": aaseq_indices,
                "generated_caption": generated_captions,
            }
        )


class UniformRandomCaptionEval(RandomCaptionEvalBase):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        model_config["sample_method"] = "uniform"
        super().__init__(model_config, eval_args, model_args, device)


class WeightedRandomCaptionEval(RandomCaptionEvalBase):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        model_config["sample_method"] = "weighted"
        model_config["sample_from"] = "full_dataset"
        super().__init__(model_config, eval_args, model_args, device)


class MajorityRuleRandomCaptionEval(RandomCaptionEvalBase):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        model_config["sample_method"] = "majority_rule"
        model_config["sample_from"] = "full_dataset"
        super().__init__(model_config, eval_args, model_args, device)


class RandomRetrievalEvalBase(AbstractRetrievalModel):
    """Random baseline for captioning.

    Rather than generating captions in any sensible way, this
    baseline just randomly selects reference captions from the
    target dataset.
    """

    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        self.sample_from = model_config.get("sample_from", "all_targets")
        assert self.sample_from in ["all_targets", "full_dataset"]

        self.sample_method = model_config.get("sample_method", "uniform")
        assert self.sample_method in ["uniform", "weighted", "majority_rule"]

        self.rng = np.random.default_rng(seed=eval_args.seed)

    def get_predictions(
        self,
        query_loader: DataLoader,
        target_loader: DataLoader,
        query_order: List,
        target_order: List,
    ) -> torch.Tensor:
        if isinstance(query_loader.dataset, AASeqTextUnifiedDataset):
            relations = load_unified_aaseq_text_relations(
                query_loader.dataset.data_dir,
                query_loader.dataset.aaseq_type,
                query_loader.dataset.text_type,
                query_loader.dataset.text_split_method,
                query_loader.dataset.relation_type,
            )
        elif isinstance(query_loader.dataset, AASeqDataset):
            relations = load_unified_aaseq_relations(
                query_loader.dataset.data_dir,
                query_loader.dataset.aaseq_type,
                query_loader.dataset.relation_type,
            )
        else:
            raise ValueError(
                f"random retrieval model got unexpected query dataset type: {type(query_loader.dataset)}"
            )

        # Subset to just the train set for generating predictions.
        relations = relations.query("split == 'CL_train'")

        if self.sample_method == "uniform":
            choices = target_order
            weights = None
        elif self.sample_method in ["weighted", "majority_rule"]:
            # sample_method is weighted or majority rule, either way
            # we need the relation counts.

            counts = relations.seq_id.value_counts()
            if self.sample_from == "all_targets" and len(counts) != len(target_order):
                print(
                    f"WARNING: random retrieval eval for dataset {query_loader.dataset.name()}: ",
                    "sampling from 'full_dataset' with weighted sampling, but "
                    "specified relations for weighting do not cover the full dataset. Falling "
                    "back to uniform sampling",
                )
                choices = target_order
                weights = None
            elif self.sample_method == "weighted":
                choices = counts.index.tolist()
                weights = counts / counts.sum()
            else:
                # Majority rule
                choices = counts.index.tolist()
                weights = counts / counts.sum()
                sampled_order = weights.sort_values(ascending=False).index.tolist()

        values = torch.linspace(start=1, end=0, steps=len(choices) + 1)[:-1]
        ret = torch.zeros(len(query_order), len(target_order))
        for query_idx in range(len(query_order)):
            if self.sample_method == "uniform":
                sampled_order = self.rng.choice(
                    target_order,
                    size=len(target_order),
                    replace=False,
                    p=None,
                )
            elif self.sample_method == "weighted":
                sampled_order = self.rng.choice(
                    choices,
                    size=len(choices),
                    replace=False,
                    p=weights,
                )
            ret[query_idx, sampled_order] = values
        return ret


class UniformRandomRetrievalEval(RandomRetrievalEvalBase):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        model_config["sample_method"] = "uniform"
        super().__init__(model_config, eval_args, model_args, device)


class WeightedRandomRetrievalEval(RandomRetrievalEvalBase):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        model_config["sample_method"] = "weighted"
        model_config["sample_from"] = "full_dataset"
        super().__init__(model_config, eval_args, model_args, device)


class MajorityRuleRandomRetrievalEval(RandomRetrievalEvalBase):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        model_config["sample_method"] = "majority_rule"
        model_config["sample_from"] = "full_dataset"
        super().__init__(model_config, eval_args, model_args, device)
