import logging, sys, os, random

import numpy as np
import pandas as pd

from typing import Any, Dict, List, Optional, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

import evaluate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers.trainer_utils import SchedulerType
from transformers import AutoTokenizer
from esm.data import Alphabet
import deepspeed

from procyon.data.dataset import (
    ProteinDataset,
    TextCLDataset,
    ProteinGODataset,
    ProteinProteinDataset,
    DomainGODataset,
    DomainPfamDataset,
    ProteinGODataset_OLD,
    AASeqTextDataset,
    AASeqDataset,
)

from procyon.data.data_collator import (
    ProteinMLMCollator,
    TextCLCollator,
    ProteinGOCLCollator,
    ProteinProteinCLCollator,
    DomainGOCLCollator,
    DomainPfamCLCollator,
)
from procyon.data.it_collator import RetrievalCollator, QACollator, CaptionCollator
from procyon.data.it_data_config import (
    ITMultiDatasetConfig,
    package_collators_for_trainer,
)

from procyon.training.training_args import TrainArgs, DataArgs, ModelArgs
from procyon.training.wandb_logger import WandbLogger

from procyon.data.metadataset import MetaDataset, MetaCollator

# TODO: make all argument names consistent with the variable names they are become mapped to
#       (makes code much easier to follow (e.g. through global code searches))


##########
# data utils
##########
def get_datasets(data_args: DataArgs, task_type="qa") -> Tuple[Dataset]:
    # train_protein_dataset, train_protein_go_dataset, train_protein_protein_dataset, train_pfam_dataset = None, None, None, None
    (
        train_protein_dataset,
        train_text_cl_dataset,
        train_protein_go_dataset,
        train_protein_protein_dataset,
        train_domain_go_dataset,
        train_domain_pfam_dataset,
    ) = (None, None, None, None, None, None)

    # val_protein_dataset, val_protein_go_dataset, val_protein_protein_dataset, val_pfam_dataset = None, None, None, None
    (
        val_protein_dataset,
        val_text_cl_dataset,
        val_protein_go_dataset,
        val_protein_protein_dataset,
        val_domain_go_dataset,
        val_domain_pfam_dataset,
    ) = (None, None, None, None, None, None)

    if data_args.use_protein_mlm:
        train_protein_dataset = ProteinDataset(
            data_dir=data_args.data_dir, training=True
        )
        val_protein_dataset = ProteinDataset(
            data_dir=data_args.data_dir, training=False
        )
    if data_args.use_text_cl or data_args.use_text_cl_unsupervised_only:
        train_text_cl_dataset = TextCLDataset(
            data_dir=data_args.data_dir,
            go_split_method=data_args.go_split_method,
            training=True,
        )
        val_text_cl_dataset = TextCLDataset(
            data_dir=data_args.data_dir,
            go_split_method=data_args.go_split_method,
            training=False,
        )
    if data_args.use_protein_go_cl:
        train_protein_go_dataset = ProteinGODataset_OLD(
            data_dir=data_args.data_dir,
            go_split_method=data_args.go_split_method,
            negative_sampling_strategy=data_args.negative_sampling_strategy_protein_go,
            protein_sims_type=data_args.protein_sims_type,
            go_sims_type=data_args.go_sims_type,
            num_neg_samples_protein_go_per_protein=data_args.num_neg_samples_protein_go_per_protein,
            num_neg_samples_protein_go_per_go=data_args.num_neg_samples_protein_go_per_go,
            use_only_goa_proteins=data_args.use_only_goa_proteins,
            use_only_goa_gos=data_args.use_only_goa_gos,
            training=True,
        )
        val_protein_go_dataset = ProteinGODataset_OLD(
            data_dir=data_args.data_dir,
            go_split_method=data_args.go_split_method,
            negative_sampling_strategy=data_args.negative_sampling_strategy_protein_go,
            protein_sims_type=data_args.protein_sims_type,
            go_sims_type=data_args.go_sims_type,
            num_neg_samples_protein_go_per_protein=data_args.num_neg_samples_protein_go_per_protein,
            num_neg_samples_protein_go_per_go=data_args.num_neg_samples_protein_go_per_go,
            use_only_goa_proteins=data_args.use_only_goa_proteins,
            use_only_goa_gos=data_args.use_only_goa_gos,
            training=False,
        )
    # TODO: Mask back in if we need it
    # if data_args.use_protein_protein_cl:
    #     train_protein_protein_dataset = ProteinProteinDataset(
    #         data_dir=data_args.data_dir,
    #         use_only_ppi_proteins=data_args.use_only_ppi_proteins,
    #         training=True
    #     )
    #     val_protein_protein_dataset = ProteinProteinDataset(
    #         data_dir=data_args.data_dir,
    #         use_only_ppi_proteins=data_args.use_only_ppi_proteins,
    #         training=False
    #     )

    # if data_args.use_domain_go_dataset:
    #     train_domain_go_dataset = DomainGODataset(
    #         data_dir=data_args.data_dir,
    #         go_split_method=data_args.go_split_method,
    #         negative_sampling_strategy=data_args.negative_sampling_strategy_domain_go,
    #         domain_sims_type=data_args.domain_sims_type,
    #         go_sims_type=data_args.go_sims_type,
    #         num_neg_samples_domain_go_per_domain=data_args.num_neg_samples_domain_go_per_domain,
    #         num_neg_samples_domain_go_per_go=data_args.num_neg_samples_domain_go_per_go,
    #         use_only_domain_go_domains=data_args.use_only_domain_go_domains,
    #         use_only_domain_go_gos=data_args.use_only_domain_go_gos,
    #         training=True
    #     )
    #     val_domain_go_dataset = DomainGODataset(
    #         data_dir=data_args.data_dir,
    #         go_split_method=data_args.go_split_method,
    #         negative_sampling_strategy=data_args.negative_sampling_strategy_domain_go,
    #         domain_sims_type=data_args.domain_sims_type,
    #         go_sims_type=data_args.go_sims_type,
    #         num_neg_samples_domain_go_per_domain=data_args.num_neg_samples_domain_go_per_domain,
    #         num_neg_samples_domain_go_per_go=data_args.num_neg_samples_domain_go_per_go,
    #         use_only_domain_go_domains=data_args.use_only_domain_go_domains,
    #         use_only_domain_go_gos=data_args.use_only_domain_go_gos,
    #         training=False
    #     )
    # if data_args.use_pfam_dataset:
    #     train_domain_pfam_dataset = DomainPfamDataset(
    #         data_dir=data_args.data_dir,
    #         pfam_split_method=data_args.pfam_split_method,
    #         negative_sampling_strategy=data_args.negative_sampling_strategy_domain_pfam,
    #         domain_sims_type=data_args.domain_sims_type,
    #         pfam_sims_type=data_args.pfam_sims_type,
    #         num_neg_samples_domain_pfam_per_domain=data_args.num_neg_samples_domain_pfam_per_domain,
    #         num_neg_samples_domain_pfam_per_pfam=data_args.num_neg_samples_domain_pfam_per_pfam,
    #         use_only_domain_pfam_domains=data_args.use_only_domain_pfam_domains,
    #         use_only_domain_pfam_pfams=data_args.use_only_domain_pfam_pfams,
    #         training=True
    #     )
    #     val_domain_pfam_dataset = DomainPfamDataset(
    #         data_dir=data_args.data_dir,
    #         pfam_split_method=data_args.pfam_split_method,
    #         negative_sampling_strategy=data_args.negative_sampling_strategy_domain_pfam,
    #         domain_sims_type=data_args.domain_sims_type,
    #         pfam_sims_type=data_args.pfam_sims_type,
    #         num_neg_samples_domain_pfam_per_domain=data_args.num_neg_samples_domain_pfam_per_domain,
    #         num_neg_samples_domain_pfam_per_pfam=data_args.num_neg_samples_domain_pfam_per_pfam,
    #         use_only_domain_pfam_domains=data_args.use_only_domain_pfam_domains,
    #         use_only_domain_pfam_pfams=data_args.use_only_domain_pfam_pfams,
    #         training=False
    #     )

    # return (train_protein_dataset, train_protein_go_dataset, train_protein_protein_dataset, train_pfam_dataset), (val_protein_dataset, val_protein_go_dataset, val_protein_protein_dataset, val_pfam_dataset)

    return (
        train_protein_dataset,
        train_text_cl_dataset,
        train_protein_go_dataset,
        train_protein_protein_dataset,
        train_domain_go_dataset,
        train_domain_pfam_dataset,
    ), (
        val_protein_dataset,
        val_text_cl_dataset,
        val_protein_go_dataset,
        val_protein_protein_dataset,
        val_domain_go_dataset,
        val_domain_pfam_dataset,
    )


def get_all_datasets(
    data_args,
    aaseq_types=["protein", "protein", "protein"],
    text_types=["go", "drugbank", "omim"],
    relation_types=[
        ["all"],
        ["drug_target", "drug_carrier", "drug_transporter", "drug_enzyme"],
        ["all"],
    ],
    task_type="qa",
):
    train_datasets = dict()
    val_datasets = dict()
    for aaseq_type, text_type, relation_type in zip(
        aaseq_types, text_types, relation_types
    ):
        for relation in relation_type:
            print(f"{aaseq_type} {text_type} {relation}")
            (
                train_datasets[text_type + "_" + relation],
                val_datasets[text_type + "_" + relation],
            ) = get_IT_datasets(
                data_args,
                task_type=task_type,
                aaseq_type=aaseq_type,
                text_type=text_type,
                relation_type=relation,
            )
    return train_datasets, val_datasets


def parse_datasets_from_args(data_args):
    """
    Parses relevant data args into arguments that can be used by get_all_datasets
    """

    pass


def get_IT_datasets(
    data_args: DataArgs,
    task_type: str,
    aaseq_type: str = "protein",
    text_type: str = "go",
    relation_type: str = "all",
    testing: bool = False,
    testing_kwargs=None,
):
    """
    Get datasets that are specifically designed for instruction tuning pipeline
    NOTE: As of 07/06, the output datasets are only Protein-GO for prototyping
    TODO: Add optional to only get validation datasets, or make separate function
    """
    neg_per_aaseq = 0
    neg_per_text = 0
    # Bug check on task type:
    task_type = task_type.lower()

    if task_type == "mlm":
        if not data_args.use_protein_mlm:
            return None, None
        train_dataset = ProteinDataset(data_dir=data_args.data_dir, training=True)
        val_dataset = ProteinDataset(data_dir=data_args.data_dir, training=False)
    elif task_type == "qa":
        if not data_args.use_qa:
            return None, None

        negative_sampling_strategy = data_args.negative_sampling_strategy_qa
        if negative_sampling_strategy == "text_only":
            neg_per_aaseq = 0
            neg_per_text = data_args.num_neg_samples_qa
        elif negative_sampling_strategy == "aaseq_only":
            neg_per_aaseq = data_args.num_neg_samples_qa
            neg_per_text = 0
        elif negative_sampling_strategy == "aaseq_text_both":
            neg_per_aaseq = data_args.num_neg_samples_qa // 2
            neg_per_text = data_args.num_neg_samples_qa - neg_per_aaseq
        elif negative_sampling_strategy == "in_batch":
            pass
        else:
            raise ValueError(
                f"unexpected value for 'negative_sampling_strategy_qa': {negative_sampling_strategy}"
            )

        testing_kwargs = {
            "shot_level": data_args.val_split_type,
            "use_preset_negatives": True,
            "num_negatives": data_args.num_neg_samples_qa,  # By structure of the data, will always sample proteins (fixed negatives)
        }

    elif task_type == "retrieval":
        if not data_args.use_retrieval:
            return None, None

        # TODO: Need to make conditional on the type of sampling - e.g. protein-only, protein-go-only, etc.
        neg_per_aaseq = data_args.num_neg_samples_retrieval // 2
        neg_per_text = data_args.num_neg_samples_retrieval - neg_per_aaseq
        negative_sampling_strategy = data_args.negative_sampling_strategy_retrieval

        testing_kwargs = {
            "shot_level": data_args.val_split_type,
            "use_preset_negatives": False,
            "num_negatives": None,
        }

    elif task_type == "caption":
        if not data_args.use_caption:
            return None, None
        neg_per_aaseq = (
            1  # NOTE: This has no effect for captioning, so left as a constant
        )
        neg_per_text = 1
        negative_sampling_strategy = data_args.negative_sampling_strategy_retrieval

        testing_kwargs = {
            "shot_level": data_args.val_split_type,
            "use_preset_negatives": False,
            "num_negatives": None,
        }
    else:
        raise NotImplementedError("Invalid task_type: {}".format(task_type))

    train_dataset, val_dataset = None, None
    # FIXME: Update this as an argument when training for true benchmarking
    text_split_method = (
        data_args.go_split_method
        if text_type == "go"
        else f"random_{text_type}_centric"
    )

    if data_args.use_old_data:
        dataset_kwargs = {
            "data_dir": data_args.data_dir,
            "go_split_method": data_args.go_split_method,
            "negative_sampling_strategy": negative_sampling_strategy,
            "protein_sims_type": data_args.protein_sims_type,
            "go_sims_type": data_args.go_sims_type,
            "num_neg_samples_protein_go_per_protein": neg_per_aaseq,
            "num_neg_samples_protein_go_per_go": neg_per_text,
            "use_only_goa_proteins": data_args.use_only_goa_proteins,
            "use_only_goa_gos": data_args.use_only_goa_gos,
        }

        train_dataset = ProteinGODataset_OLD(
            training=True,
            **dataset_kwargs,
        )
        val_dataset = ProteinGODataset_OLD(
            training=False,
            **dataset_kwargs,
        )
    # Abuse of 'text_type' to represent second amino acid sequence in a
    # AA seq <-> AA seq interaction
    elif aaseq_type == text_type:
        dataset_kwargs = {
            "data_dir": data_args.data_dir,
            "aaseq_type": aaseq_type,
            "relation_type": relation_type,
            "negative_sampling_strategy": negative_sampling_strategy,
            "aaseq_sims_type": data_args.protein_sims_type,
            "num_neg_samples_per_aaseq": neg_per_aaseq,
            "use_perplexity_filtered_set": data_args.use_perplexity_filtered_set,
        }

        train_dataset = AASeqDataset(
            split="train",
            **dataset_kwargs,
        )

        # Currently only implemented AA seq <-> AA seq dataset is PPI, which
        # is only being used for training, and thus doesn't have any testing/validation
        # splits prepared.
        val_dataset = None
    else:
        dataset_kwargs = {
            "data_dir": data_args.data_dir,
            "aaseq_type": aaseq_type,
            "text_type": text_type,
            "relation_type": relation_type,
            "text_split_method": text_split_method,
            "negative_sampling_strategy": negative_sampling_strategy,
            "aaseq_sims_type": data_args.protein_sims_type,
            "text_sims_type": None,
            "num_neg_samples_aaseq_text_per_aaseq": neg_per_aaseq,
            "num_neg_samples_aaseq_text_per_text": neg_per_text,
            "use_only_aaseq_text_aaseqs": data_args.use_only_goa_proteins,
            "use_only_aaseq_text_texts": data_args.use_only_goa_gos,
        }

        train_dataset = AASeqTextDataset(
            split="train",
            **dataset_kwargs,
        )
        val_dataset = AASeqTextDataset(
            split="val" if (not testing) else "test",
            val_split_type=data_args.val_split_type,
            testing_kwargs=testing_kwargs,
            **dataset_kwargs,
        )

    if testing:
        return val_dataset
    else:
        return train_dataset, val_dataset


def get_data_collators(data_args: DataArgs, model_args: ModelArgs) -> Tuple[Any]:
    assert not model_args.is_protein_tokenized, "Not yet supported"
    assert not model_args.is_go_tokenized, "Not yet supported"

    # TODO only create this tokenizer (and others) if necessary
    protein_tokenizer = Alphabet.from_architecture(model_args.protein_tokenizer_name)

    text_tokenizer = AutoTokenizer.from_pretrained(
        data_args.data_dir + f"/model_weights/{model_args.text_tokenizer_name}"
    )
    text_tokenizer.text_tokenizer_name = (
        model_args.text_tokenizer_name
    )  # used to determine how to preprocess sequences

    # protein_mlm_collator, protein_go_collator, protein_protein_collator, pfam_collator = None, None, None, None
    (
        protein_mlm_collator,
        text_cl_collator,
        protein_go_collator,
        protein_protein_collator,
        domain_go_collator,
        domain_pfam_collator,
    ) = (None, None, None, None, None, None)

    if data_args.use_protein_mlm:
        protein_mlm_collator = ProteinMLMCollator(
            data_dir=data_args.data_dir,
            is_protein_tokenized=model_args.is_protein_tokenized,
            protein_tokenizer=protein_tokenizer,
            max_protein_len=model_args.max_protein_len,
            mlm=True,
            masking_strategy=data_args.protein_mlm_masking_strategy,
            mlm_probability=data_args.protein_mlm_probability,
        )
    if data_args.use_text_cl or data_args.use_text_cl_unsupervised_only:
        assert not (
            data_args.use_text_cl and data_args.use_text_cl_unsupervised_only
        ), "Only one of these can be true"
        text_cl_collator = TextCLCollator(
            data_dir=data_args.data_dir,
            go_split_method=data_args.go_split_method,
            go_tokenizer=text_tokenizer,
            max_go_len=model_args.max_text_len,
            unsupervised_only=data_args.use_text_cl_unsupervised_only,
        )
    if data_args.use_protein_go_cl:
        protein_go_collator = ProteinGOCLCollator(
            data_dir=data_args.data_dir,
            go_split_method=data_args.go_split_method,
            negative_sampling_strategy=data_args.negative_sampling_strategy_protein_go,
            protein_sims_type=data_args.protein_sims_type,
            num_neg_samples_protein_go_per_protein=data_args.num_neg_samples_protein_go_per_protein,
            use_only_goa_proteins=data_args.use_only_goa_proteins,
            is_protein_tokenized=model_args.is_protein_tokenized,
            is_go_tokenized=model_args.is_go_tokenized,
            use_go_embeddings=model_args.use_text_embeddings,
            use_protein_embeddings=model_args.use_aaseq_embeddings,
            go_def_col=data_args.go_def_col,
            protein_tokenizer=protein_tokenizer,
            go_tokenizer=text_tokenizer,
            max_protein_len=model_args.max_protein_len,
            max_go_len=model_args.max_text_len,
        )
    if data_args.use_protein_protein_cl:
        protein_protein_collator = ProteinProteinCLCollator(
            data_dir=data_args.data_dir,
            negative_sampling_strategy=data_args.negative_sampling_strategy_protein_protein,
            protein_sims_type=data_args.protein_sims_type,
            num_neg_samples_protein_protein_per_protein=data_args.num_neg_samples_protein_protein_per_protein,
            use_protein_embeddings=model_args.use_aaseq_embeddings,
            is_protein_tokenized=model_args.is_protein_tokenized,
            use_only_ppi_proteins=data_args.use_only_ppi_proteins,
            protein_tokenizer=protein_tokenizer,
            max_protein_len=model_args.max_protein_len,
        )
    # if data_args.use_pfam_cl:
    #     pfam_collator = PfamRelationsCLCollator(
    #         data_dir=data_args.data_dir,
    #         num_domains_sampled_per_pfam=data_args.num_domains_sampled_per_pfam,
    #         negative_sampling_strategy_pfam_protein=data_args.negative_sampling_strategy_pfam_protein,
    #         negative_sampling_strategy_pfam_pfam=data_args.negative_sampling_strategy_pfam_pfam,
    #         protein_sims_type=data_args.protein_sims_type,
    #         pfam_sims_type=data_args.pfam_sims_type,
    #         num_neg_samples_pfam_protein_per_protein=data_args.num_neg_samples_pfam_protein_per_protein,
    #         num_neg_samples_pfam_pfam_per_pfam=data_args.num_neg_samples_pfam_pfam_per_pfam,
    #         use_only_pfam_protein_proteins=data_args.use_only_pfam_protein_proteins,
    #         use_only_pfam_pfam_pfams=data_args.use_only_pfam_pfam_pfams,
    #         is_domain_tokenized=model_args.is_protein_tokenized, # TODO(tom) remove / add data_arg? and check tokenizer creation above
    #         is_protein_tokenized=model_args.is_protein_tokenized,
    #         is_go_tokenized=model_args.is_go_tokenized,
    #         use_go_embeddings=model_args.use_text_embeddings,
    #         use_protein_embeddings=model_args.use_protein_embeddings,
    #         use_domain_embeddings=model_args.use_domain_embeddings,
    #         protein_tokenizer=protein_tokenizer,
    #         go_tokenizer=text_tokenizer,
    #         max_protein_len=model_args.max_protein_len,
    #         max_go_len=data_args.max_go_len,
    #         use_pfam_protein_cl=data_args.use_pfam_protein_cl,
    #         use_pfam_go_cl=data_args.use_pfam_go_cl,
    #         use_pfam_pfam_cl=data_args.use_pfam_pfam_cl
    #     )
    if data_args.use_domain_go_cl:
        domain_go_collator = DomainGOCLCollator(
            data_dir=data_args.data_dir,
            go_split_method=data_args.go_split_method,
            negative_sampling_strategy=data_args.negative_sampling_strategy_domain_go,
            domain_sims_type=data_args.domain_sims_type,
            num_neg_samples_domain_go_per_domain=data_args.num_neg_samples_domain_go_per_domain,
            use_only_domain_go_domains=data_args.use_only_domain_go_domains,
            is_domain_tokenized=model_args.is_protein_tokenized,
            is_go_tokenized=model_args.is_go_tokenized,
            use_go_embeddings=model_args.use_text_embeddings,
            use_domain_embeddings=model_args.use_aaseq_embeddings,
            go_def_col=data_args.go_def_col,
            domain_tokenizer=protein_tokenizer,
            go_tokenizer=text_tokenizer,
            max_domain_len=model_args.max_protein_len,
            max_go_len=model_args.max_text_len,
        )
    if data_args.use_domain_pfam_cl:
        domain_pfam_collator = DomainPfamCLCollator(
            data_dir=data_args.data_dir,
            pfam_split_method=data_args.pfam_split_method,
            negative_sampling_strategy=data_args.negative_sampling_strategy_domain_pfam,
            domain_sims_type=data_args.domain_sims_type,
            num_neg_samples_domain_pfam_per_domain=data_args.num_neg_samples_domain_pfam_per_domain,
            use_only_domain_pfam_domains=data_args.use_only_domain_pfam_domains,
            is_domain_tokenized=model_args.is_protein_tokenized,
            is_pfam_tokenized=model_args.is_pfam_tokenized,
            use_pfam_embeddings=model_args.use_text_embeddings,
            use_domain_embeddings=model_args.use_aaseq_embeddings,
            domain_tokenizer=protein_tokenizer,
            pfam_tokenizer=text_tokenizer,
            max_domain_len=model_args.max_protein_len,
            max_pfam_len=model_args.max_text_len,
        )

    # return protein_mlm_collator, protein_go_collator, protein_protein_collator, pfam_collator
    return (
        protein_mlm_collator,
        text_cl_collator,
        protein_go_collator,
        protein_protein_collator,
        domain_go_collator,
        domain_pfam_collator,
    )


def get_data_collators_IT(data_args: DataArgs, model_args: ModelArgs) -> Tuple[Any]:
    assert not model_args.is_protein_tokenized, "Not yet supported"
    assert not model_args.is_go_tokenized, "Not yet supported"

    # TODO only create this tokenizer (and others) if necessary
    protein_tokenizer = Alphabet.from_architecture(model_args.protein_tokenizer_name)

    text_tokenizer = AutoTokenizer.from_pretrained(
        data_args.data_dir + f"/model_weights/{model_args.text_tokenizer_name}"
    )
    # text_tokenizer = AutoTokenizer.from_pretrained(f'{model_args.text_tokenizer_name}')
    text_tokenizer.text_tokenizer_name = (
        model_args.text_tokenizer_name
    )  # used to determine how to preprocess sequences

    protein_mlm_collator, qa_collator, retrieval_collator, caption_collator = (
        None,
        None,
        None,
        None,
    )

    if data_args.use_protein_mlm:
        print("Get MLM")
        protein_mlm_collator = ProteinMLMCollator(
            data_dir=data_args.data_dir,
            is_protein_tokenized=model_args.is_protein_tokenized,
            protein_tokenizer=protein_tokenizer,
            max_protein_len=model_args.max_protein_len,
            mlm=True,
            masking_strategy=data_args.protein_mlm_masking_strategy,
            mlm_probability=data_args.protein_mlm_probability,
        )
    if data_args.use_qa:
        print("Get QA")
        qa_collator = QACollator(
            data_dir=data_args.data_dir,
            go_split_method=data_args.go_split_method,
            protein_sims_type=data_args.protein_sims_type,
            is_protein_tokenized=model_args.is_protein_tokenized,
            is_text_tokenized=model_args.is_go_tokenized,
            use_text_embeddings=model_args.use_text_embeddings,
            use_protein_embeddings=model_args.use_aaseq_embeddings,
            go_def_col=data_args.go_def_col,
            protein_tokenizer=protein_tokenizer,
            text_tokenizer=text_tokenizer,
            max_protein_len=model_args.max_protein_len,
            max_text_len=model_args.max_text_len,
            sample_num_examples=data_args.sample_num_instruction_examples,
        )
    if data_args.use_retrieval:
        print("Get Retrieval")
        # retrieval_collator = AASeqTextITCollatorRetrieval_GOTest(
        retrieval_collator = RetrievalCollator(
            train_retrieval_lm=model_args.train_retrieval_lm,
            data_dir=data_args.data_dir,
            go_split_method=data_args.go_split_method,
            protein_sims_type=data_args.protein_sims_type,
            is_protein_tokenized=model_args.is_protein_tokenized,
            is_text_tokenized=model_args.is_go_tokenized,
            use_text_embeddings=model_args.use_text_embeddings,
            use_protein_embeddings=model_args.use_aaseq_embeddings,
            go_def_col=data_args.go_def_col,
            protein_tokenizer=protein_tokenizer,
            text_tokenizer=text_tokenizer,
            max_protein_len=model_args.max_protein_len,
            max_text_len=model_args.max_text_len,
            testing_only=False,
            sample_num_examples=data_args.sample_num_instruction_examples,
        )
        retrieval_eval_collator = RetrievalCollator(
            train_retrieval_lm=model_args.train_retrieval_lm,
            data_dir=data_args.data_dir,
            go_split_method=data_args.go_split_method,
            protein_sims_type=data_args.protein_sims_type,
            is_protein_tokenized=model_args.is_protein_tokenized,
            is_text_tokenized=model_args.is_go_tokenized,
            use_text_embeddings=model_args.use_text_embeddings,
            use_protein_embeddings=model_args.use_aaseq_embeddings,
            go_def_col=data_args.go_def_col,
            protein_tokenizer=protein_tokenizer,
            text_tokenizer=text_tokenizer,
            max_protein_len=model_args.max_protein_len,
            max_text_len=model_args.max_text_len,
            testing_only=True,
            sample_num_examples=False,  # Set to False for evaluation
        )
    if data_args.use_caption:
        print("Get Caption")
        caption_collator = CaptionCollator(
            data_dir=data_args.data_dir,
            go_split_method=data_args.go_split_method,
            protein_sims_type=data_args.protein_sims_type,
            is_protein_tokenized=model_args.is_protein_tokenized,
            is_text_tokenized=model_args.is_go_tokenized,
            use_text_embeddings=model_args.use_text_embeddings,
            use_protein_embeddings=model_args.use_aaseq_embeddings,
            go_def_col=data_args.go_def_col,
            protein_tokenizer=protein_tokenizer,
            text_tokenizer=text_tokenizer,
            max_protein_len=model_args.max_protein_len,
            max_text_len=model_args.max_text_len,
            sample_num_examples=data_args.sample_num_instruction_examples,
        )

    return (
        protein_mlm_collator,
        qa_collator,
        retrieval_collator,
        retrieval_eval_collator,
        caption_collator,
    )


def get_data_collators_IT_new(
    data_args: DataArgs,
    model_args: ModelArgs,
    aaseq_types=["protein", "protein", "protein"],
    text_types=["go", "drugbank", "omim"],
    relation_types=[
        ["all"],
        ["drug_target", "drug_transporter", "drug_enzyme", "drug_carrier"],
        ["all"],
    ],
    evaluation=False,
) -> Tuple[Any]:
    assert not model_args.is_protein_tokenized, "Not yet supported"
    assert not model_args.is_go_tokenized, "Not yet supported"

    assert not isinstance(aaseq_types, str), "aaseq_types must be a list"
    assert not isinstance(text_types, str), "text_types must be a list"
    assert not isinstance(relation_types, str), "relation_types must be a list of lists"

    # TODO only create this tokenizer (and others) if necessary
    protein_tokenizer = Alphabet.from_architecture(model_args.protein_tokenizer_name)

    text_tokenizer = AutoTokenizer.from_pretrained(
        data_args.data_dir + f"/model_weights/{model_args.text_tokenizer_name}"
    )
    text_tokenizer.text_tokenizer_name = (
        model_args.text_tokenizer_name
    )  # used to determine how to preprocess sequences

    protein_mlm_collator, qa_collators, retrieval_collators, caption_collators = (
        None,
        None,
        None,
        None,
    )

    if data_args.use_protein_mlm:
        print("Get MLM")
        protein_mlm_collator = ProteinMLMCollator(
            data_dir=data_args.data_dir,
            is_protein_tokenized=model_args.is_protein_tokenized,
            protein_tokenizer=protein_tokenizer,
            max_protein_len=model_args.max_protein_len,
            mlm=True,
            masking_strategy=data_args.protein_mlm_masking_strategy,
            mlm_probability=data_args.protein_mlm_probability,
        )
    if data_args.use_qa:
        print("Get QA")

        qa_collators = dict()
        for aaseq_type, text_type, relation_type in zip(
            aaseq_types, text_types, relation_types
        ):
            # FIXME: Update this as an argument when training for true benchmarking
            text_split_method = (
                data_args.go_split_method
                if text_type == "go"
                else f"random_{text_type}_centric"
            )
            for relation in relation_type:
                qa_collators[text_type + "_" + relation] = QACollator(
                    data_dir=data_args.data_dir,
                    aaseq_type=aaseq_type,
                    text_type=text_type,
                    relation_type=relation,
                    text_variant_type=data_args.text_variant_type,  # TODO: If in the future we want sampling for this parameter, can enable it in the collator
                    aaseq_sims_type=data_args.protein_sims_type,
                    is_aaseq_tokenized=model_args.is_protein_tokenized,
                    is_text_tokenized=model_args.is_go_tokenized,
                    use_text_embeddings=model_args.use_text_embeddings,
                    use_aaseq_embeddings=model_args.use_aaseq_embeddings,
                    aaseq_tokenizer=protein_tokenizer,
                    text_tokenizer=text_tokenizer,
                    max_aaseq_len=model_args.max_protein_len,
                    max_text_len=model_args.max_text_len,
                    num_examples=data_args.num_instruction_examples,
                    sample_num_examples=data_args.sample_num_instruction_examples,
                    evaluation=evaluation,
                )

    if data_args.use_retrieval:
        print("Get Retrieval")

        retrieval_collators = dict()
        for aaseq_type, text_type, relation_type in zip(
            aaseq_types, text_types, relation_types
        ):
            # FIXME: Update this as an argument when training for true benchmarking
            text_split_method = (
                data_args.go_split_method
                if text_type == "go"
                else f"random_{text_type}_centric"
            )
            for relation in relation_type:
                retrieval_collators[text_type + "_" + relation] = RetrievalCollator(
                    train_retrieval_lm=model_args.train_retrieval_lm,
                    data_dir=data_args.data_dir,
                    aaseq_type=aaseq_type,
                    text_type=text_type,
                    relation_type=relation,
                    text_variant_type=data_args.text_variant_type,  # TODO: If in the future we want sampling for this parameter, can enable it in the collator
                    aaseq_sims_type=data_args.protein_sims_type,
                    is_aaseq_tokenized=model_args.is_protein_tokenized,
                    is_text_tokenized=model_args.is_go_tokenized,
                    use_text_embeddings=model_args.use_text_embeddings,
                    use_aaseq_embeddings=model_args.use_aaseq_embeddings,
                    aaseq_tokenizer=protein_tokenizer,
                    text_tokenizer=text_tokenizer,
                    max_aaseq_len=model_args.max_protein_len,
                    max_text_len=model_args.max_text_len,
                    num_examples=data_args.num_instruction_examples,
                    sample_num_examples=data_args.sample_num_instruction_examples,
                    evaluation=evaluation,
                )
    if data_args.use_caption:
        print("Get Caption")
        # caption_collator = CaptionCollator(
        #     data_dir=data_args.data_dir,
        #     go_split_method=data_args.go_split_method,
        #     protein_sims_type=data_args.protein_sims_type,
        #     is_protein_tokenized=model_args.is_protein_tokenized,
        #     is_text_tokenized=model_args.is_go_tokenized,
        #     use_text_embeddings=model_args.use_text_embeddings,
        #     use_protein_embeddings=model_args.use_aaseq_embeddings,
        #     go_def_col=data_args.go_def_col,
        #     protein_tokenizer=protein_tokenizer,
        #     text_tokenizer=text_tokenizer,
        #     max_protein_len=model_args.max_protein_len,
        #     max_text_len=model_args.max_text_len,
        # )
        caption_collators = dict()
        for aaseq_type, text_type, relation_type in zip(
            aaseq_types, text_types, relation_types
        ):
            # FIXME: Update this as an argument when training for true benchmarking
            text_split_method = (
                data_args.go_split_method
                if text_type == "go"
                else f"random_{text_type}_centric"
            )
            for relation in relation_type:
                caption_collators[text_type + "_" + relation] = CaptionCollator(
                    data_dir=data_args.data_dir,
                    aaseq_type=aaseq_type,
                    text_type=text_type,
                    relation_type=relation,
                    text_variant_type=data_args.text_variant_type,  # TODO: If in the future we want sampling for this parameter, can enable it in the collator
                    aaseq_sims_type=data_args.protein_sims_type,
                    is_aaseq_tokenized=model_args.is_protein_tokenized,
                    is_text_tokenized=model_args.is_go_tokenized,
                    use_text_embeddings=model_args.use_text_embeddings,
                    use_aaseq_embeddings=model_args.use_aaseq_embeddings,
                    aaseq_tokenizer=protein_tokenizer,
                    text_tokenizer=text_tokenizer,
                    max_aaseq_len=model_args.max_protein_len,
                    max_text_len=model_args.max_text_len,
                    num_examples=data_args.num_instruction_examples,
                    sample_num_examples=data_args.sample_num_instruction_examples,
                    evaluation=evaluation,
                )

    return protein_mlm_collator, qa_collators, retrieval_collators, caption_collators


def get_datasets_and_collators_from_config(
    data_args: DataArgs,
    model_args: ModelArgs,
    train_args: TrainArgs,
    evaluation: bool = False,
) -> Tuple[Dict, Dict, Tuple]:

    protein_tokenizer = Alphabet.from_architecture(model_args.protein_tokenizer_name)

    text_tokenizer = AutoTokenizer.from_pretrained(
        data_args.data_dir + f"/model_weights/{model_args.text_tokenizer_name}"
    )
    text_tokenizer.text_tokenizer_name = (
        model_args.text_tokenizer_name
    )  # used to determine how to preprocess sequences

    # Load datasets. Protein MLM datasets directly, IT datasets via config.
    train_protein_dataset, val_protein_dataset = get_IT_datasets(
        data_args, task_type="mlm"
    )

    config = ITMultiDatasetConfig.load_from_yaml(data_args.it_data_config_yml)
    it_datasets, it_collators = config.get_datasets_and_collators(
        data_args, model_args, protein_tokenizer, text_tokenizer, evaluation=evaluation
    )

    # Load MLM collator, repackage IT collators.
    if data_args.use_protein_mlm:
        protein_mlm_collator = ProteinMLMCollator(
            data_dir=data_args.data_dir,
            is_protein_tokenized=model_args.is_protein_tokenized,
            protein_tokenizer=protein_tokenizer,
            max_protein_len=model_args.max_protein_len,
            mlm=True,
            masking_strategy=data_args.protein_mlm_masking_strategy,
            mlm_probability=data_args.protein_mlm_probability,
        )
    else:
        protein_mlm_collator = None
    qa_collators, retrieval_collators, caption_collators = (
        package_collators_for_trainer(it_collators)
    )

    # packaged_train_datasets = (
    #     train_protein_dataset,
    #     it_datasets["train"]["qa"],
    #     it_datasets["train"]["retrieval"],
    #     it_datasets["train"]["caption"],
    # )
    # packaged_validation_datasets = (
    #     train_protein_dataset,
    #     it_datasets["validation"]["qa"],
    #     it_datasets["validation"]["retrieval"],
    #     it_datasets["validation"]["caption"],
    # )

    if data_args.shuffle_seed_metadataset is None:
        seed_arguments = {"shuffle": False, "seed": None}
    else:
        seed_arguments = {"shuffle": True, "seed": data_args.shuffle_seed_metadataset}

    packaged_train_datasets = (
        train_protein_dataset,
        (
            MetaDataset(
                dataset_dict=it_datasets["train"]["qa"],
                batch_size=train_args.qa_batch_size,
                **seed_arguments,
            )
            if data_args.use_qa
            else None
        ),
        (
            MetaDataset(
                dataset_dict=it_datasets["train"]["retrieval"],
                batch_size=train_args.retrieval_batch_size,
                **seed_arguments,
            )
            if data_args.use_retrieval
            else None
        ),
        (
            MetaDataset(
                dataset_dict=it_datasets["train"]["caption"],
                batch_size=train_args.caption_batch_size,
                **seed_arguments,
            )
            if data_args.use_caption
            else None
        ),
    )
    packaged_validation_datasets = (
        val_protein_dataset,
        (
            MetaDataset(
                dataset_dict=it_datasets["validation"]["qa"],
                batch_size=train_args.qa_batch_size,
                **seed_arguments,
            )
            if data_args.use_qa
            else None
        ),
        (
            MetaDataset(
                dataset_dict=it_datasets["validation"]["retrieval"],
                batch_size=train_args.retrieval_batch_size,
                **seed_arguments,
            )
            if data_args.use_retrieval
            else None
        ),
        (
            MetaDataset(
                dataset_dict=it_datasets["validation"]["caption"],
                batch_size=train_args.caption_batch_size,
                **seed_arguments,
            )
            if data_args.use_caption
            else None
        ),
    )

    packaged_collators = (
        protein_mlm_collator,
        MetaCollator(qa_collators) if data_args.use_qa else None,
        MetaCollator(retrieval_collators) if data_args.use_retrieval else None,
        MetaCollator(caption_collators) if data_args.use_caption else None,
    )

    return packaged_train_datasets, packaged_validation_datasets, packaged_collators


###########
# loss utils
###########
def get_mlm_loss(
    logits: torch.Tensor,
    true_tokens: torch.Tensor,
    calc_accuracy=True,
    calc_per_token_accuracy=False,
    calc_perplexity=True,
):
    """
    Args:
        result: dict storing results
        logits: predicted logits for all tokens, shape [batch_size, seq_len, vocab_size]
        true_tokens: true labels for all tokens, shape [batch_size, seq_len], most (all positions that are not masked) are -100
    """
    result = dict()

    # Flatten batch_size and seq_len dimensions to match CrossEntropyLoss API
    logits = logits.reshape(-1, logits.shape[-1])
    true_tokens = true_tokens.flatten()
    loss = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)(logits, true_tokens)
    # TODO: Investigate where the parallelization is happening
    result["loss"] = loss.mean()

    if calc_accuracy or calc_per_token_accuracy or calc_perplexity:
        mask = true_tokens != -100
        mask_logits = logits[mask]
        mask_tokens = true_tokens[mask]

    if calc_accuracy:
        result["accuracy"] = (
            (mask_logits.argmax(-1) == mask_tokens).float().mean().item()
        )

    if calc_per_token_accuracy:
        indices, counts = torch.unique(mask_tokens.cpu(), return_counts=True)
        label_counts = torch.zeros(mask_logits.shape[1])
        label_counts.scatter_(
            dim=0, index=indices.type(torch.int64), src=counts.float()
        )

        per_token_accuracy = torch.zeros(mask_logits.shape[1])
        per_token_accuracy.scatter_(
            dim=0,
            index=mask_tokens.cpu(),
            src=(mask_logits.argmax(-1) == mask_tokens).float().cpu(),
        )

        per_token_accuracy = per_token_accuracy / label_counts
        per_token_accuracy[label_counts == 0] = 0

        result["per_token_accuracy"] = per_token_accuracy
        result["label_counts"] = label_counts

    if calc_perplexity:
        perplexity = torch.exp(
            loss
        ).mean()  # E[exp(-log(p_{x_{i\in M}} | x_{j\not \in M} \cup \tilde{x}_{i\in M}))]
        result["perplexity"] = perplexity.item()

    return result


def get_kepler_loss(
    scores: torch.Tensor,
    margin: float,
    is_neg: bool,
):
    """
    Adapted from DRAGON.  Note we did not use the original KEPLER loss, but a reversely optimized objective that aims to achieve higher scores for positive pairs.  This function computes the component \log\sigma(\gamma+d) of the loss.
    For each sample, we have:
    $$
    \ell_{CL} = \ell_{CL}=-\log \sigma(\gamma+d(h, r, t)) + \frac{1}{n}\sum_{i=1}^n \log \sigma\left(\gamma + d\left(h_i^{\prime}, r, t_i^{\prime}\right)\right)
    $$
    """
    if is_neg:
        return F.logsigmoid(-margin + (1 - scores)).mean()
        # return F.logsigmoid(- margin + torch.tanh(1 - scores)).mean()
    else:
        return F.logsigmoid(margin - (1 - scores)).mean()
        # return F.logsigmoid(margin - torch.tanh(1 - scores)).mean()
    # TODO: Clamp on the scores


def get_cl_metrics(pos_scores: np.ndarray, neg_scores: np.ndarray):
    """
    Calculate rank-based metrics to get some grasp of how well we are doing on CL.  Right now we are only considering AUPR and AUROC.  Can consider others if needed.
    """
    # TODO: Add more metrics: Recall@K, MRR, etc.
    pos_count = len(pos_scores)
    neg_count = len(neg_scores)
    labels = np.array([1] * pos_count + [0] * neg_count)
    auroc = roc_auc_score(labels, np.concatenate([pos_scores, neg_scores]))
    auprc = average_precision_score(labels, np.concatenate([pos_scores, neg_scores]))

    return pos_count, neg_count, auroc, auprc


def get_retrieval_scores(cdict):
    pos_s_z, pos_t_z = (
        cdict["positive"]["sequence"].detach().clone().cpu(),
        cdict["positive"]["text"].detach().clone().cpu(),
    )
    neg_s_z, neg_t_z = (
        cdict["negative"]["sequence"].detach().clone().cpu(),
        cdict["negative"]["text"].detach().clone().cpu(),
    )

    pos_scores = F.cosine_similarity(pos_s_z, pos_t_z)
    neg_scores = F.cosine_similarity(neg_s_z, neg_t_z)

    return pos_scores, neg_scores


def get_retrieval_scores_inbatch(cdict):
    # TODO: Fix to work with projection layers in the contrastive learning loss
    pos_s_z, pos_t_z = (
        cdict["positive"]["sequence"].detach().clone().cpu(),
        cdict["positive"]["text"].detach().clone().cpu(),
    )

    # Compute cosine similarity
    pos_s_z_fp = pos_s_z.to(dtype=torch.float32)
    pos_t_z_fp = pos_t_z.to(dtype=torch.float32)

    scores_mat = torch.matmul(
        F.normalize(pos_s_z_fp, dim=-1), F.normalize(pos_t_z_fp, dim=-1).t()
    )

    pos_scores = torch.diagonal(scores_mat).clone()  # Already flattens
    # Hack to get off-diagonal elements:
    M = scores_mat.clone()
    M.diagonal().fill_(-100.0)
    neg_scores = M.flatten()
    neg_scores = neg_scores[neg_scores > -99.0]  # Cancels out all diagonal elements

    return pos_scores, neg_scores


# OLD:
# l_yn = logits.softmax(dim = -1)[:,0,[yes_token, no_token]] # Now size (B, 2)
# yes_score = (l_yn[:,0] - l_yn[:,1] + 1) / 2 # Normalizes to [0,1] range
# #import ipdb; ipdb.set_trace()
# return yes_score # Size (B,)


def get_qa_scores_OLD(model_out, yes_token: int, no_token: int, padding_token: int):

    # Idea: get yes and no scores, take difference b/w them, interpolate to [0,1] range
    #   - This will then constitute the binary score of the model, means we don't need to store logits
    # Prepare and extract predictions:
    preds = model_out["outputs"].logits.softmax(dim=-1).detach().clone().cpu()

    # Prepare labels:
    y_tok_total = model_out["text_toks"].detach().clone().cpu()
    y_toks_inds = get_final_tokens(y_tok_total, padding_token=padding_token)
    y_toks = y_tok_total[torch.arange(y_tok_total.shape[0]), y_toks_inds]

    pred_tok_total = preds.argmax(dim=-1)
    # NOTE: Must subtract 1 from y_toks_inds because the output is shifted due to causal language modeling
    #   - Very difficult bug to find, but consider the documentation in BioGPT
    pred_toks = pred_tok_total[torch.arange(pred_tok_total.shape[0]), (y_toks_inds - 1)]

    return pred_toks, y_toks


def get_qa_scores(model_out, padding_token=None, answer_token=None):
    preds = model_out["outputs"].logits.softmax(dim=-1).detach().clone().cpu()

    # Prepare labels:
    y_tok_total = model_out["text_toks"].detach().clone().cpu()
    if padding_token is not None:
        y_toks_inds = get_final_tokens(y_tok_total, padding_token=padding_token)
    elif answer_token is not None:
        y_toks_inds = get_after_answer_tokens(y_tok_total, answer_token=answer_token)
    else:
        raise ValueError(
            "One of padding_token or answer_token for get_qa_metrics must not be None"
        )

    y_toks = y_tok_total[torch.arange(y_tok_total.shape[0]), y_toks_inds]

    pred_tok_total = preds.argmax(dim=-1)
    # pred_toks_inds = get_final_tokens(pred_tok_total, padding_token = padding_token)
    # NOTE: Must subtract 1 from y_toks_inds because the output is shifted due to causal language modeling
    #   - Very difficult bug to find, but consider the documentation in BioGPT
    pred_toks = pred_tok_total[torch.arange(pred_tok_total.shape[0]), (y_toks_inds - 1)]

    return pred_toks.detach().clone().cpu(), y_toks.detach().clone().cpu()


def get_qa_metrics_OLD(logits, labels, yes_token: int, no_token: int, causal_qa=True):
    """
    Deprecated, don't use; keeping here for logging previous approach
    """
    # TODO: make more robust

    if causal_qa:
        preds = get_qa_scores(
            logits.detach().clone().cpu(), yes_token=yes_token, no_token=no_token
        )
    else:
        preds = logits.softmax(dim=-1)[:, 1].detach().clone().cpu()
    y = (labels.detach().clone().cpu()[:, 1] == yes_token).int()

    preds, y = preds.numpy(), y.numpy()

    auroc = roc_auc_score(y, preds)
    auprc = average_precision_score(y, preds)

    return auroc, auprc


def get_final_tokens(text_toks, padding_token: int):
    num_pads = (text_toks == padding_token).sum(dim=-1)
    # NOTE: Why subtract 2 below?
    #   Reason: with the subtract w num_pads, we basically find the index at which the last 1 occurs
    #   Subtract 2 bc: 2 = 1 (0-indexing adjustment) + 1 (adjust for EOS token)
    inds = (torch.full_like(num_pads, text_toks.shape[1]) - num_pads) - 2
    return inds


def get_after_answer_tokens(text_toks, answer_token: int, get_final=True):
    where_answer = (text_toks == answer_token).nonzero()

    if get_final:
        found_map = []
        for i in range(text_toks.shape[0]):
            highest_ind = (
                where_answer[where_answer[:, 0] == i, 1].max().item()
            )  # Finds maximum column with answer_idx that was found in row i
            found_map.append(highest_ind)
        found_map = torch.tensor(found_map, device=text_toks.device)
        return found_map + 1
    else:
        return where_answer[:, 1] + 1


def get_qa_metrics(
    model_out,
    yes_token: int,
    no_token: int,
    padding_token: int = None,
    answer_token: int = None,
):
    preds = model_out["outputs"].logits.softmax(dim=-1).detach().clone().cpu()

    # Prepare labels:
    y_tok_total = model_out["text_toks"].detach().clone().cpu()
    if padding_token is not None:
        y_toks_inds = get_final_tokens(y_tok_total, padding_token=padding_token)
    elif answer_token is not None:
        y_toks_inds = get_after_answer_tokens(y_tok_total, answer_token=answer_token)
    else:
        raise ValueError(
            "One of padding_token or answer_token for get_qa_metrics must not be None"
        )

    y_toks = y_tok_total[torch.arange(y_tok_total.shape[0]), y_toks_inds]

    pred_tok_total = preds.argmax(dim=-1)
    # pred_toks_inds = get_final_tokens(pred_tok_total, padding_token = padding_token)
    # NOTE: Must subtract 1 from y_toks_inds because the output is shifted due to causal language modeling
    #   - Very difficult bug to find, but consider the documentation in BioGPT
    pred_toks = pred_tok_total[torch.arange(pred_tok_total.shape[0]), (y_toks_inds - 1)]
    # pred_toks = pred_tok_total[torch.arange(pred_tok_total.shape[0]),(y_toks_inds)]

    # import ipdb; ipdb.set_trace()

    # TODO: Take this out later if it's an efficiency problem
    yes = (y_toks == yes_token).sum()
    no = (y_toks == no_token).sum()
    nel = y_toks.numel()
    # if (yes + no) != nel:
    #     import ipdb; ipdb.set_trace()
    assert (
        yes + no
    ) == nel, f"Tokens in predictions other than yes {yes} and no {no} (compared to {nel} total)"

    acc = (pred_toks == y_toks).float().mean()
    f1 = f1_score(y_toks.numpy(), pred_toks.numpy(), average="macro")

    return acc, f1


def get_qa_metrics_from_preds(
    pred_toks, y_toks, yes_token: int, no_token: int, padding_token: int = None
):
    # TODO: Take this out later if it's an efficiency problem
    yes = (y_toks == yes_token).sum()
    no = (y_toks == no_token).sum()
    nel = y_toks.numel()
    assert (
        yes + no
    ) == nel, f"Tokens in y other than yes {yes} and no {no} (compared to {nel} total)"

    pred_yes = (pred_toks == yes_token).sum()
    pred_no = (pred_toks == no_token).sum()
    if pred_yes + pred_no != nel:
        print(
            f"In QA eval, received {nel - pred_yes - pred_no} / {nel} predictions that are not yes/no"
        )

    acc = (pred_toks == y_toks).float().mean()
    f1 = f1_score(y_toks.numpy(), pred_toks.numpy(), average="macro")

    return acc, f1


def get_caption_output_tokens(pred_toks, input_tok_mask, eos_id: int):
    """Get the tokens corresponding to model's output for a caption task.

    Returns all tokens after the last [ANSWER] token and up to the first
    [EOS] token.

    Args:
    pred_toks -- torch.Tensor containing output prediction
                 tokens (i.e. out['logits'].argmax(dim=-1))
    input_tok_mask -- torch.Tensor of booleans giving mask defining tokens
                    over which loss was computed
                      (i.e. (out['text_toks'] > -100))
    eos_id -- int ID of [EOS] token

    """
    first_idx = torch.nonzero(input_tok_mask)[0].item()
    pred_toks = pred_toks[first_idx:]
    eos_idxs = torch.nonzero(pred_toks == eos_id).squeeze(1)

    # Handle the case where the model never output EOS
    last_idx = len(pred_toks)
    if len(eos_idxs) > 0:
        last_idx = eos_idxs[0].item()
    return pred_toks[:last_idx]


def get_caption_pairs(out, tokenizer) -> List[Tuple[str, str]]:
    """Gets a list of [(output, reference)] pairs given output of a caption task batch.

    Returns a list of [(generated_caption, reference_caption)] pairs given
    the output of a single `model.train_forward` on a caption task batch,

    Args:
    out -- the raw output from `model.train_forward`
    tokenizer -- tokenizer used by model
    """
    # Argmax over vocab dimension
    pred_tokens = out["outputs"].logits.argmax(dim=-1).detach().cpu()
    # TODO(rcalef): explicitly define the no backprop token as a constant somewhere
    mask = (out["full_labels"] > -100).detach().cpu()

    # "batch_decode" just does a list-comprehension, so no efficiency gain from using
    # it and we also need to apply mask before decoding (i.e. at the token level)
    decoded_pairs = []
    for i in torch.arange(pred_tokens.shape[0]):
        # -1 to drop EOS token
        ref_tokens = out["text_toks"][i][mask[i]][:-1].detach().cpu()
        ref_text = tokenizer.decode(ref_tokens)

        answer_tokens = get_caption_output_tokens(
            pred_tokens[i], mask[i], tokenizer.eos_token_id
        )
        answer_text = tokenizer.decode(answer_tokens)

        decoded_pairs.append((answer_text, ref_text))
    return decoded_pairs


def get_caption_metrics_from_preds(prediction_pairs, bert_lang="en-sci", **kwargs):
    """Calculate BERTScore metrics from a set of caption outputs-reference pairs.

    Returns BERTScore metrics (precision, recall, F1) averaged over all pairs.
    Default BERT model is SciBERT.

    Args:
    prediction_pairs -- list of (generated, reference) caption pairs
    bert_lang -- name of 'language' BERT model to use, defaults to "en-sci" for
                 SciBERT but can switch to 'en' for a general English language
                 model
    kwargs -- additional kwargs for BERTScore, see:
              https://huggingface.co/spaces/evaluate-metric/bertscore
    """

    # TODO (rcalef): could set this up to only be loaded once (e.g. global variable)
    #               if we find that this ends up getting called in a loop
    bertscore = evaluate.load("bertscore")
    predictions, references = zip(*prediction_pairs)
    results = bertscore.compute(
        predictions=predictions, references=references, lang=bert_lang, **kwargs
    )

    return {
        metric: np.mean(values)
        for (metric, values) in results.items()
        if metric != "hashcode"
    }


###########
# training utils
###########
def get_root_logger(fname=None, file=True, local_rank=None, global_rank=None):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    if local_rank is None:
        global_rank = global_rank if global_rank is not None else local_rank
        format = logging.Formatter("[%(asctime)-10s] %(message)s", "%m/%d/%Y %H:%M:%S")
    else:
        format = logging.Formatter(
            "[%(asctime)-10s] <LR = "
            + f"{local_rank}"
            + " GR = "
            + f"{global_rank}"
            + "> %(message)s",
            "%m/%d/%Y %H:%M:%S",
        )

    if file:
        handler = logging.FileHandler(fname, mode="a")
        handler.setFormatter(format)
        logger.addHandler(handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(format)
    logger.addHandler(logging.StreamHandler())

    return logger


# TODO: Investigate if this is needed instead of simply torch.device('cuda')
def get_device(args, is_cuda_available):
    """Adapted from DRAGON. Get the devices to put the data and the model based on whether to use GPUs and, if so, how many of them are available."""

    if args.local_rank == -1 and is_cuda_available:
        device = torch.device("cuda:0")
    elif args.local_rank == -1:
        device = torch.device("cpu")
    else:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # torch.distributed.init_process_group(backend="nccl")

    args.world_size = torch.distributed.get_world_size() if args.local_rank != -1 else 1
    print(
        "Process rank: %s, device: %s, distributed training: %s, world_size: %s"
        % (args.local_rank, device, bool(args.local_rank != -1), args.world_size),
        file=sys.stderr,
    )

    return device


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_training_steps: int,
    num_protein_encoder_warmup_steps: int,
    num_text_encoder_warmup_steps: int,
    num_embedding_warmup_steps: int,
    num_decoder_warmup_steps: int,
    num_contrastive_warmup_steps: int,
    last_epoch=-1,
):
    """
    Adapted from OntoProtein.  First increases linearly to lr, then decreases linearly to 0.
    """

    def lr_lambda_for_protein_encoder(current_step: int):
        if current_step < num_protein_encoder_warmup_steps:
            return float(current_step) / float(max(1, num_protein_encoder_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_protein_encoder_warmup_steps)),
        )

    def lr_lambda_for_text_encoder(current_step: int):
        if current_step < num_text_encoder_warmup_steps:
            return float(current_step) / float(max(1, num_text_encoder_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_text_encoder_warmup_steps)),
        )

    def lr_lambda_for_embedding(current_step: int):
        if current_step < num_embedding_warmup_steps:
            return float(current_step) / float(max(1, num_embedding_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_embedding_warmup_steps)),
        )

    def lr_lambda_for_decoder(current_step: int):
        if current_step < num_decoder_warmup_steps:
            return float(current_step) / float(max(1, num_decoder_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_decoder_warmup_steps)),
        )

    def lr_lambda_for_contrastive(current_step: int):
        if current_step < num_contrastive_warmup_steps:
            return float(current_step) / float(max(1, num_contrastive_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_contrastive_warmup_steps)),
        )

    # Following order in train.create_optimizer
    return LambdaLR(
        optimizer,
        [
            lr_lambda_for_protein_encoder,
            lr_lambda_for_protein_encoder,
            lr_lambda_for_text_encoder,
            lr_lambda_for_text_encoder,
            lr_lambda_for_embedding,
            lr_lambda_for_decoder,
            lr_lambda_for_contrastive,
        ],
        last_epoch,
    )


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_training_steps: int,
    num_protein_encoder_warmup_steps: int,
    # num_text_encoder_warmup_steps: int,
    num_embedding_warmup_steps: int,
    num_decoder_warmup_steps: int,
    last_epoch=-1,
):
    # FIXME
    pass


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    num_protein_encoder_warmup_steps: Optional[int] = None,
    num_text_encoder_warmup_steps: Optional[int] = None,
    num_embedding_warmup_steps: Optional[int] = None,
    num_decoder_warmup_steps: Optional[int] = None,
    num_contrastive_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
):
    """
    Adapted from OntoProtein.  Unified API to get any scheduler from its name.

    Args:
        name (:obj:`str` or `:obj:`SchedulerType`):
            The name of the scheduler to use.
        optimizer (:obj:`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (:obj:`int`, `optional`):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (:obj:`int`, `optional`):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
    """
    TYPE_TO_SCHEDULER_FUNCTION = {
        SchedulerType.LINEAR: get_linear_schedule_with_warmup,
        SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    }

    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    # All other schedulers require `num_warmup_steps`
    if (
        num_protein_encoder_warmup_steps is None
        or num_decoder_warmup_steps is None
        or num_embedding_warmup_steps is None
    ):
        raise ValueError(
            f"{name} requires `num_warmup_steps`, please provide that argument."
        )

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(
            f"{name} requires `num_training_steps`, please provide that argument."
        )

    return schedule_func(
        optimizer,
        num_protein_encoder_warmup_steps=num_protein_encoder_warmup_steps,
        num_text_encoder_warmup_steps=num_text_encoder_warmup_steps,
        num_embedding_warmup_steps=num_embedding_warmup_steps,
        num_decoder_warmup_steps=num_decoder_warmup_steps,
        num_contrastive_warmup_steps=num_contrastive_warmup_steps,
        num_training_steps=num_training_steps,
    )


#########
# others
#########


def batched_split_long_seq(
    toks: torch.Tensor,
    padding_idx: int,
    eos_idx: int,
    long_protein_strategy: str = "split",
    max_protein_len: int = 1024,
):
    """
    toks: torch.Tensor
        - Batched token input
    padding_idx: int
        - Index of padding token in sequence
    eos_idx: int
        - Index of EOS index
    max_protein_len: Maximum sequence length BEFORE adding CLS and EOS tokens
    """

    cls_idx = toks[0, 0].item()

    if long_protein_strategy == "split":
        # First identify all sequences that need treatment:
        eos_loc = toks == eos_idx
        # Compute EOS locations:
        eos_loc = [
            eos_loc[i, :].nonzero(as_tuple=True)[0] for i in range(eos_loc.shape[0])
        ]
        inds_to_split = [
            i for i in range(len(eos_loc)) if eos_loc[i] > (max_protein_len + 1)
        ]
        # inds_to_split = torch.tensor(eos_over).nonzero(as_tuple=True)[0]
        # inds_to_split = (eos_loc > (max_protein_len + 1)).nonzero(as_tuple=True)[0]
        to_add_list = []
        batch_keys = list(range(toks.shape[0]))
        for i in inds_to_split:
            # Get overage amount:
            overage = (toks[i, :] == eos_idx).nonzero(as_tuple=True)[0][0].item()

            # Get number of additional seq's you'll have to make:
            num_add_splits = overage // (max_protein_len + 1)

            for j in range(num_add_splits):
                # ***Must adjust for cls and eos tokens***
                bot = (
                    j + 1
                ) * max_protein_len + 1  # IGNORE first one (already in place)
                new_empty = (
                    torch.ones(1, toks.shape[1], dtype=int) * padding_idx
                )  # All fill with padding idx
                new_empty = new_empty.to(toks.device)

                # Copy over from prev. iteration
                new_tmp = toks[i, bot:].clone().unsqueeze(0)
                # print('tok, {}, {}, {}'.format(i, j, toks[i,(bot-5):(bot+5)]))
                # rem_shape = toks.shape[1] - new_tmp.shape[1]
                new_empty[0, 1 : (new_tmp.shape[1] + 1)] = new_tmp
                new_empty[0, 0] = cls_idx
                if j < (num_add_splits - 1):
                    new_empty[0, max_protein_len + 1] = (
                        eos_idx  # Else, the EOS index should already be there
                    )
                    new_empty[0, (max_protein_len + 2) :] = 1

                to_add_list.append(new_empty)
                # print('new_emp, {}, {}, {} (front)'.format(i, j, new_empty[0,:7]))
                # print('new_emp, {}, {}, {} (end)'.format(i, j, new_empty[0,(max_protein_len-3):(max_protein_len+5)]))
                # print('new_emp full', new_empty)
                # print('')
                batch_keys.append(i)  # Ind in original toks

                cur_ind = toks.shape[1] + len(to_add_list)

            toks[i, (max_protein_len + 2) :] = padding_idx
            toks[i, (max_protein_len + 1)] = eos_idx

        new_toks = torch.cat([toks] + to_add_list, dim=0)
        new_toks = new_toks[:, : (max_protein_len + 2)]  # Cut down size
        batch_keys = torch.tensor(batch_keys, dtype=int)

    elif long_protein_strategy == "truncate":
        # Simple truncation:
        if toks.shape[1] > (max_protein_len + 2):
            new_toks = toks[
                :, : (max_protein_len + 2)
            ]  # Truncate to appropriate length
            no_pad = (
                new_toks[:, -1] != padding_idx
            )  # Get all samples that DO NOT have padding at the end of their seqs
            new_toks[:, no_pad] = eos_idx  # Replace ending with EOS token if needed
        else:
            new_toks = toks
        batch_keys = None
        eos_loc = None

    # eos_loc = [(new_toks[i,:] == eos_idx).nonzero(as_tuple=True)[0] for i in range(new_toks.shape[0])]

    # for i in range(new_toks.shape[0]):
    #     print(f'new_tok {i}, ', new_toks[i,:])
    # print(new_toks.shape)

    return new_toks, batch_keys, eos_loc


def reverse_batched_split(protein_embeds, batch_keys, eos_locs: list):
    """
    We know that protein_embeds have CLS tokens at each starting spot, EOS at each ending spot
    """
    max_ind = batch_keys.max().item()
    full_protein_embeds = []
    # max_size = 0
    for i in range(max_ind + 1):
        iship = batch_keys == i
        if iship.sum() == 0:  # Allow for breaks in continuously increasing integers
            continue
        iship_inds = iship.nonzero(as_tuple=True)[0].sort()[
            0
        ]  # SORT TO REMAIN CONSISTENT WITH batched_split_long_seq PROCESS
        # Reshape (#,S,d) -> (1,S + #,d), an essential flattening along dimension 1
        common_prot = protein_embeds[iship_inds, :, :]
        eos_inprot = torch.ones(common_prot.shape[0], common_prot.shape[1], dtype=bool)
        eos_inprot[:-1, -1] = (
            False  # All but last index in sub-batch (contains actual EOS token)
        )
        cls_inprot = torch.ones(common_prot.shape[0], common_prot.shape[1], dtype=bool)
        cls_inprot[1:, 0] = (
            False  # All but first sequence in sub-batch (contains actual CLS token)
        )

        common_prot = common_prot.reshape(1, -1, protein_embeds.shape[-1]).squeeze(0)
        eos_inprot = eos_inprot.flatten()
        cls_inprot = cls_inprot.flatten()

        # Trim common_prot by eos_inprot and cls_inprot
        common_prot = common_prot[(eos_inprot & cls_inprot), :]

        # Remove CLS and EOS for middles
        full_protein_embeds.append(common_prot)
        # max_size = common_prot.shape[0] if common_prot.shape[0] > max_size else max_size

    # Pad to max size:
    max_size = max(eos_locs) + 1
    for i in range(len(full_protein_embeds)):
        diff_size = max_size - full_protein_embeds[i].shape[0]
        if diff_size > 0:
            tocat = torch.zeros(diff_size, protein_embeds.shape[-1]).to(
                protein_embeds.device
            )
            full_protein_embeds[i] = torch.cat([full_protein_embeds[i], tocat], dim=0)
        elif diff_size < 0:
            full_protein_embeds[i] = full_protein_embeds[i][:max_size, :]

    new_prot_embeds = torch.stack(full_protein_embeds)

    return new_prot_embeds


def get_relation_types(data_args):

    df = pd.read_csv(
        data_args.relation_file, names=["relation_idx", "relation_type", "symmetric"]
    )
    num_relations = df.shape[0]
    symmetries = torch.from_numpy(df["symmetric"].to_numpy()).bool()

    return num_relations, symmetries


def convert_all_to_device(d, device):
    # DFS a dictionary of potentially multiple types, convert everything to device:
    # print(' dtype', type(d))

    if isinstance(d, torch.Tensor):
        d = d.to(device)
        return
    elif d is None:
        return
    elif isinstance(d, dict):
        for k in d.keys():
            if d[k] is None:
                continue
            if isinstance(d[k], torch.Tensor):
                d[k] = d[k].to(device)
            else:
                convert_all_to_device(d[k], device)
    elif isinstance(d, list):
        return
    else:
        for i in range(len(d)):
            if isinstance(d[i], np.ndarray):
                d[i] = torch.from_numpy(d[i]).to(device)
            elif d[i] is None:
                continue
            elif isinstance(d[i], torch.Tensor):
                d[i] = d[i].to(device)
            else:
                convert_all_to_device(d[i], device)


def concat_tensor_dict(Ld, dim=0):
    """
    Util fn to support concatenating an arbitrary dictionary,
        as long as it's depth 1 and contains all tensor values.

    Assumes all dicts have same keys

    Args:
        Ld: list of dict of tensors
    """
    if len(Ld) == 1:
        return Ld[0]

    a = Ld[0].keys()

    concat_dict = {k: Ld[0][k] for k in a}

    for i in range(1, len(Ld)):
        for k in a:
            concat_dict[k] = torch.cat([concat_dict[k], Ld[i][k]], dim=dim)

    return concat_dict


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    From OntoProtein:
    Recursively unwraps a model from potential containers (as used in distributed training).
    Args:
        model (:obj:`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


def truncate_descriptions(descriptions, cutoff=1000):
    """
    Truncates descriptions in order to fit into instructions
        - For now, this truncation is naive - just cut off the last words
    TODO: Implement using the tokenizer, this method does a naive approximation
    """
    return [d[:cutoff] for d in descriptions]


def decompose_dataset_name(name):
    # f"{self.aaseq_type}_{self.text_type}_{relation}"
    all_splits = name.split("_")
    aaseq_type = all_splits[0]
    text_type = all_splits[1]
    relation = "_".join(all_splits[2:])
    # aaseq_type, text_type, relation = name.split('_')
    return aaseq_type, text_type, relation


import torch.distributed as dist


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
