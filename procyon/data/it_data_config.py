import re
import yaml

from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import pandas as pd

from esm.data import Alphabet
from transformers import AutoTokenizer


from procyon.data.dataset import (
    AASeqTextUnifiedDataset,
    AASeqDataset,
)
from procyon.data.it_collator import (
    RetrievalCollator,
    QACollator,
    CaptionCollator,
)
from procyon.data.constants import *
from procyon.evaluate.framework.constants import SPLIT_MAPS
from procyon.training.training_args_IT import (
    DataArgs,
    ModelArgs,
    postprocess_args,
)

def get_IT_dataset(
    aaseq_type: str,
    text_type: str,
    relation_type: str,
    task_type: str,
    splits_to_use: List[str],
    data_args: DataArgs,
    text_split_method: Optional[str] = None,
    deduplicate_dataset: Optional[str] = None,
) -> Union[AASeqDataset, AASeqTextUnifiedDataset]:
    if task_type not in ["caption", "retrieval", "qa"]:
        raise ValueError(f"Unexpected task type: {task_type}")

    if task_type == "qa":
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

    elif task_type == "retrieval":
        # TODO: Need to make conditional on the type of sampling - e.g. protein-only, protein-go-only, etc.
        neg_per_aaseq = data_args.num_neg_samples_retrieval // 2
        neg_per_text = data_args.num_neg_samples_retrieval - neg_per_aaseq
        negative_sampling_strategy = data_args.negative_sampling_strategy_retrieval

    elif task_type == "caption":
        if not data_args.use_caption:
            return None, None
        neg_per_aaseq = (
            1  # NOTE: This has no effect for captioning, so left as a constant
        )
        neg_per_text = 1
        negative_sampling_strategy = data_args.negative_sampling_strategy_retrieval

    if any(
        map(lambda x: re.search("with_(\d+)_negatives", x) is not None, splits_to_use)
    ):
        negative_sampling_strategy = "preset"

    if text_split_method is None:
        if ":" in text_type:
            text_split_method = (
                data_args.go_split_method
                if text_type == "go"
                else f'random_{text_type.split(":")[0]}_centric'
            )
        else:
            text_split_method = (
                data_args.go_split_method
                if text_type == "go"
                else f"random_{text_type}_centric"
            )

    if data_args.aaseq_subset_tsv_path is not None:
        aaseq_subset = pd.read_table(data_args.aaseq_subset_tsv_path).seq_id.to_list()
    else:
        aaseq_subset = None

    # Abuse of 'text_type' to represent second amino acid sequence in a
    # AA seq <-> AA seq interaction
    if aaseq_type == text_type:
        dataset = AASeqDataset(
            data_dir=data_args.data_dir,
            aaseq_type=aaseq_type,
            relation_type=relation_type,
            splits_to_use=splits_to_use,
            negative_sampling_strategy=negative_sampling_strategy,
            aaseq_sims_type=data_args.protein_sims_type,
            num_neg_samples_per_aaseq=neg_per_aaseq,
            use_perplexity_filtered_set=data_args.use_perplexity_filtered_set,
            store_reverse_edges=data_args.ppi_store_reverse_edges,
            swap_prob=data_args.ppi_edge_swap_prob,
        )
    else:
        dataset = AASeqTextUnifiedDataset(
            data_dir=data_args.data_dir,
            aaseq_type=aaseq_type,
            text_type=text_type,
            relation_type=relation_type,
            splits_to_use=splits_to_use,
            text_split_method=text_split_method,
            negative_sampling_strategy=negative_sampling_strategy,
            aaseq_sims_type=data_args.protein_sims_type,
            text_sims_type=None,
            num_neg_samples_aaseq_text_per_aaseq=neg_per_aaseq,
            num_neg_samples_aaseq_text_per_text=neg_per_text,
            use_only_aaseq_text_aaseqs=data_args.use_only_goa_proteins,
            use_only_aaseq_text_texts=data_args.use_only_goa_gos,
            use_perplexity_filtered_set=data_args.use_perplexity_filtered_set,
            deduplicate_dataset=deduplicate_dataset,
            aaseq_subset=aaseq_subset,
        )

    return dataset


def get_IT_collator(
    aaseq_type: str,
    text_type: str,
    relation_type: str,
    task: str,
    data_args: DataArgs,
    model_args: ModelArgs,
    protein_tokenizer: Alphabet,
    text_tokenizer: AutoTokenizer,
    evaluation: bool = False,
    text_split_method=None,
) -> Union[CaptionCollator, RetrievalCollator, QACollator]:
    assert not model_args.is_protein_tokenized, "Not yet supported"
    assert not model_args.is_go_tokenized, "Not yet supported"

    col_subset = None
    if (task == "qa") and (data_args.qa_subset_version is not None):
        col_subset = QA_SUBSETS[data_args.qa_subset_version]
    elif (task == "retrieval") and (data_args.retrieval_subset_version is not None):
        col_subset = RETRIEVAL_SUBSETS[data_args.retrieval_subset_version]
    elif (task == "caption") and (data_args.caption_subset_version is not None):
        col_subset = CAPTION_SUBSETS[data_args.caption_subset_version]

    if text_split_method is None:
        text_split_method = (
            data_args.go_split_method
            if text_type == "go"
            else f"random_{text_type}_centric"
        )

    opt_use_entity_rephrasings = data_args.use_entity_rephrasings
    if (task == "caption") and opt_use_entity_rephrasings:
        opt_use_entity_rephrasings = data_args.rephrase_caption_entities

    shared_kwargs = {
        "data_dir": data_args.data_dir,
        "aaseq_type": aaseq_type,
        "text_type": text_type,
        "relation_type": relation_type,
        "text_variant_type": data_args.text_variant_type,
        "aaseq_sims_type": data_args.protein_sims_type,
        "is_aaseq_tokenized": model_args.is_protein_tokenized,
        "is_text_tokenized": model_args.is_go_tokenized,
        "use_text_embeddings": model_args.use_text_embeddings,
        "use_aaseq_embeddings": model_args.use_aaseq_embeddings,
        "aaseq_tokenizer": protein_tokenizer,
        "text_tokenizer": text_tokenizer,
        "max_aaseq_len": model_args.max_protein_len,
        "max_text_len": model_args.max_text_len,
        "num_examples": data_args.num_instruction_examples,
        "sample_num_examples": data_args.sample_num_instruction_examples,
        "use_entity_compositions": data_args.use_entity_compositions,
        "sample_entity_compositions": data_args.sample_entity_compositions,  # "constant" if static_caption_subset_condition else
        "evaluation": evaluation,
        "use_instructions": data_args.use_instructions,
        "use_drug_embeddings": model_args.use_drug_embeddings,
        "column_subset": col_subset,
        "insert_disease_function_context": data_args.insert_disease_function_context,
        "disease_function_context_dropout": data_args.disease_function_context_dropout,
        "text_split_method": text_split_method,
        "insert_go_ontology_context": data_args.insert_go_ontology_context,
        "go_ontology_rag_level_upper_limit": data_args.go_ontology_rag_level_upper_limit,
        "go_ontology_rag_num_context": data_args.go_ontology_rag_num_context,
        "go_ontology_rag_sample_num_context": data_args.go_ontology_rag_sample_num_context,
        "insert_go_ontology_level": data_args.insert_go_ontology_level,
        "use_go_ontology_level_groups": data_args.use_go_ontology_level_groups,
        "insert_reactome_ontology_context": data_args.insert_reactome_ontology_context,
        "reactome_ontology_rag_level_upper_limit": data_args.reactome_ontology_rag_level_upper_limit,
        "reactome_ontology_rag_num_context": data_args.reactome_ontology_rag_num_context,
        "reactome_ontology_rag_sample_num_context": data_args.reactome_ontology_rag_sample_num_context,
        "insert_reactome_ontology_level": data_args.insert_reactome_ontology_level,
        "use_reactome_ontology_level_groups": data_args.use_reactome_ontology_level_groups,
        "use_drug_context_augmentation": data_args.use_drug_context_augmentation,
        "use_entity_rephrasings": opt_use_entity_rephrasings,  # Defined above, derived from data_args
        "use_task_def_rephrasings": data_args.use_task_def_rephrasings,
        "rephrasing_sample_prob": data_args.rephrasing_sample_prob,
        "use_personality_prompts_rephrasing": data_args.use_personality_prompts_rephrasing,
        "exclude_levels_in_ontology_captioning": data_args.exclude_levels_in_ontology_captioning,
        "fixed_rephrasing_expertise_level": data_args.fixed_rephrasing_expertise_level,
        "fixed_rephrasing_entity_rephrase_level": data_args.fixed_rephrasing_entity_rephrase_level,
        "fixed_rephrasing_task_def_rephrase_level": data_args.fixed_rephrasing_task_def_rephrase_level,
        "long_protein_strategy": model_args.long_protein_strategy,
    }
    if task == "qa":
        collator = QACollator(**shared_kwargs)

    elif task == "retrieval":
        collator = RetrievalCollator(
            train_retrieval_lm=model_args.train_retrieval_lm, **shared_kwargs
        )
    elif task == "caption":
        collator = CaptionCollator(**shared_kwargs)

    return collator


class ITDatasetConfig(object):
    def __init__(
        self,
        aaseq_type: str,
        text_type: str,
        relations: List[str],
        tasks: List[str],
        splits: List[str],
        dataset_args: Dict = {},
        # eval_args is used to store any extra parameters needed
        # for downstream evaluations using this dataset, and
        # generally should be ignored when loading the dataset.
        eval_args: Dict = {},
        # key_suffix can be used to disambiguate datasets that
        # might otherwise look the same (i.e. same aaseq_type,
        # text_type, relation).
        key_suffix: str = "",
        split_method: str = "random",  # text_split_method in datasets - if not provided, it will be set to default random
    ) -> None:
        self.aaseq_type = aaseq_type
        self.text_type = text_type
        self.relations = relations
        self.tasks = tasks
        self.splits = splits

        dset_name = "{}_{}".format(self.aaseq_type, self.text_type)
        for i in range(len(self.splits)):
            if self.splits[i].startswith("EVAL:"):
                simple_split_name = self.splits[i].split(":")[-1]
                if not dset_name in SPLIT_MAPS:
                    raise ValueError(f"dataset name not in SPLIT_MAPS: {dset_name}")
                if not simple_split_name in SPLIT_MAPS[dset_name]:
                    raise ValueError(
                        f"dataset {dset_name}, split not in SPLIT_MAPS: {simple_split_name}"
                    )
                self.splits[i] = SPLIT_MAPS[dset_name][simple_split_name]

        self.key_suffix = key_suffix
        self.dataset_args = dataset_args
        self.eval_args = eval_args
        if split_method == "random":
            self.text_split_method = f"random_{text_type}_centric"
        else:
            self.text_split_method = split_method

    def to_dict(self) -> Dict:
        return {
            "aaseq_type": self.aaseq_type,
            "text_type": self.text_type,
            "relations": self.relations,
            "tasks": self.tasks,
            "splits": self.splits,
            "key_suffix": self.key_suffix,
            "dataset_args": self.dataset_args,
            "eval_args": self.eval_args,
            "text_split_method": self.text_split_method,
        }

    def _construct_key(
        self,
        relation: str,
    ) -> str:
        parts = [self.aaseq_type, self.text_type, relation]
        if self.key_suffix != "":
            parts.append(self.key_suffix)
        return "_".join(parts)

    def get_datasets_and_collators(
        self,
        data_args: DataArgs,
        model_args: ModelArgs,
        protein_tokenizer: Alphabet,
        text_tokenizer: AutoTokenizer,
        evaluation: bool = False,
        deduplicate_dataset: bool = False,
    ) -> Dict:
        datasets_and_collators_by_task = defaultdict(dict)
        data_args = replace(data_args, **self.dataset_args)
        _, data_args, model_args = postprocess_args(None, data_args, model_args)
        for task in self.tasks:
            unique_dim = None
            if deduplicate_dataset:
                if task == "retrieval":
                    unique_dim = "text"
                elif task == "caption":
                    unique_dim = "aaseq"

            for relation in self.relations:
                dataset = get_IT_dataset(
                    self.aaseq_type,
                    self.text_type,
                    relation,
                    task,
                    self.splits,
                    data_args,
                    text_split_method=self.text_split_method,
                    deduplicate_dataset=unique_dim,
                )
                collator = get_IT_collator(
                    self.aaseq_type,
                    self.text_type,
                    relation,
                    task,
                    data_args,
                    model_args,
                    protein_tokenizer,
                    text_tokenizer,
                    evaluation=evaluation,
                    text_split_method=self.text_split_method,
                )
                key = self._construct_key(relation)
                datasets_and_collators_by_task[task][key] = (dataset, collator)
        return datasets_and_collators_by_task

    def __repr__(self) -> str:
        return yaml.dump(self.to_dict(), sort_keys=False)


class ITMultiDatasetConfig(object):
    def __init__(
        self,
        train: List[ITDatasetConfig],
        validation: List[ITDatasetConfig],
        testing: List[ITDatasetConfig],
    ) -> None:
        self.train_datasets = train
        self.validation_datasets = validation
        self.testing_datasets = testing

    @classmethod
    def load_from_yaml(cls, path: str):
        with open(path, "r") as fh:
            raw_data = yaml.safe_load(fh)
        if "it_datasets" in raw_data:
            raw_data = raw_data["it_datasets"]

        splits = ["train", "validation", "testing"]
        parsed_datasets = {split: [] for split in splits}
        for split in splits:
            if split not in raw_data:
                continue
            for dataset in raw_data[split]:
                parsed_datasets[split].append(ITDatasetConfig(**dataset))
        return ITMultiDatasetConfig(**parsed_datasets)

    def get_datasets_and_collators(
        self,
        data_args: DataArgs,
        model_args: ModelArgs,
        protein_tokenizer: Alphabet,
        text_tokenizer: AutoTokenizer,
        evaluation: bool = False,
        deduplicate_dataset: bool = False,
    ) -> Tuple[Dict, Dict]:
        """Returns datasets and collators organized by split (train/val/test) and task (retrieval/caption/qa)"""
        organized_datasets = {
            "train": self.train_datasets,
            "validation": self.validation_datasets,
            "testing": self.testing_datasets,
        }
        datasets_by_use = {
            "train": defaultdict(dict),
            "validation": defaultdict(dict),
            "testing": defaultdict(dict),
        }
        collators_by_use = {
            "train": defaultdict(dict),
            "validation": defaultdict(dict),
            "testing": defaultdict(dict),
        }
        for split, datasets in organized_datasets.items():
            for dataset in datasets:
                ret = dataset.get_datasets_and_collators(
                    data_args,
                    model_args,
                    protein_tokenizer,
                    text_tokenizer,
                    evaluation=evaluation,
                    deduplicate_dataset=deduplicate_dataset,
                )
                for task, vals in ret.items():
                    for key, (dataset, collator) in vals.items():
                        if key in datasets_by_use[split][task]:
                            raise ValueError(
                                f"conflicting dataset key found, consider: {key} "
                                "consider using key_suffix to disambiguate"
                            )
                        datasets_by_use[split][task][key] = dataset
                        collators_by_use[split][task][key] = collator
        return datasets_by_use, collators_by_use

    def get_eval_args_by_dataset(self) -> Dict[str, Dict]:
        eval_args_by_datset = {}
        for dataset in self.testing_datasets:
            for relation in dataset.relations:
                key = dataset._construct_key(relation)
                eval_args_by_datset[key] = dataset.eval_args
        return eval_args_by_datset

    def __repr__(self):
        return yaml.dump(
            {
                "train": [x.to_dict() for x in self.train_datasets],
                "validation": [x.to_dict() for x in self.validation_datasets],
                "testing": [x.to_dict() for x in self.testing_datasets],
            },
            sort_keys=False,
        )


def expand_datasets_on_splits(
    datasets: List[ITDatasetConfig],
    keep_union: bool = False,
) -> List[ITDatasetConfig]:
    """Expand a list of splits into multiple datasets, one per split"""
    expanded_sets = []

    for dataset in datasets:
        if keep_union:
            expanded_sets.append(dataset)
        for split in dataset.splits:
            copy = deepcopy(dataset)
            copy.splits = [split]

            if dataset.key_suffix != "":
                copy.key_suffix = f"{split}_{dataset.key_suffix}"
            elif split != "all":
                copy.key_suffix = split
            expanded_sets.append(copy)

    return expanded_sets


def package_collators_for_trainer(collators: Dict) -> Tuple[Dict, Dict, Dict]:
    splits = ["train", "validation", "testing"]
    qa_collators = {}
    retrieval_collators = {}
    caption_collators = {}
    for split in splits:
        qa_collators.update(collators[split]["qa"])
        retrieval_collators.update(collators[split]["retrieval"])
        caption_collators.update(collators[split]["caption"])
    return qa_collators, retrieval_collators, caption_collators
