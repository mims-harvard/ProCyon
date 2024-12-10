import os
from enum import Enum
from dataclasses import dataclass, field, fields
from transformers.training_args import TrainingArguments
from typing import Tuple
from dataclasses import asdict

from procyon.data.data_utils import DATA_DIR

@dataclass
class EvalArgs:
    '''
    Class that represents all arguments used for benchmark evaluation
    '''
    model_dir: str = field(
        default = None,
        metadata = {
            'help': "Checkpoint directory of model to evaluate"
        }
    )

    # ------------------------------- Processing arguments --------------------------------------------
    batch_size: int = field(
        default = None,
        metadata = {
            "help": "Maximum size of batch to feed into model during evaluation - based on memory constraints"
        }
    )

    max_num_positive_qa_samples: int = field(
        default = None,
        metadata = {
            "help": "Maximum number of (positive) samples on which to evaluate QA"
        }
    )

    max_num_captioning_samples: int = field(
        default = None,
        metadata = {
            "help": "Maximum number of samples on which to evaluate captioning"
        }
    )

    # ------------------------------- Evaluation task options --------------------------------------------
    evaluate_qa: bool = field(
        default = False,
        metadata = {
            "help": "Evaluate model on QA task"
        }
    )

    evaluate_retrieval: bool = field(
        default = False,
        metadata = {
            "help": "Evaluate model on retrieval task"
        }
    )

    evaluate_caption: bool = field(
        default = False,
        metadata = {
            "help": "Evaluate model on captioning task"
        }
    )

    # ------------------------------- Dataset options --------------------------------------------

    shot_level: str = field(
        default = False,
        metadata = {
            "help": "Shot level for datasets",
            "choices": ["pt_ft", "five_shot", "zero_shot"]
        }
    )

    aaseq_type: str = field(
        default = 'protein',
        metadata = {
            "help": "Type of amino acid sequence in the dataset",
            "choices": ["protein", "domain"],
        }
    )

    text_type: str = field(
        default = "go",
        metadata = {
            "help": "Type of text in the dataset desired",
        }
    )

    relation_type: str = field(
        default = "all",
        metadata = {
            "help": "Type of relation to consider for the evaluation",
            # TODO: Add choices
        }
    )

    text_variant_type: str = field(
        default='standard',
        metadata={
            "help": "The type of description to use for text.",
        }
    )


    # ------------------------------- Metric parameters --------------------------------------------
    eval_k: int = field(
        default = 25,
        metadata = {
            "help": "k number for retrieval evaluation calculation"
        }
    )

    num_neg_samples_qa: int = field(
        default = 1,
        metadata = {
            "help": "Number of negative examples for QA evaluation"
        }
    )

    # ------------------------------- GO-specific arguments --------------------------------------------
    go_def_col: str = field(
        #default="description_combined",
        default='standard', # TODO: Change this default later when old models are deprecated
        metadata={
            "help": "The name of the text to use for GO descriptions during training.",
            "choices": ["description_combined", "standard", "name_def", "def_only"]
        }
    )
    go_split_method: str = field(
        default="sample_aware_ontology_go_centric",
        metadata={
            "help": "The method to split GO terms into CL train, CL val, eval pt-ft, eval few-shot, and eval zero-shot sets.",
            "choices": ["sample_random_random_go_centric", "sample_random_random_pair_centric", "sample_aware_random_go_centric", "sample_random_time_aware_go_centric", "sample_aware_time_aware_go_centric", "sample_random_ontology_aware_go_centric", "sample_aware_ontology_aware_go_centric"]
        }
    )
