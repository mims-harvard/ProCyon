from dataclasses import dataclass, field
from typing import List


@dataclass
class EvalArgs:
    """Arguments for model evaluation and benchmarking"""

    from_yaml: str = field(
        default=None,
        metadata={
            "help": "Path to yaml file specifying all arguments for EvalArgs, "
            "DataArgs, and ModelArgs. If specified, overrides all other "
            "command line arguments.",
        },
    )
    output_dir: str = field(
        default=None,
        metadata={
            "help": "Path to output directory for writing metrics to TSV. Defaults "
            "current directory if not specified.",
        },
    )
    models_config_yml: str = field(
        default=None,
        metadata={
            "help": "Path to yaml file specifying models and associated arguments",
        },
    )
    batch_size: int = field(
        default=16,
        metadata={
            "help": "Maximum size of batch to feed into model during evaluation - based on memory constraints",
        },
    )
    num_workers: int = field(
        default=0,
        metadata={
            "help": "Number of workers for data loading.",
        },
    )
    retrieval_eval_all_aaseqs: bool = field(
        default=True,
        metadata={
            "help": "Compute retrieval metrics over all AA seqs of the given domain (i.e. proteins or domains).",
        },
    )
    retrieval_top_k_vals: List[int] = field(
        default=5,
        metadata={
            "help": "Values of k for computing top k metrics for retrieval.",
            "nargs": "+",
        },
    )
    retrieval_use_cached_target_embeddings: int = field(
        default=True,
        metadata={
            "help": "Use (or create) cached embeddings of targets for retrieval.",
        },
    )
    retrieval_balanced_metrics_num_samples: int = field(
        default=None,
        metadata={
            "help": "Rather than calculating retrieval metrics across the entire (likely imbalanced) "
            "dataset, instead average metrics over a series of samplings of negatives. If None, "
            "then metrics are calculated over the entire dataset. If set to an int N, then for "
            "each query, we perform N rounds of sampling a number of negatives equal to some "
            "multiple of the number of positives, and average metrics across these rounds. "
            "Number of sampled negatives per positive is controlled by the "
            "`retrieval_balanced_metrics_neg_per_pos` argument."
        },
    )
    retrieval_balanced_metrics_neg_per_pos: int = field(
        default=1,
        metadata={
            "help": "Controls number of sampled negatives per positive when calculating class-balanced "
            "retrieval metrics. See description of `retrieval_balanced_metrics_num_samples` "
            "for more details. Only used if `retrieval_balanced_metrics_num_samples` is set."
        },
    )
    retrieval_auroc_auprc_per_query: bool = field(
        default=True,
        metadata={
            "help": "If true, calculate AUROC and AUPRC for each query individually and then average "
            "the values. Otherwise calculates single AUROC/AUPRC values over all query-target pairs."
        },
    )
    model_args_from_checkpoint: str = field(
        default="",
        metadata={
            "help": "Load ModelArgs from a ProCyon checkpoint. Overrides YAML or CLI, "
            "this can be important for data loading to match what the saved "
            "model expects.",
        },
    )
    data_args_from_checkpoint: str = field(
        default="",
        metadata={
            "help": "Load DataArgs from a ProCyon checkpoint. Overrides YAML or CLI, "
            "this can be important for data loading to match what the saved "
            "data expects.",
        },
    )
    override_model_data_args_yml: str = field(
        default=None,
        metadata={
            "help": "Path to yaml file specifying ModelArgs and DataArgs to override."
            "Arguments specified here will take precedence over args parsed "
            "from a model checkpoint via `model_args_from_checkpoint` and "
            "`data_args_from_checkpoint`.",
        },
    )
    filter_training_pairs: bool = field(
        default=True,
        metadata={
            "help": "If True, filters out training data from each eval dataset's respective training set"
        },
    )
    separate_splits: bool = field(
        default=True,
        metadata={
            "help": "Treat multiple splits within a single data config entry as specifying separate "
            "instances of the same dataset with different splits, as opposed to a single dataset "
            "that merges all splits. This is largely a quality-of-life option to reduce redundancy "
            "in dataset configs.",
        },
    )
    keep_splits_union: bool = field(
        default=True,
        metadata={
            "help": "Only matters if `separate_splits == True`. Also evaluate on the union of the listed "
            "splits in addition to the individual listed splits.",
        },
    )
    use_cached_results: bool = field(
        default=True,
        metadata={
            "help": "Used cached predictions to regenerate evaluation metrics. Helpful "
            "if you just want to tweak metric settings or test new changes",
        },
    )
    seed: int = field(
        default=42, metadata={"help": "Random seed used for subsampling."}
    )
    qa_num_samples: int = field(
        default=None,
        metadata={
            "help": "Number of samples (linearly scanning dataset) to run for QA evaluation"
        },
    )
    caption_max_len: int = field(
        default=64,
        metadata={"help": "Maximum length (in tokens) for caption generation."},
    )
