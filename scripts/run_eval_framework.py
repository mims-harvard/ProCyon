from transformers.hf_argparser import HfArgumentParser

from procyon.evaluate.framework.args import EvalArgs
from procyon.evaluate.framework.core import run_evaluation

from procyon.training.training_args_IT import (
    DataArgs,
    ModelArgs,
    postprocess_args,
)

if __name__ == "__main__":
    parser = HfArgumentParser((EvalArgs, DataArgs, ModelArgs))
    eval_args, data_args, model_args = parser.parse_args_into_dataclasses()

    if eval_args.from_yaml is not None:
        eval_args, data_args, model_args = parser.parse_yaml_file(eval_args.from_yaml)
    _, data_args, model_args = postprocess_args(None, data_args, model_args)

    _ = run_evaluation(eval_args, data_args, model_args)
