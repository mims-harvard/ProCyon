import os
from typing import Dict, Optional, Tuple, Union
from pathlib import Path

import argparse
from loguru import logger
import pandas as pd
import torch

from procyon.data.data_utils import DATA_DIR
from procyon.data.inference_utils import (
    create_input_retrieval,
    get_proteins_from_embedding,
)
from procyon.evaluate.framework.utils import move_inputs_to_device
from procyon.model.model_unified import UnifiedProCyon
from procyon.training.train_utils import DataArgs

from procyon.inference.retrieval_utils import startup_retrieval, do_retrieval

CKPT_NAME = os.path.expanduser(os.getenv("CHECKPOINT_PATH"))


def single_retrieval(
    task_desc_infile: Path, disease_desc_infile: Path, inference_bool: bool = True
) -> Union[pd.DataFrame, None]:
    """
    This function uses the pre-trained ProCyon model to perform one protein retrieval run
    for a given disease using DisGeNET data.
    Args:
        task_desc_infile (Path): The path to the file containing the task description.
        disease_desc_infile (Path): The path to the file containing the disease description.
        inference_bool (bool): OPTIONAL; choose this if you do not intend to do inference
    Returns:
        None
    """

    model, device, data_args = startup_retrieval(inference_bool)

    results_df = do_retrieval(
        model=model,
        data_args=data_args,
        device=device,
        inference_bool=inference_bool,
        task_desc_infile=task_desc_infile,
        disease_desc_infile=disease_desc_infile,
    )
    if results_df is not None:
        logger.info(f"top results: {results_df.head(10).to_dict(orient='records')}")

    logger.info("DONE WITH ALL WORK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_desc_infile",
        type=str,
        help="Description of the task.",
    )
    parser.add_argument(
        "--disease_desc_infile",
        type=str,
        help="Description of the task.",
    )
    parser.add_argument(
        "--inference_bool",
        action="store_false",
        help="OPTIONAL; choose this if you do not intend to do inference or load the model",
        default=True,
    )
    args = parser.parse_args()

    single_retrieval(
        args.task_desc_infile, args.disease_desc_infile, args.inference_bool
    )
