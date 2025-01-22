import os
from typing import Dict, Optional, Tuple, Union
from pathlib import Path

import argparse
from loguru import logger
import pandas as pd

from procyon.inference.retrieval_utils import startup_retrieval, do_retrieval

CKPT_NAME = os.path.expanduser(os.getenv("CHECKPOINT_PATH"))


def single_retrieval(
    task_desc_infile: Path,
    disease_desc_infile: Path,
    instruction_source_dataset: str,
    inference_bool: bool = True,
) -> Union[pd.DataFrame, None]:
    """
    This function uses the pre-trained ProCyon model to perform one protein retrieval run
    for a given disease using DisGeNET data.
    Args:
        task_desc_infile (Path): The path to the file containing the task description.
        disease_desc_infile (Path): The path to the file containing the disease description.
        instruction_source_dataset (str): Dataset source for instructions - either "disgenet" or "omim"
        inference_bool (bool): OPTIONAL; choose this if you do not intend to do inference
    Returns:
        Optional[pd.DataFrame]: DataFrame with results if inference_bool is True, None otherwise
    """
    model, device, data_args, all_protein_embeddings = startup_retrieval(inference_bool)

    results_df = do_retrieval(
        model=model,
        data_args=data_args,
        device=device,
        inference_bool=inference_bool,
        task_desc_infile=task_desc_infile,
        disease_desc_infile=disease_desc_infile,
        instruction_source_dataset=instruction_source_dataset,
        all_protein_embeddings=all_protein_embeddings,
    )
    if results_df is not None:
        logger.info(f"top results: {results_df.head(10).to_dict(orient='records')}")

    logger.info("DONE WITH ALL WORK")
    return results_df


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
    parser.add_argument(
        "--instruction_source_dataset",
        type=str,
        choices=["disgenet", "omim"],
        help="Dataset source for instructions - either 'disgenet' or 'omim'",
    )
    args = parser.parse_args()

    single_retrieval(
        args.task_desc_infile,
        args.disease_desc_infile,
        args.instruction_source_dataset,
        args.inference_bool,
    )
