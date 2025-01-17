import os
from typing import Dict, Optional, Tuple
from pathlib import Path

import argparse
from huggingface_hub import login as hf_login
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

CKPT_NAME = os.path.expanduser(os.getenv("CHECKPOINT_PATH"))


def load_model_onto_device() -> Tuple[UnifiedProCyon, torch.device, DataArgs]:
    # Load the pre-trained ProCyon model
    logger.info("Loading pretrained model")
    # Replace with the path where you downloaded a pre-trained ProCyon model (e.g. ProCyon-Full)
    data_args = torch.load(os.path.join(CKPT_NAME, "data_args.pt"))
    model, _ = UnifiedProCyon.from_pretrained(checkpoint_dir=CKPT_NAME)
    logger.info("Done loading pretrained model")

    logger.info("Quantizing the model to a smaller precision")
    model.bfloat16()  # Quantize the model to a smaller precision
    logger.info("Done quantizing the model to a smaller precision")

    logger.info("Setting the model to evaluation mode")
    model.eval()
    logger.info("Done setting the model to evaluation mode")

    logger.info("Applying pretrained model to device")
    logger.info(f"Total memory allocated by PyTorch: {torch.cuda.memory_allocated()}")
    # identify available devices on the machine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Total memory allocated by PyTorch: {torch.cuda.memory_allocated()}")

    model.bfloat16()  # Quantize the model to a smaller precision
    _ = model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("Done loading model and applying it to compute device")

    return model, device, data_args


def startup_retrieval(
    task_desc_infile: Path, disease_desc_infile: Path, inference_bool: bool = True
):

    if inference_bool:
        logger.info("Inference is enabled.")
    else:
        logger.info("Inference is disabled.")

    logger.info("Logging into huggingface hub")
    hf_login(token=os.getenv("HF_TOKEN"))
    logger.info("Done logging into huggingface hub")

    if inference_bool:
        # load the pre-trained ProCyon model
        model, device, data_args = load_model_onto_device()
    else:
        # loading the model takes much time and memory, so we skip it if we don't need it
        model = None
        device = None
        data_args = None

    return model, device, data_args


def single_retrieval(
    task_desc_infile: Path, disease_desc_infile: Path, inference_bool: bool = True
):
    """
    This function uses the pre-trained ProCyon model to perform protein retrieval
    for a given disease using DisGeNET data.
    """

    model, device, data_args = startup_retrieval(
        task_desc_infile, disease_desc_infile, inference_bool
    )

    do_retrieval(
        model=model,
        data_args=data_args,
        device=device,
        inference_bool=inference_bool,
        task_desc_infile=task_desc_infile,
        disease_desc_infile=disease_desc_infile,
    )

    logger.info("DONE WITH ALL WORK")


def do_retrieval(
    model: UnifiedProCyon,
    data_args: DataArgs,
    device: torch.device,
    inference_bool: bool = True,
    task_desc_infile: Path = None,
    disease_desc_infile: Path = None,
    task_desc: str = None,
    disease_desc: str = None,
) -> Optional[pd.DataFrame]:
    # Load the pre-calculated protein target embeddings
    logger.info("Load protein target embeddings")
    all_protein_embeddings, all_protein_ids = torch.load(
        os.path.join(CKPT_NAME, "protein_target_embeddings.pkl")
    )
    all_protein_embeddings = all_protein_embeddings.float()
    logger.info(
        f"shape of precalculated embeddings matrix: {all_protein_embeddings.shape}"
    )

    #
    logger.info("entering task description and prompt")
    if task_desc_infile is not None:
        if task_desc is not None:
            raise ValueError(
                "Only one of task_desc_infile and task_desc can be provided."
            )
        # read the task description from a file
        with open(task_desc_infile, "r") as f:
            task_desc = f.read()
    elif task_desc is None:
        raise ValueError("Either task_desc_infile or task_desc must be provided.")

    if disease_desc_infile is not None:
        if disease_desc is not None:
            raise ValueError(
                "Only one of disease_desc_infile and disease_desc can be provided."
            )
        # read the disease description from a file
        with open(disease_desc_infile, "r") as f:
            disease_desc = f.read()
    elif disease_desc is None:
        raise ValueError("Either disease_desc_infile or disease_desc must be provided.")

    task_desc = task_desc.replace("\n", " ")
    disease_desc = disease_desc.replace("\n", " ")
    disease_prompt = "Disease: {}".format(disease_desc)

    logger.info("Done entering task description and prompt")

    if inference_bool:
        logger.info("Now performing protein retrieval for example 1")

        # Create input for retrieval
        input_simple = create_input_retrieval(
            input_description=disease_prompt,
            data_args=data_args,
            task_definition=task_desc,
            instruction_source_dataset="disgenet",  # Changed from "drugbank" to "disgenet"
            instruction_source_relation="all",
            aaseq_type="protein",
            icl_example_number=1,  # 0, 1, 2
        )

        input_simple = move_inputs_to_device(input_simple, device=device)
        with torch.no_grad():
            model_out = model(
                inputs=input_simple,
                retrieval=True,
                aaseq_type="protein",
            )
        # The script can run up to here without a GPU, but the following line requires a GPU
        df_dep = get_proteins_from_embedding(
            all_protein_embeddings, model_out, top_k=None
        )
        logger.info(f"top results: {df_dep.head(10).to_dict(orient='records')}")

        logger.info("Done performaing protein retrieval for example 1")

        return df_dep


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
