import os
from pathlib import Path

from huggingface_hub import login as hf_login
from loguru import logger
import pandas as pd
from typing import Optional, Tuple
import torch

from procyon.data.inference_utils import (
    create_input_retrieval,
    get_proteins_from_embedding,
)
from procyon.evaluate.framework.utils import move_inputs_to_device
from procyon.model.model_unified import UnifiedProCyon
from procyon.training.train_utils import DataArgs

CKPT_NAME = os.path.expanduser(os.getenv("CHECKPOINT_PATH"))


def startup_retrieval(
        inference_bool: bool = True,
) -> Tuple[
    UnifiedProCyon | None,
    torch.device | None,
    DataArgs | None,
    torch.Tensor,
]:
    """
    This function performs startup functions to initiate protein retrieval:
    Logs into the huggingface hub and loads the pre-trained ProCyon model.
    Args:
        inference_bool (bool): OPTIONAL; choose this if you do not intend to do inference or load the model.
            Loading the model is time-consuming, so consider using this to test that the CLI works.
    Returns:
        model (UnifiedProCyon): The pre-trained ProCyon model
        device (torch.device): The compute device (GPU or CPU) on which the model is loaded
        data_args (DataArgs): The data arguments defined by the pre-trained model
        all_protein_embeddings (torch.Tensor): The pre-calculated protein target embeddings
    """
    logger.info("Now running startup functions for protein retrieval")

    logger.info("Now logging into huggingface hub")
    hf_login(token=os.getenv("HF_TOKEN"))
    logger.info("Done logging into huggingface hub")

    if inference_bool:
        logger.info("Inference is enabled.")

        # load the pre-trained ProCyon model
        model, device, data_args = load_model_onto_device()
    else:
        logger.info("Inference is disabled.")
        # loading the model requires time and memory, so we skip it if we are only testing the CLI
        model = None
        device = None
        data_args = None

    # Load the pre-calculated protein target embeddings
    logger.info("Now loading protein target embeddings")
    all_protein_embeddings, all_protein_ids = torch.load(
        os.path.join(CKPT_NAME, "protein_target_embeddings.pkl")
    )
    all_protein_embeddings = all_protein_embeddings.float()
    logger.info(
        f"shape of precalculated embeddings matrix: {all_protein_embeddings.shape}"
    )
    logger.info("Done loading protein target embeddings")
    logger.info("Done running startup functions for protein retrieval")

    return model, device, data_args, all_protein_embeddings


def load_model_onto_device() -> Tuple[UnifiedProCyon, torch.device, DataArgs]:
    """
    Load the pre-trained ProCyon model and move it to the compute device.
    Returns:
        model (UnifiedProCyon): The pre-trained ProCyon model
        device (torch.device): The compute device (GPU or CPU) on which the model is loaded
        data_args (DataArgs): The data arguments defined by the pre-trained model
    """
    # Load the pre-trained ProCyon model
    logger.info("Now loading pretrained model")
    # Replace with the path where you downloaded a pre-trained ProCyon model (e.g. ProCyon-Full)
    data_args = torch.load(os.path.join(CKPT_NAME, "data_args.pt"))
    model, _ = UnifiedProCyon.from_pretrained(checkpoint_dir=CKPT_NAME)
    logger.info("Done loading pretrained model")

    logger.info("Now quantizing the model to a smaller precision")
    model.bfloat16()  # Quantize the model to a smaller precision
    logger.info("Done quantizing the model to a smaller precision")

    logger.info("Now setting the model to evaluation mode")
    model.eval()
    logger.info("Done setting the model to evaluation mode")

    logger.info("Now applying pretrained model to device")
    logger.info(f"Total memory allocated by PyTorch: {torch.cuda.memory_allocated()}")
    # identify available devices on the machine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Total memory allocated by PyTorch: {torch.cuda.memory_allocated()}")

    logger.info("Done loading model and applying it to compute device")

    return model, device, data_args


def do_retrieval(
        model: UnifiedProCyon,
        data_args: DataArgs,
        device: torch.device,
        instruction_source_dataset: str,
        all_protein_embeddings: torch.Tensor,
        inference_bool: bool = True,
        task_desc_infile: Path = None,
        disease_desc_infile: Path = None,
        task_desc: str = None,
        disease_desc: str = None,
) -> Optional[pd.DataFrame]:
    """
    This function performs protein retrieval for a given disease using the pre-trained ProCyon model.
    Args:
        model (UnifiedProCyon): The pre-trained ProCyon model
        data_args (DataArgs): The data arguments defined by the pre-trained model
        device (torch.device): The compute device (GPU or CPU) on which the model is loaded
        instruction_source_dataset (str): Dataset source for instructions - either "disgenet" or "omim"
        all_protein_embeddings (torch.Tensor): The pre-calculated protein target embeddings
        inference_bool (bool): OPTIONAL; choose this if you do not intend to do inference
        task_desc_infile (Path): The path to the file containing the task description.
        disease_desc_infile (Path): The path to the file containing the disease description.
        task_desc (str): The task description.
        disease_desc (str): The disease description.
    Returns:
        df_dep (pd.DataFrame): The DataFrame containing the top protein retrieval results
    """
    logger.info("Now performing protein retrieval")

    if instruction_source_dataset not in ["disgenet", "omim"]:
        raise ValueError(
            'instruction_source_dataset must be either "disgenet" or "omim"'
        )

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
    logger.info("Done entering task description and prompt")

    if inference_bool:
        logger.info("Now performing protein retrieval")

        # Create input for retrieval
        input_simple = create_input_retrieval(
            input_description=disease_desc,
            data_args=data_args,
            task_definition=task_desc,
            instruction_source_dataset=instruction_source_dataset,
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

        logger.info("Done performing protein retrieval")

        return df_dep
