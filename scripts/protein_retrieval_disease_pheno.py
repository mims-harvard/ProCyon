import os
import math
from typing import Dict, Optional, Tuple
from pathlib import Path

import argparse
from huggingface_hub import login as hf_login
from loguru import logger
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from scipy import stats
from tqdm import trange

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logger.info("Done loading model and applying it to compute device")

    return model, device, data_args


def run_retrieval(task_desc_infile: Path,
                 disease_desc_infile: Path,
                 inference_bool: bool = True):
    """
    This function uses the pre-trained ProCyon model to perform protein retrieval
    for a given disease using DisGeNET data.
    """

    if task_desc_infile is None:
        raise ValueError("task_desc_infile must be provided.")
    if disease_desc_infile is None:
        raise ValueError("disease_desc_infile must be provided.")
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

    # TODO: generate a script that generically handles retrieval
    # Load the pre-calculated protein target embeddings
    logger.info("Load protein target embeddings")
    all_protein_embeddings, all_protein_ids = torch.load(
        os.path.join(CKPT_NAME, "protein_target_embeddings.pkl")
    )
    all_protein_embeddings = all_protein_embeddings.float()
    logger.info(
        f"shape of precalculated embeddings matrix: {all_protein_embeddings.shape}"
    )

    logger.info("Loading DrugBank info")
    # Load DrugBank info, namely the mapping from DrugBank IDs to mechanism
    # of action descriptions and ProCyon-Instruct numeric IDs.
    drugbank_info = pd.read_pickle(
        os.path.join(
            DATA_DIR,
            "integrated_data",
            "v1",
            "drugbank",
            "drugbank_info_filtered_composed.pkl",
        )
    )
    db_map = {row["drugbank_id"]: row["moa"] for _, row in drugbank_info.iterrows()}
    db_idx_map = {
        row["drugbank_id"]: row["index"] for _, row in drugbank_info.iterrows()
    }
    logger.info("Done loading DrugBank info")

    logger.info("entering task description and prompt")
    # read the task description from a file
    with open(args.task_desc_infile, "r") as f:
        task_desc = f.read()
    task_desc = task_desc.replace("\n", " ")
    logger.info(f"Task description: {task_desc}")

    # Next we set up the specific prompt contexts provided for retrieval using bupropion and depression.
    db_id = "DB01156" # DrugBank ID for bupropion
    drug_desc = db_map[db_id] # the drug description from the DrugBank map

    # read the disease description from a file
    with open(args.disease_desc_infile, "r") as f:
        disease_desc = f.read()
    disease_desc = disease_desc.replace("\n", " ")
    disease_prompt = "Disease: {} Drug: {}".format(disease_desc, drug_desc)
    logger.info(f"Task description: {disease_prompt}")

    logger.info("Done entering task description and prompt")

    logger.info("Now performing protein retrieval for example 1")

    if inference_bool:

        # Create input for retrieval
        input_simple = create_input_retrieval(
            input_description=disease_prompt,
            data_args=data_args,
            drug_input_idx=db_idx_map[db_id],
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
        df_dep = get_proteins_from_embedding(all_protein_embeddings, model_out, top_k=None)
        logger.info(f"top results: {df_dep.head(10).to_dict(orient='records')}")

        logger.info("Done performaing protein retrieval for example 1")

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



    run_retrieval(args.task_desc_infile, args.disease_desc_infile, args.inference_bool)
