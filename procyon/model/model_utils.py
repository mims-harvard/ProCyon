import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from procyon.data.data_utils import DATA_DIR
from procyon.data.dataset import ProteinEvalDataset


def create_mlp(n_layers, in_features, out_features, hidden_features = 256, dropout_rate = 0.25):
    """
    From ChatGPT:
    Create a PyTorch sequential module with 'n' linear layers and ReLU activations.

    Parameters:
        n_layers (int): Number of linear layers in the sequential module.

    Returns:
        nn.Sequential: PyTorch sequential module.
    """
    layers = []

    if n_layers == 1:
        return nn.Sequential(nn.Linear(in_features, out_features, bias=False))

    for i in range(n_layers):
        # Add linear layer with ReLU activation except for the last layer
        in_size = hidden_features if i > 0 else in_features
        if i < n_layers - 1:
            layers.append(nn.Linear(in_size, hidden_features))
            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.GELU())
        else:
            # For the last layer, don't apply ReLU activation
            layers.append(nn.Linear(in_size, out_features))

    return nn.Sequential(*layers)

ALL_PROTEINS_FILE = os.path.join(DATA_DIR, "integrated_data/v1/protein/protein_info_filtered.pkl")
def get_all_protein_embeddings(
        model,
        batch_size = 16,
        all_proteins_file = ALL_PROTEINS_FILE
    ):
    '''
    Gets embeddings for every protein in our stored protein file
    '''
    df_all_prots = pd.read_pickle(ALL_PROTEINS_FILE)
    protein_dataset = ProteinEvalDataset(df_all_prots['index'])
    protein_dataloader = DataLoader(
        protein_dataset,
        batch_size = batch_size,
        num_workers = 2,
        drop_last = False,
        shuffle = False,
        pin_memory = True,
    )

    # Run inference loop:
    # Passing proteins via dataloader:
    model_device = model.device
    extra_protein_embeddings = []
    all_prot_inds = []
    for i, model_inputs in enumerate(protein_dataloader):
        if isinstance(model_inputs, dict):
            model_inputs["data"] = model_inputs["data"].to(model_device)
            protein_inputs = model_inputs
        else:
            all_prot_inds.append(model_inputs)
            protein_inputs = model_inputs.to(model_device)

        out = model.forward_sequences(protein_inputs)
        extra_protein_embeddings.append(out["shared"].detach().clone().cpu())

    extra_prot_embeds = torch.cat(extra_protein_embeddings, dim = 0)
    all_prot_inds = torch.cat(all_prot_inds).flatten()

    return extra_prot_embeds, all_prot_inds

ARCH_ARGS = [
    "protein_encoder_num_params",
    "aaseq_encoder_num_params",
    "protein_tokenizer_name",
    "aaseq_tokenizer_name",
    "text_encoder_fname",
    "text_tokenizer_name",
    "num_layers_token_projector",
    "num_layers_shared_projector",
    "num_layers_lm_projector",
    "roll_num",
]

def check_architecture_args(new_config, source_config):
    '''
    Asserts that the architecture arguments match between the two configs
    '''
    for n in ARCH_ARGS:
        if getattr(new_config, n) != getattr(source_config, n):
            return False
    return True

def detect_zero_stages(
        train_args,
        source_zero_stage = None,
        target_zero_stage = None,
    ):
    # Detect source and target zero stages:
    if source_zero_stage is None:

        # 1. attempt to get the zero stage from checkpoint
        # resume_from_checkpoint -> train args -> deepspeed_config -> zero stage
        ckpt_train_args_path = os.path.join(train_args.resume_from_checkpoint, "training_args.pt")
        assert os.path.exists(ckpt_train_args_path), "Training args for checkpoint do not exist, please provide source_zero_stage"

        ds_config = json.load(open(torch.load(ckpt_train_args_path).deepspeed_config, "r"))
        if not ("zero_optimization" in ds_config.keys()):
            source_zero_stage = -1
        else:
            source_zero_stage = ds_config["zero_optimization"]["stage"]

    if target_zero_stage is None:
        assert os.path.exists(train_args.deepspeed_config), f"Cannot automatically detect zero stage if deepspeed config not provided. {train_args.deepspeed_config} path does not exist."
        ds_config = json.load(open(train_args.deepspeed_config, "r"))
        if not ("zero_optimization" in ds_config.keys()):
            target_zero_stage = -2 # Must be different than above
        else:
            target_zero_stage = ds_config["zero_optimization"]["stage"]

    return source_zero_stage, target_zero_stage

def compute_conflict_matrix(id1, id2):

    # Repeat along each dimension to catch index conflicts:
    id1_r0 = id1.repeat(id1.shape[0],1)
    id1_r1 = id1.unsqueeze(1).repeat(1,id1.shape[0])

    id2_r0 = id2.repeat(id2.shape[0],1)
    id2_r1 = id2.unsqueeze(1).repeat(1,id2.shape[0])

    conflict_matrix = (id1_r0 == id1_r1) & (~(id2_r0 == id2_r1))

    return conflict_matrix

def compute_full_twosided_conflict_matrix():
    pass

def left_pad_tensors(tensors, pad_value=0):
    # Determine the maximum length
    max_length = max(tensor.size(0) for tensor in tensors)

    # Pad tensors and create attention masks
    padded_tensors = []
    attention_masks = []
    for tensor in tensors:
        pad_size = max_length - tensor.size(0)
        padded_tensor = torch.cat([torch.full((pad_size,), pad_value), tensor])
        attention_mask = torch.cat([torch.zeros(pad_size), torch.ones(tensor.size(0))])

        padded_tensors.append(padded_tensor)
        attention_masks.append(attention_mask)

    # Stack tensors and masks to create batched tensor and mask
    batched_tensor = torch.stack(padded_tensors)
    attention_mask = torch.stack(attention_masks)

    return batched_tensor, attention_mask