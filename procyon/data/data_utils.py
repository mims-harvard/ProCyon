import logging
from typing import List, Optional, Any

import torch
from torch import nn
from esm.data import Alphabet, BatchConverter

import pandas as pd
import numpy as np
import pickle

import os

from procyon.data.constants import ENTITY_DESCRIPTION_NAMES

from dotenv import load_dotenv

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")

try:  # If HOME_DIR can't be retrieved, that's ok bc it's only used for Deepspeed, some other training args
    HOME_DIR = os.getenv("HOME_DIR")
except:
    HOME_DIR = ""
MODEL_DIR = os.path.join(DATA_DIR, "model_outputs/pretrain")
assert DATA_DIR is not None, "DATA_DIR must be set in .env file"


def process_go_sims(go_sims, negative_sampling_strategy):
    return go_sims


def process_text_sims(text_sims, negative_sampling_strategy):
    return text_sims


def process_aaseq_sims(aaseq_sims, negative_sampling_strategy):
    return aaseq_sims


def process_pfam_sims(pfam_sims, negative_sampling_strategy):
    return pfam_sims


def process_protein_sims(protein_sims, negative_sampling_strategy):
    return protein_sims


def process_domain_sims(domain_sims, negative_sampling_strategy):
    return domain_sims


def convert_batch_protein(
    ids: List[int],
    is_protein_tokenized: bool,
    batch_converter: BatchConverter,
    protein_sequences: List[str],
    protein_tokens: List[int],
    protein_tokenizer: Alphabet,
    max_protein_len: Optional[int] = None,
) -> torch.Tensor:
    """Convert protein ids to protein tokens
    Assume that:
    1. if tokenized, must already added [CLS] and [EOS]
    2. protein_tokenizer is from ESM (aka alphabet)
    3. if needed batch_converter and max_protein_len, the input batch_converter has already required max_protein_len when initiated
    """
    if not is_protein_tokenized:
        batch_sequences = [("", protein_sequences[idx]) for idx in ids]
        _, _, batch_toks = batch_converter(batch_sequences)  # added [EOS] and [BOS]

    else:
        seqs_toks = [protein_tokens[idx] for idx in ids]

        # validate that the tokens already have [EOS] and [BOS]
        assert seqs_toks[0][0] == protein_tokenizer.cls_idx
        assert seqs_toks[0][-1] == protein_tokenizer.eos_idx

        if max_protein_len is not None:
            seqs_toks = [seq_toks[:max_protein_len] for seq_toks in seqs_toks]
        max_len = max(len(toks) for toks in seqs_toks)
        batch_toks = torch.empty(
            (len(ids), max_len),
            dtype=torch.long,
        )
        batch_toks.fill_(protein_tokenizer.padding_idx)

        for i, seq_toks in enumerate(seqs_toks):
            batch_toks[i, : len(seq_toks)] = seq_toks

    return batch_toks


def convert_batch_text(
    ids: List[int],
    is_text_tokenized: bool,
    text_sequences: np.ndarray,
    text_tokens: List[int],
    text_tokenizer: Any,
    max_text_len: int = None,
):
    """Convert text sequences (indexed by some ids) to text tokens"""

    if not is_text_tokenized:

        assert text_tokenizer.text_tokenizer_name is not None

        if text_tokenizer.text_tokenizer_name.lower().startswith("pubmedbert"):
            inputs = text_tokenizer(
                text_sequences[ids].tolist(),
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_text_len,
            )
            batch_toks, batch_attn_masks = inputs["input_ids"], inputs["attention_mask"]

        else:
            raise ValueError(
                f"Tokenizer {text_tokenizer.text_tokenizer_name} not expected"
            )

    else:
        seqs_toks = [text_tokens[idx] for idx in ids]

        if max_text_len is not None:
            seqs_toks = [seq_toks[:max_text_len] for seq_toks in seqs_toks]
        max_len = max(len(toks) for toks in seqs_toks)
        batch_toks = torch.empty(
            (len(ids), max_len),
            dtype=torch.long,
        )
        batch_toks.fill_(text_tokenizer.pad_token_id)

        for i, seq_toks in enumerate(seqs_toks):
            batch_toks[i, : len(seq_toks)] = seq_toks

        batch_attn_masks = (batch_toks != text_tokenizer.pad_token_id).long()

    return batch_toks, batch_attn_masks


def get_text_sequences(text_type, text_info, text_variant_type="standard"):
    if text_type == "go":
        go_type_converter = {
            "Process": "Biological Process",
            "Function": "Molecular Function",
            "Component": "Cellular Component",
        }
        if (
            text_variant_type == "standard"
            or text_variant_type == "description_combined"
        ):
            return (
                text_info["go_name"]
                + ":\n"
                + text_info["go_type"].apply(lambda x: go_type_converter[x])
                + ";\n"
                + text_info["go_def"]
                + "\n"
                + text_info["go_abstracts"].apply(lambda x: "; ".join(x))
                + "\n"
            ).values
        elif text_variant_type == "name_def":
            return (text_info["go_name"] + ":\n" + text_info["go_def"] + "\n").values
        elif text_variant_type == "def_only":
            return (text_info["go_def"] + "\n").values
        elif text_variant_type == "abstract_only":
            return (
                text_info["go_name"]
                + ":\n"
                + text_info["go_abstracts"].apply(lambda x: "; ".join(x))
                + "\n"
            ).values
        else:
            raise NotImplementedError
    elif text_type == "pfam":
        if text_variant_type == "standard":
            return (text_info["description_combined"] + "\n").values
        else:
            raise NotImplementedError
    elif text_type == "reactome":
        if text_variant_type == "standard":
            return (
                text_info["name"].apply(lambda name_list: "; ".join(name_list) + ":\n")
                + text_info["description"].apply(
                    lambda name_list: ". ".join(name_list) + "\n"
                )
            ).values
        else:
            raise NotImplementedError
    elif text_type == "ec":
        if text_variant_type == "standard":
            return (
                text_info["explorenz_accepted_name"]
                + text_info["mcsa_enzyme_name"].apply(
                    lambda name: (
                        " (" + name + ")" if name == name and name is not None else ""
                    )
                )
                + ":\n"
                + "Reaction: "
                + text_info["explorenz_reaction"]
                + "\n"
                + text_info["explorenz_comments"].apply(
                    lambda comment: (
                        comment + "\n"
                        if comment == comment and comment is not None
                        else ""
                    )
                )
                + text_info["mcsa_description"].apply(
                    lambda description: (
                        description + "\n"
                        if description == description and description is not None
                        else ""
                    )
                )
                + text_info["mcsa_mechanisms_text"].apply(
                    lambda mechanisms: (
                        mechanisms + "\n"
                        if mechanisms == mechanisms and mechanisms is not None
                        else ""
                    )
                )
                + text_info["mcsa_steps"].apply(
                    lambda steps: (
                        "; ".join(
                            [f"Step {i+1}: " + step for i, step in enumerate(steps)]
                        )
                        + "\n"
                        if steps == steps
                        else ""
                    )
                )
            ).values
        elif text_variant_type == "explorenz":
            return (
                text_info["explorenz_accepted_name"]
                + text_info["mcsa_enzyme_name"].apply(
                    lambda name: (
                        " (" + name + ")" if name == name and name is not None else ""
                    )
                )
                + ":\n"
                + "Reaction: "
                + text_info["explorenz_reaction"]
                + "\n"
                + text_info["explorenz_comments"].apply(
                    lambda comment: (
                        comment + "\n"
                        if comment == comment and comment is not None
                        else ""
                    )
                )
            ).values
        elif text_variant_type == "full_with_wikidoc":
            raise NotImplementedError
        else:
            raise NotImplementedError
    elif text_type == "drugbank":
        if text_variant_type == "standard":
            return (
                text_info["drug_name"]
                + ":\n"
                + text_info["description"]
                + "\n"
                + text_info["indication"].apply(
                    lambda indication: indication + "\n" if indication != "" else ""
                )
                + "\n"
                + text_info["moa"].apply(lambda moa: moa + "\n" if moa != "" else "")
            ).values
        elif text_variant_type == "description_only":
            return (
                text_info["drug_name"] + ":\n" + text_info["description"] + "\n"
            ).values
        elif text_variant_type == "indication_only":
            return (
                text_info["drug_name"]
                + ":\n"
                + text_info["indication"].apply(
                    lambda indication: indication + "\n" if indication != "" else ""
                )
            ).values
        elif text_variant_type == "moa_only":
            return (
                text_info["drug_name"]
                + ":\n"
                + text_info["moa"].apply(lambda moa: moa + "\n" if moa != "" else "")
            ).values
        else:
            raise NotImplementedError
    elif text_type == "omim":
        if text_variant_type == "standard":
            return (
                text_info["omim_title"]
                + ":\n"
                + text_info["omim_description"].apply(
                    lambda desc: (
                        desc + "\n" if desc == desc and desc is not None else ""
                    )
                )
                + text_info["mondo_definition"].apply(
                    lambda desc: (
                        desc + "\n" if desc == desc and desc is not None else ""
                    )
                )
                + text_info["umls_description"].apply(
                    lambda desc: (
                        desc + "\n" if desc == desc and desc is not None else ""
                    )
                )
                + text_info["orphanet_definition"].apply(
                    lambda desc: (
                        desc + "\n" if desc == desc and desc is not None else ""
                    )
                )
                + text_info["orphanet_clinical_description"].apply(
                    lambda desc: (
                        desc + "\n" if desc == desc and desc is not None else ""
                    )
                )
                + text_info["mayo_symptoms"].apply(
                    lambda desc: (
                        desc + "\n" if desc == desc and desc is not None else ""
                    )
                )
                + text_info["mayo_causes"].apply(
                    lambda desc: (
                        desc + "\n" if desc == desc and desc is not None else ""
                    )
                )
                + text_info["mayo_risk_factors"].apply(
                    lambda desc: (
                        desc + "\n" if desc == desc and desc is not None else ""
                    )
                )
            ).values
        else:
            raise NotImplementedError
    elif text_type == "disgenet":
        if text_variant_type == "standard":
            return (text_info["allDescriptions"] + "\n").values
        else:
            raise NotImplementedError
    elif text_type == "protein":
        # Currently abusing 'text_type' to be the other half of "protein" in a "protein_protein" relation.
        return None
    else:
        raise NotImplementedError


def get_text_sequences_compositions(text_type, text_info, column_subset=None):
    if text_type == "protein":
        return None
    else:
        if column_subset is None:
            text_cols = ENTITY_DESCRIPTION_NAMES[text_type]
        else:
            text_cols = column_subset[text_type]
        return text_info[text_cols]


def load_aaseq_embeddings(
    aaseq_embeddings_path: str, aaseq_embeddings_idmap_path: str, aaseq_type: str
):
    """Load saved embeddings from specified paths, and perform reordering so that the embeddings match the order of {aaseq}_info"""
    aaseq_embeddings = torch.load(aaseq_embeddings_path)
    aaseq_info = pd.read_pickle(
        os.path.join(
            DATA_DIR, f"integrated_data/v1/{aaseq_type}/{aaseq_type}_info_filtered.pkl"
        )
    ).rename(columns={"index": "idx"})
    with open(aaseq_embeddings_idmap_path, "rb") as f:
        aaseq_map = pickle.load(f)
    aaseq_map = [aaseq.split(" ")[0] for aaseq in aaseq_map]
    aaseq_embeddings = aaseq_embeddings[
        aaseq_info.set_index(f"{aaseq_type}_id")
        .loc[aaseq_map]
        .reset_index()
        .reset_index()
        .sort_values("idx")["index"]
        .values
    ]
    return aaseq_embeddings


def load_protein_struct_embeddings(protein_struct_embeddings_path):
    with open(protein_struct_embeddings_path, "rb") as f:
        z = torch.load(f)
    return z


def load_drug_structure_embeddings(drug_struct_embeddings_path):
    with open(drug_struct_embeddings_path, "rb") as f:
        z = torch.load(f)
    return z


def load_text_embeddings(text_embeddings_path: str, text_type: str):
    """Load saved text embeddings from specfied path."""
    assert text_embeddings_path is not None, "text embeddings path must be supplied"
    text_info = pd.read_pickle(
        os.path.join(
            DATA_DIR, f"integrated_data/v1/{text_type}/{text_type}_info_filtered.pkl"
        )
    )
    text_embeddings = torch.load(text_embeddings_path).float()
    assert len(text_info) == len(text_embeddings)
    return text_embeddings


def first_unique_value_in_pandas_df(df, col):
    """
    From ChatGPT (also docstring)

    Return the first unique value in a specified column of a pandas DataFrame along with its corresponding row.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        col (str): The name of the column to find unique values.

    Returns:
        pandas.DataFrame: A DataFrame containing the first row for each unique value in the specified column.

    Example:
        Consider a DataFrame 'df' with columns 'Col_1', 'Col_2', and 'Col_3':

        df
          Col_1  Col_2 Col_3
        0     A     12   R14
        1     B     71   R10
        2     A     20   R35
        3     F     23   R67
        4     D     53   R12

        Calling the function with `first_unique_value_in_pandas_df(df, 'Col_1')` will return:

        result_df
          Col_1  Col_2 Col_3
        0     A     12   R14
        1     B     71   R10
        3     F     23   R67
        4     D     53   R12
    """

    unique_values = df[col].unique().tolist()

    # Initialize an empty list to store the filtered DataFrames
    filtered_dfs = []

    for val in unique_values:
        # Filter the DataFrame for each unique value in Col_1
        filtered_df = df[df[col] == val]
        # Get the first row of the filtered DataFrame
        first_row = filtered_df.iloc[0]
        # Append the first row to the list
        filtered_dfs.append(first_row)

    # Create a new DataFrame with the first rows of each unique value
    result_df = pd.DataFrame(filtered_dfs)

    return result_df


def get_relation_fname(
    aaseq_type,  # Should be the type of amino acid sequence used in the dataset (e.g., "protein")
    text_type,  # Should be the dataset key or text type (e.g., "go")
    go_split_method,  # Should be random_{text_type}_centric or other
    shot_level,  # Should be "five_shot" or "zero_shot" or "pt_ft", etc. based on the dataset
    split="test",
):
    assert split in {
        "val",
        "test",
    }, "This function does not retrieve training relation fname"
    sub_split_name = "eval" if split == "test" else "CL_val"
    text_split_method = (
        go_split_method if text_type == "go" else f"random_{text_type}_centric"
    )
    name = os.path.join(
        "integrated_data",
        "v1",
        f"{aaseq_type}_{text_type}",
        text_split_method,
        f"{aaseq_type}_{text_type}_relations_{sub_split_name}_{shot_level}_indexed_with_10_negatives.csv",
    )
    return name
