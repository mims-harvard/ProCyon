import json
import math
import os

from collections.abc import Callable
from typing import (
    Dict,
    List,
    Optional,
)

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import trange

from procyon.data.constants import (
    CAPTION_SUBSETS,
    RETRIEVAL_SUBSETS,
    QA_SUBSETS,
)
from procyon.data.data_utils import (
    DATA_DIR,
    HOME_DIR,
    get_text_sequences_compositions,
)
from procyon.data.instruct_tune.instruct_constructor import (
    get_prompt,
    get_prompt_open_def,
)
from procyon.data.it_collator import construct_task_id
from procyon.evaluate.framework.qa import AbstractQAModel
from procyon.evaluate.framework.utils import move_inputs_to_device
from procyon.training.training_args_IT import DataArgs
from procyon.training.train_utils import (
    get_final_tokens,
    get_after_answer_tokens,
)

UNIPROT_IDS = pd.read_pickle(
    os.path.join(DATA_DIR, "integrated_data/v1/protein/", "protein_info_filtered.pkl")
)[["index", "protein_id", "name"]]

functional_descriptions = pd.read_pickle(
    os.path.join(
        DATA_DIR, "integrated_data/v1/protein/uniprot_functional_descriptions.pkl"
    )
).sort_values("index", axis=0)["function"]
# Make sure it's sorted by protein index


def uniprot_id_to_index(uniprot_id: str) -> int:
    assert (
        UNIPROT_IDS["protein_id"] == uniprot_id
    ).sum() == 1, "ID {} not found in internal database".format(uniprot_id)
    i = UNIPROT_IDS["index"].loc[UNIPROT_IDS["protein_id"] == uniprot_id].item()
    return i


def index_to_uniprot_id(i: int) -> str:
    uniprot_id = UNIPROT_IDS["protein_id"].loc[UNIPROT_IDS["index"] == i].item()
    return uniprot_id


def create_caption_input_simple(
    input_aaseq_ids: List[int],
    data_args: DataArgs,
    input_description: str = None,
    drug_inputs: List[int] = None,
    task_definition: str = None,
    instruction_source_dataset: str = None,
    instruction_source_relation: str = "all",
    aaseq_type: str = "protein",
    task_type: str = "caption",
    icl_example_number: int = 1,
    device=None,
    disease_context_augmentation=False,
) -> Dict:
    """
    Args:
        input_protein_ids: List[int]
            - Protein IDs you input to your example (i.e. not ICL examples)
        data_args: procyon.training.training_args_IT.DataArgs
            - DataArgs config used for the model that will ingest the generated instructions
        input_description: str
            - Provide only for QA - no effect in captioning
        task_definition: str
            - If provided, this will replace the part after the "Definition: " tag in the instructions
        instruction_source_dataset: str
            - Should describe the dataset, or text type
            - Options: ["disgenet", "pfam", "drugbank", "ec", "go", "gtop", "omim", "uniprot", "reactome", "protein" (which is STRING protein-protein)]
        instruction_source_relation: str
            - Should describe the relation within that dataset
            - For most datasets, this is "all", but for some datasets you can make it more precise:
                go: "process", "function", "component"
                drugbank: "target", "carrier", "enzyme", "transporter"
                protein: "coexpression", "homology", "experiments"
        aaseq_type: str
            - Should be "protein", "domain", or later "peptide"
            - Tells the model which embedding layer we should access, i.e., embeddings for domains or proteins
            - Also important for defining the instruction - ex: domain-go vs. protein-go
        task_type: str
            - Options: ["caption", "qa"]
            - Toggle between caption and QA task
        icl_example_number: int
            - Number of in-context examples to use for the given dataset
            - We use 1 in all of our training, so recommended to set to 1
            - Options: [0, 1, 2]
    Output:
        Dict of model inputs expected for `UnifiedProcyon.forward()`
    """

    assert drug_inputs is None

    example_descriptions = []
    if instruction_source_dataset is not None:
        instruction_source_dataset = instruction_source_dataset.lower()

        # Load instructions from instruction constructor
        task_id = construct_task_id(
            aaseq_type=aaseq_type,
            text_type=instruction_source_dataset,
            relation_type=instruction_source_relation,
            task_type="caption",
        )
        fpath = os.path.join(
            HOME_DIR, f"procyon/data/instruct_tune/tasks/{task_id}.json"
        )

        with open(fpath, "r") as j:
            task_json = json.loads(j.read())

        if task_definition is None:
            instruction, _, _, example_text_ids, example_aaseq_ids = get_prompt(
                task=task_json,
                num_examples=icl_example_number,
                is_special_definition=False,
                is_ppi=(instruction_source_dataset == "protein"),
                aaseq_type=aaseq_type,
            )
        else:
            instruction, _, _, _, example_text_ids, example_aaseq_ids = (
                get_prompt_open_def(
                    task=task_json,
                    num_examples=icl_example_number,
                    is_special_definition=False,
                    is_ppi=(instruction_source_dataset == "protein"),
                    aaseq_type=aaseq_type,
                )
            )

            instruction = instruction.format(definition=task_definition)

        col_subset = None
        if (task_type == "qa") and (data_args.qa_subset_version is not None):
            col_subset = QA_SUBSETS[data_args.qa_subset_version]
        elif (task_type == "caption") and (
            data_args.caption_subset_version is not None
        ):
            col_subset = CAPTION_SUBSETS[data_args.caption_subset_version]

        text_df = pd.read_pickle(
            os.path.join(
                DATA_DIR,
                "integrated_data",
                "v1",
                instruction_source_dataset,
                f"{instruction_source_dataset}_info_filtered_composed.pkl",
            )
        )
        text_sequences = get_text_sequences_compositions(
            text_type=instruction_source_dataset,
            text_info=text_df,
            column_subset=col_subset,
        )

        # Get example text
        example_descriptions = text_sequences.iloc[example_text_ids, 0].tolist()

    else:
        # Insert task definition manually
        raise NotImplementedError

    unique_aaseq_indices = torch.LongTensor(example_aaseq_ids + input_aaseq_ids).to(
        device
    )

    descriptions = example_descriptions

    if input_description is not None:
        descriptions = descriptions + [input_description]

    instruction = (
        instruction[:-5] if instruction[-5:] == "[EXT]" else instruction
    )  # Trim EXT

    if disease_context_augmentation:
        instruction = instruction.replace("[CONTEXT]", "[EXT]")
        # Access contexts based on aaseq_indices
        contexts = functional_descriptions.iloc[
            unique_aaseq_indices.cpu().numpy()
        ].tolist()
        # Interleave context with descriptions:
        desc_new = []
        for i in range(len(contexts)):
            c = contexts[i]
            desc_new.append("Context: {}".format(c))
            if i != (len(contexts) - 1):
                d = descriptions[i]
                desc_new.append(d)

        # for c, d in zip(contexts, descriptions):
        #     desc_new.append(d)
        #     desc_new.append("Context: {}".format(c))
        descriptions = desc_new
    else:
        instruction = instruction.replace("[CONTEXT]", "")

    model_input = {
        "data": {
            "seq": unique_aaseq_indices,
            "seq_idx": unique_aaseq_indices,
            "text": descriptions,
            "drug": None,  # Optional
        },
        "input": {
            "seq": torch.arange((unique_aaseq_indices).shape[0]).unsqueeze(0).tolist(),
            "text": torch.arange(len(descriptions))
            .unsqueeze(0)
            .tolist(),  # List not tensor
            "drug": None,
            # "drug": np.arange(len(unique_drug_indices)).tolist(),
        },
        "target": {
            "seq": None,
            "text": None,
            "drug": None,
        },
        "instructions": [instruction],
    }

    return model_input


def create_qa_input_simple(
    input_aaseq_ids: List[int],
    data_args: DataArgs,
    input_description: str,
    drug_inputs: List[int] = None,
    task_definition: str = None,
    instruction_source_dataset: str = None,
    instruction_source_relation: str = "all",
    aaseq_type: str = "protein",
    icl_example_number: int = 1,
    device=None,
    disease_context_augmentation=False,
) -> Dict:
    """
    - Should use this if you're using QA with the preset proteins/domains - i.e., not free-form amino acid sequences

    Args:
        input_protein_ids: List[int]
            - Protein IDs you input to your example (i.e. not ICL examples)
        data_args: procyon.training.training_args.DataArgs
            - DataArgs config used for the model that will ingest the generated instructions
        input_description: str
            - Provide only for QA - no effect in captioning
        task_definition: str
            - If provided, this will replace the part after the "Definition: " tag in the instructions
        instruction_source_dataset: str
            - Should describe the dataset, or text type
            - Options: ["disgenet", "pfam", "drugbank", "ec", "go", "gtop", "omim", "uniprot", "reactome", "protein" (which is STRING protein-protein)]
        instruction_source_relation: str
            - Should describe the relation within that dataset
            - For most datasets, this is "all", but for some datasets you can make it more precise:
                go: "process", "function", "component"
                drugbank: "target", "carrier", "enzyme", "transporter"
                protein: "coexpression", "homology", "experiments"
        aaseq_type: str
            - Should be "protein", "domain", or later "peptide"
            - Tells the model which embedding layer we should access, i.e., embeddings for domains or proteins
            - Also important for defining the instruction - ex: domain-go vs. protein-go
        task_type: str
            - Options: ["caption", "qa"]
            - Toggle between caption and QA task
        icl_example_number: int
            - Number of in-context examples to use for the given dataset
            - We use 1 in all of our training, so recommended to set to 1
            - Options: [0, 1, 2]
    Output:
        Dict of model inputs expected for `UnifiedProcyon.forward()`
    """

    assert drug_inputs is None

    example_descriptions = []
    if instruction_source_dataset is not None:
        instruction_source_dataset = instruction_source_dataset.lower()

        # Load instructions from instruction constructor
        task_id = construct_task_id(
            aaseq_type=aaseq_type,
            text_type=instruction_source_dataset,
            relation_type=instruction_source_relation,
            task_type="qa",
        )
        fpath = os.path.join(
            HOME_DIR, f"procyon/data/instruct_tune/tasks/{task_id}.json"
        )

        with open(fpath, "r") as j:
            task_json = json.loads(j.read())

        if task_definition is None:
            instruction, _, _, example_text_ids, example_aaseq_ids = get_prompt(
                task=task_json,
                num_examples=icl_example_number,
                is_special_definition=False,
                is_ppi=(instruction_source_dataset == "protein"),
                aaseq_type=aaseq_type,
            )
        else:
            instruction, _, _, _, example_text_ids, example_aaseq_ids = (
                get_prompt_open_def(
                    task=task_json,
                    num_examples=icl_example_number,
                    is_special_definition=False,
                    is_ppi=(instruction_source_dataset == "protein"),
                    aaseq_type=aaseq_type,
                )
            )

            instruction = instruction.format(definition=task_definition)

        col_subset = QA_SUBSETS[data_args.qa_subset_version]

        text_df = pd.read_pickle(
            os.path.join(
                DATA_DIR,
                "integrated_data",
                "v1",
                instruction_source_dataset,
                f"{instruction_source_dataset}_info_filtered_composed.pkl",
            )
        )
        text_sequences = get_text_sequences_compositions(
            text_type=instruction_source_dataset,
            text_info=text_df,
            column_subset=col_subset,
        )

        # Get example text
        # example_descriptions = text_sequences.iloc[example_text_ids,0].tolist()
        example_descriptions = [
            text_sequences.iloc[example_text_ids[i], :]
            .loc[text_sequences.iloc[example_text_ids[i], :].notna()]
            .tolist()[0]
            for i in range(len(example_text_ids))
        ]

    else:
        # Insert task definition manually
        raise NotImplementedError

    unique_aaseq_indices = torch.LongTensor(example_aaseq_ids + input_aaseq_ids).to(
        device
    )

    descriptions = example_descriptions

    if input_description is not None:
        descriptions = descriptions + [input_description]

    instruction = (
        instruction[:-5] if instruction[-5:] == "[EXT]" else instruction
    )  # Trim EXT

    if disease_context_augmentation:
        instruction = instruction.replace("[CONTEXT]", "[EXT]")
        # Access contexts based on aaseq_indices
        contexts = functional_descriptions.iloc[
            unique_aaseq_indices.cpu().numpy()
        ].tolist()
        # Interleave context with descriptions:
        desc_new = []
        for c, d in zip(contexts, descriptions):
            desc_new.append(d)
            desc_new.append("Context: {}".format(c))
        descriptions = desc_new
    else:
        instruction = instruction.replace("[CONTEXT]", "")

    instruction = instruction.format(answer="null")

    model_input = {
        "data": {
            "seq": unique_aaseq_indices,
            "seq_idx": unique_aaseq_indices,
            "text": descriptions,
            "drug": None,  # Optional
        },
        "input": {
            "seq": torch.arange((unique_aaseq_indices).shape[0]).unsqueeze(0).tolist(),
            "text": torch.arange(len(descriptions))
            .unsqueeze(0)
            .tolist(),  # List not tensor
            "drug": None,
            # "drug": np.arange(len(unique_drug_indices)).tolist(),
        },
        "target": {
            "seq": None,
            "text": None,
            "drug": None,
        },
        "instructions": [instruction],
    }

    return model_input


def create_qa_input_aaseq(
    input_aaseq: str,
    data_args,
    input_description: str,
    drug_inputs: List[int] = None,
    task_definition: str = None,
    instruction_source_dataset: str = None,
    instruction_source_relation: str = "all",
    aaseq_type: str = "protein",
    icl_example_number: int = 1,
    device=None,
):
    """
    - Should use this if you're using QA with the preset proteins/domains - i.e., not free-form amino acid sequences

    Args:
        input_protein_ids: List[int]
            - Protein IDs you input to your example (i.e. not ICL examples)

        input_description: str
            - Provide only for QA - no effect in captioning
        task_definition: str
            - If provided, this will replace the part after the "Definition: " tag in the instructions
        instruction_source_dataset: str
            - Should describe the dataset, or text type
            - Options: ["disgenet", "pfam", "drugbank", "ec", "go", "gtop", "omim", "uniprot", "reactome", "protein" (which is STRING protein-protein)]
        instruction_source_relation: str
            - Should describe the relation within that dataset
            - For most datasets, this is "all", but for some datasets you can make it more precise:
                go: "process", "function", "component"
                drugbank: "target", "carrier", "enzyme", "transporter"
                protein: "coexpression", "homology", "experiments"
        aaseq_type: str
            - Should be "protein", "domain", or later "peptide"
            - Tells the model which embedding layer we should access, i.e., embeddings for domains or proteins
            - Also important for defining the instruction - ex: domain-go vs. protein-go
        task_type: str
            - Options: ["caption", "qa"]
            - Toggle between caption and QA task
        icl_example_number: int
            - Number of in-context examples to use for the given dataset
            - We use 1 in all of our training, so recommended to set to 1
            - Options: [0, 1, 2]
    Output:
        Dict of model inputs expected for `UnifiedProcyon.forward()`
    """

    assert drug_inputs is None

    example_descriptions = []
    if instruction_source_dataset is not None:
        instruction_source_dataset = instruction_source_dataset.lower()

        # Load instructions from instruction constructor
        task_id = construct_task_id(
            aaseq_type=aaseq_type,
            text_type=instruction_source_dataset,
            relation_type=instruction_source_relation,
            task_type="qa",
        )
        fpath = os.path.join(
            HOME_DIR, f"procyon/data/instruct_tune/tasks/{task_id}.json"
        )

        with open(fpath, "r") as j:
            task_json = json.loads(j.read())

        if task_definition is None:
            instruction, _, _, example_text_ids, example_aaseq_ids = get_prompt(
                task=task_json,
                num_examples=icl_example_number,
                is_special_definition=False,
                is_ppi=(instruction_source_dataset == "protein"),
                aaseq_type=aaseq_type,
            )
        else:
            instruction, _, _, _, example_text_ids, example_aaseq_ids = (
                get_prompt_open_def(
                    task=task_json,
                    num_examples=icl_example_number,
                    is_special_definition=False,
                    is_ppi=(instruction_source_dataset == "protein"),
                    aaseq_type=aaseq_type,
                )
            )

            instruction = instruction.format(definition=task_definition)

        icl_seqs = None
        if icl_example_number > 0:
            if aaseq_type == "protein":
                icl_seqs = [PROTEIN_SEQS[i] for i in example_aaseq_ids]
            elif aaseq_type == "domain":
                icl_seqs = [DOMAIN_SEQS[i] for i in example_aaseq_ids]
            else:
                raise NotImplementedError

        col_subset = QA_SUBSETS[data_args.qa_subset_version]

        text_df = pd.read_pickle(
            DATA_DIR
            + f"integrated_data/v1/{instruction_source_dataset}/{instruction_source_dataset}_info_filtered_composed.pkl"
        )
        text_sequences = get_text_sequences_compositions(
            text_type=instruction_source_dataset,
            text_info=text_df,
            column_subset=col_subset,
        )

        # Get example text
        example_descriptions = text_sequences.iloc[example_text_ids, 0].tolist()

    else:
        # Insert task definition manually
        raise NotImplementedError

    unique_aaseq_indices = torch.LongTensor(example_aaseq_ids + [0]).to(device)

    input_seq_tokens = convert_batch_sequences(icl_seqs + [input_aaseq]).to(device)

    descriptions = example_descriptions

    if input_description is not None:
        descriptions = descriptions + [input_description]

    instruction = (
        instruction[:-5] if instruction[-5:] == "[EXT]" else instruction
    )  # Trim EXT

    instruction = instruction.replace("[CONTEXT]", "")

    instruction = instruction.format(answer="null")

    model_input = {
        "data": {
            "seq": input_seq_tokens,
            "seq_idx": unique_aaseq_indices,
            "text": descriptions,
            "drug": None,  # Optional
        },
        "input": {
            "seq": torch.arange((unique_aaseq_indices).shape[0]).unsqueeze(0).tolist(),
            "text": torch.arange(len(descriptions))
            .unsqueeze(0)
            .tolist(),  # List not tensor
            "drug": None,
            # "drug": np.arange(len(unique_drug_indices)).tolist(),
        },
        "target": {
            "seq": None,
            "text": None,
            "drug": None,
        },
        "instructions": [instruction],
    }

    return model_input


def get_qa_logits_inference(model_out, padding_token=None, answer_token=None):
    preds = model_out["outputs"].logits.softmax(dim=-1).detach().clone().cpu()

    # Prepare labels:
    y_tok_total = model_out["text_toks"].detach().clone().cpu()

    if padding_token is not None:
        y_toks_inds = get_final_tokens(y_tok_total, padding_token=padding_token)
    elif answer_token is not None:
        y_toks_inds = get_after_answer_tokens(y_tok_total, answer_token=answer_token)
    else:
        raise ValueError(
            "One of padding_token or answer_token for get_qa_metrics must not be None"
        )

    y_toks = y_tok_total[torch.arange(y_tok_total.shape[0]), y_toks_inds]

    pred_tok_total = preds.argmax(dim=-1)
    # NOTE: Must subtract 1 from y_toks_inds because the output is shifted due to causal language modeling
    #   - Very difficult bug to find, but consider the documentation in BioGPT
    pred_toks = preds[torch.arange(pred_tok_total.shape[0]), (y_toks_inds - 1)]

    return pred_toks.detach().clone().cpu(), y_toks.detach().clone().cpu()


class ProCyonQAInference(AbstractQAModel):
    """
    Wrapper class based on ProCyonQAEval class
    """

    def __init__(self, model, device=None):
        self.model = model

        model.eval()
        self.device = device
        self.model = model.to(self.device)

        self.yes_token = model.yes_token
        self.no_token = model.no_token

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        """
        Forces normal calls to be no_grad, but you have the option of getting gradients
        """
        results_dict = self.fwd_pass(*args, **kwargs)

        return results_dict

    def fwd_pass(
        self,
        model_inputs,
        aaseq_type: str = "protein",
        output_attentions=None,
    ):
        results_dict = {"pred": None, "y": None, "out": None}

        out = self.model(
            move_inputs_to_device(model_inputs, self.device),
            return_mlm=False,
            retrieval=False,
            get_full_labels=True,
            aaseq_type=aaseq_type,
            crop_off=True,
            output_attentions=output_attentions,
        )

        # if output_hidden_states:
        results_dict["out"] = out

        pred_toks, _ = get_qa_logits_inference(out, answer_token=self.model.answer_idx)
        results_dict["pred"] = pred_toks

        return results_dict


DRUGMASK = torch.load(
    os.path.join(DATA_DIR, f"integrated_data/v1/drugbank/drugbank_mask.pt")
)


def create_input_retrieval(
    input_description: str,
    data_args: DataArgs,
    instruction_source_dataset: str,
    drug_input_idx: List[int] = None,
    task_definition: str = None,
    instruction_source_relation: str = "all",
    aaseq_type: str = "protein",
    icl_example_number: int = 1,
) -> Dict:
    """
    Args:
        input_description: str
            - Input description on which to query
        data_args: procyon.training.training_args.DataArgs
            - DataArgs config used for the model that will ingest the generated instructions
        instruction_source_dataset: str
            - Should describe the dataset, or text type
            - Options: ["disgenet", "pfam", "drugbank", "ec", "go", "gtop", "omim", "uniprot", "reactome", "protein" (which is STRING protein-protein)]
        task_definition: str
            - If provided, this will replace the part after the "Definition: " tag in the instructions
        instruction_source_relation: str
            - Should describe the relation within that dataset
            - For most datasets, this is "all", but for some datasets you can make it more precise:
                go: "process", "function", "component"
                drugbank: "target", "carrier", "enzyme", "transporter"
                protein: "coexpression", "homology", "experiments"
        aaseq_type: str
            - Should be "protein", "domain", or later "peptide"
            - Tells the model which embedding layer we should access, i.e., embeddings for domains or proteins
            - Also important for defining the instruction - ex: domain-go vs. protein-gok
        icl_example_number: int
            - Number of in-context examples to use for the given dataset
            - We use 1 in all of our training, so recommended to set to 1
            - Options: [0, 1, 2]
    Output:
        Dict of model inputs expected for `UnifiedProcyon.forward()`
    """

    example_descriptions = []
    instruction_source_dataset = instruction_source_dataset.lower()

    # Load instructions from instruction constructor
    task_id = construct_task_id(
        aaseq_type=aaseq_type,
        text_type=instruction_source_dataset,
        relation_type=instruction_source_relation,
        task_type="retrieval",
    )
    fpath = os.path.join(
        HOME_DIR,
        "procyon",
        "data",
        "instruct_tune",
        "tasks",
        f"{task_id}.json",
    )

    with open(fpath, "r") as j:
        task_json = json.loads(j.read())

    if task_definition is None:
        instruction, _, _, example_text_ids, example_aaseq_ids = get_prompt(
            task=task_json,
            num_examples=icl_example_number,
            is_special_definition=False,
            is_ppi=(instruction_source_dataset == "protein"),
            aaseq_type=aaseq_type,
        )
    else:
        instruction, _, _, _, example_text_ids, example_aaseq_ids = get_prompt_open_def(
            task=task_json,
            num_examples=icl_example_number,
            is_special_definition=False,
            is_ppi=(instruction_source_dataset == "protein"),
            aaseq_type=aaseq_type,
        )

        instruction = instruction.format(definition=task_definition)

    col_subset = RETRIEVAL_SUBSETS[data_args.retrieval_subset_version]

    text_df = pd.read_pickle(
        os.path.join(
            DATA_DIR,
            "integrated_data",
            "v1",
            instruction_source_dataset,
            f"{instruction_source_dataset}_info_filtered_composed.pkl",
        )
    )
    text_sequences = get_text_sequences_compositions(
        text_type=instruction_source_dataset,
        text_info=text_df,
        column_subset=col_subset,
    )

    # Get example text
    example_descriptions = text_sequences.iloc[example_text_ids, 0].tolist()

    input_drug = None
    drug_inputs = None
    if (instruction_source_dataset == "drugbank") and (drug_input_idx is not None):
        input_drug = []
        drug_inputs = []
        for i in range(len(example_descriptions)):
            if not DRUGMASK[example_text_ids[i]]:
                continue
            example_descriptions[i] = example_descriptions[i] + "\nDrug: <|drug|>"
            input_drug.append(i)
            drug_inputs.append(example_text_ids[i])
            # add to drug input

    if drug_input_idx is not None:

        # Automatic detection of if we have drug embeddings for this specific sample:
        if DRUGMASK[drug_input_idx].item():
            # Insert drug info:
            input_description = input_description + "\nDrug: <|drug|>"
            input_drug.append(len(input_drug))
            drug_inputs.append(drug_input_idx)
            # input_drug = torch.LongTensor([[0]]).tolist()
            # drug_inputs = torch.LongTensor([drug_input_idx]).to(device)
        else:
            print(
                "WARNING: not inserting drug index because we don't have one in our database"
            )
            input_drug = None
            drug_inputs = None
    else:
        input_drug = None
        drug_inputs = None

    if input_drug is not None:
        input_drug = torch.LongTensor(input_drug).unsqueeze(0).tolist()
        drug_inputs = torch.LongTensor(drug_inputs)

    if len(example_aaseq_ids) > 0:
        unique_aaseq_indices = torch.LongTensor(example_aaseq_ids)
    else:
        unique_aaseq_indices = None
    descriptions = example_descriptions + [input_description]

    instruction = (
        instruction[:-5] if instruction[-5:] == "[EXT]" else instruction
    )  # Trim EXT

    instruction = instruction.replace("[CONTEXT]", "")

    model_input = {
        "data": {
            "seq": unique_aaseq_indices,
            "seq_idx": unique_aaseq_indices,
            "text": descriptions,
            "drug": drug_inputs,  # Optional
        },
        "input": {
            "seq": (
                torch.arange((unique_aaseq_indices).shape[0]).unsqueeze(0).tolist()
                if (unique_aaseq_indices is not None)
                else None
            ),
            "text": torch.arange(len(descriptions))
            .unsqueeze(0)
            .tolist(),  # List not tensor
            "drug": input_drug,
        },
        "target": {  # This is only used for training
            "seq": None,
            "text": None,
            "drug": None,
        },
        "instructions": [instruction],
    }

    return model_input


def get_proteins_from_embedding(
    protein_embeds: torch.Tensor,
    model_out: Optional[Dict] = None,
    query_embeddings: Optional[torch.Tensor] = None,
    protein_ids: pd.DataFrame = UNIPROT_IDS,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Args:
        model_out: Optional[Dict]
            - Output dictionary returned from `UnifiedProcyon.forward` or can specify query embeddings directly via `query_embeddings`
        query_embeddings: Optional[torch.Tensor]
            - Tensor directly specifying query embeddings to use for retrieval
        protein_embeds: torch.Tensor
            - Tensor of embeddings of proteins to perform retrieval over
        protein_ids: pd.DataFrame
            - pd.DataFrame with `protein_id` and `name` columns corresponding to the proteins in `protein_embeds`
        top_k: int
            - Number of top proteins to retrieve for the scoring
            - Set this to None if you want the whole dataframe
    Output:
        pd.DataFrame with top retrieval hits, columns are `uniprot_id`, `name`, and `sim_score`
    """
    assert model_out is not None or query_embeddings is not None
    if model_out is None:
        assert query_embeddings is not None
    else:
        assert query_embeddings is None
        query_embeddings = (
            model_out["contrastive_out"]["positive"]["text"][0, :]
            .unsqueeze(0)
            .detach()
            .clone()
        )
    query_embeddings = F.normalize(query_embeddings, dim=-1).float()

    # Now compute similarities to all proteins:
    sims = torch.matmul(
        query_embeddings.cuda(),
        F.normalize(protein_embeds, dim=-1).transpose(0, 1).cuda(),
    ).squeeze()

    # Rank distances:
    sort_inds = torch.argsort(sims, descending=True)
    if top_k is not None:
        top20 = sort_inds[:top_k].detach().clone().cpu().tolist()
        sim_sub = sims[sort_inds[:top_k]].detach().clone().cpu().tolist()
    else:
        top20 = sort_inds.detach().clone().cpu().tolist()
        sim_sub = sims[sort_inds].detach().clone().cpu().tolist()

    ids = protein_ids["protein_id"].iloc[top20]
    names = protein_ids["name"].iloc[top20]

    # Construct dataframe and return:
    df = pd.DataFrame({"uniprot_id": ids, "name": names, "sim_score": sim_sub})

    return df

def perturb_by_words(
    sentence: str, generator: np.random.Generator, perturbation_pct: float = 0.1
) -> str:
    """Generate perturbed description."""
    wordlist = sentence.split()
    words_to_keep = set(
        generator.choice(
            np.arange(len(wordlist)),
            size=math.floor((1 - perturbation_pct) * len(wordlist)),
            replace=False,
        )
    )

    new_wordlist = [w for i, w in enumerate(wordlist) if (i in words_to_keep)]

    return " ".join(new_wordlist)


def desc_perturbation(
    desc: str,
    query_func: Callable,
    num_perturbations: int = 10,
    perturbation_pct: float = 0.1,
    seed: Optional[float] = None,
) -> Dict:
    """Run many perturbations for a single original description. For confidence intervals on retrieval."""
    generator = np.random.default_rng(seed)

    all_perturbations_dict = {}
    for i in trange(num_perturbations):
        # Perturb desc:
        new_desc = perturb_by_words(
            desc, generator=generator, perturbation_pct=perturbation_pct
        )
        out_dict = query_func(new_desc)
        all_perturbations_dict[f"perturb_{i}"] = out_dict

    return all_perturbations_dict