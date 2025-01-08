#! /usr/bin/env python
import csv
import os

from typing import Dict, List

import pandas as pd

from Bio import SeqIO
from transformers import HfArgumentParser

from procyon.data.data_utils import DATA_DIR
from procyon.evaluate.framework.utils import load_eval_data_loaders
from procyon.training.training_args_IT import (
    TrainArgs,
    DataArgs,
    ModelArgs,
    postprocess_args,
)

# Prepare mappings from internal ProCyon-Instruct IDs to various
# protein identifiers (Uniprot ID, gene name) and Drugbank IDs
integrated_data_path = os.path.join(
    DATA_DIR,
    "integrated_data",
    "v1",
)
uniprot_ids = pd.read_pickle(
    os.path.join(
        integrated_data_path,
        "protein",
        "protein_info_filtered.pkl",
    )
)[["index", "protein_id", "name"]]

uniprot_annotations = pd.read_table(
    os.path.join(
        DATA_DIR,
        "experimental_data",
        "llm_judge_eval",
        "selected_caption_samples",
        "uniprotkb_proteome_UP000005640_2024_06_18.tsv.gz",
    )
)
# ProCyon-Instruct ID -> Uniprot ID
id_to_upid = {row[0]:row[1].split()[0] for row in uniprot_ids[["index", "protein_id"]].itertuples(index=False)}
# Uniprot ID -> HGNC gene names (possibly multiple)
up_gene_name_map = {row[0]:row[1] for row in uniprot_annotations[["Entry", "Gene Names"]].itertuples(index=False)}
# Uniprot ID -> long form protein name
up_protein_name_map = {row[0]:row[1] for row in uniprot_annotations[["Entry", "Protein names"]].itertuples(index=False)}

drugbank_ids = pd.read_pickle(
    os.path.join(
        integrated_data_path,
        "drugbank",
        "drugbank_info_filtered.pkl",
    )
)[["index", "drugbank_id", "drug_name"]]
# ProCyon-Instruct ID - > DrugBank drug name
db_name_map = {row[0]:row[1] for row in drugbank_ids[["index", "drug_name"]].itertuples(index=False)}


def construct_text_input(
    model_input: Dict,
    encoding_name: str="gene_name",
    drugbank: bool=False,
) -> str:
    """Generate phenotype generation prompt from a ProCyon model input

    encoding_name specifies the various methods for representing a protein in text.
    Options are:
    - 'aaseq' - use full amino acid sequence
    - 'gene_name' - use HGNC gene name as provided by UniProt. In the case of multiple names, provide all of them.
    - 'protein_name' - use long-form protein name provided by UniProt
    - 'uniprot' - use UniProt ID

    drugbank specifies whether or not the inputs contain <|drug|> tokens that need to be replaced with
    the corresponding drug's name from DrugBank.
    """

    # 1. Replace <|protein|> with gene names based on above parameters
    protein_strs = []
    if encoding_name == 'aaseq':
        fasta_path = os.path.join(integrated_data_path, "protein", "protein_sequences.fa")
        fasta_sequences = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
        protein_strs = [str(fasta_sequences[id_to_upid[i]].seq) for i in model_input["data"]["seq"].tolist()]
    elif encoding_name == 'gene_name':
        protein_strs = [up_gene_name_map[id_to_upid[i]] for i in model_input["data"]["seq"].tolist()]
    elif encoding_name == 'protein_name':
        protein_strs = [up_protein_name_map[id_to_upid[i]] for i in model_input["data"]["seq"].tolist()]
    elif encoding_name == "uniprot":
        protein_strs = [id_to_upid[i] for i in model_input["data"]["seq"].tolist()]

    if drugbank:
        drug_strs = [[db_name_map[model_input["data"]["drug"][i].item()] for i in x] for x in model_input["input"]["drug"]]

    # 2. Get the <|protein|> out and replace with unique descriptions
    all_new_instructions = []
    gt_texts = []
    for i in range(len(model_input['instructions'])):
        if encoding_name is not None:
            L = [protein_strs[pi] for pi in model_input["input"]["seq"][i]]

            new_instruction = ""
            split_instruction = model_input['instructions'][i].split("<|protein|>")

            for j, substr in enumerate(split_instruction[:-1]):
                new_instruction += substr + L[j]
            else:
                new_instruction += split_instruction[-1]
        else:
            new_instruction = model_input['instructions'][i]

        # Same thing for [EXT]
        D = [model_input["data"]["text"][ti] for ti in model_input["input"]["text"][i]]
        gt_texts.append(D[-1])

        if " [EXT]" == new_instruction[-6:]:
            new_instruction = new_instruction[:-6]
        elif " yes" == new_instruction[-4:] :
            new_instruction = new_instruction[:-4]
        elif " no" == new_instruction[-3:] :
            new_instruction = new_instruction[:-3]

        new_instruction_ext = ""
        split_ext_instruction = new_instruction.split("[EXT]")
        for j, substr in enumerate(split_ext_instruction[:-1]):
            new_instruction_ext += substr + D[j]
        else:
            new_instruction_ext += split_ext_instruction[-1]

        if drugbank:
            new_instruction_final = ""
            split_drug_instruction = new_instruction_ext.split("<|drug|>")
            for j, substr in enumerate(split_drug_instruction[:-1]):
                new_instruction_final += substr + drug_strs[i][j]
            else:
                new_instruction_final += split_drug_instruction[-1]
        else:
            new_instruction_final = new_instruction_ext
        new_instruction_final = new_instruction_final.replace("[ANSWER]", "") # Remove [ANSWER]
        all_new_instructions.append(new_instruction_final)

    # Return is list of strings with all information loaded into them
    return all_new_instructions, gt_texts

def write_prompts_to_csv(
    protein_ids: List[int],
    prompts: List[str],
    file_name: str,
):
    file_is_empty = not os.path.isfile(file_name) or os.stat(file_name).st_size == 0
    with open(file_name, 'a', newline='') as f:
        writer = csv.writer(f)
        if file_is_empty:
            writer.writerow(['Protein ID', 'Prompt'])
        for protein_id, prompt in zip(protein_ids, prompts):
            writer.writerow([protein_id, prompt])

def write_qa_prompts_to_csv(
    protein_ids: List[int],
    text_ids: List[int],
    prompts: List[str],
    ground_truth: List[str],
    file_name: str,
):
    file_is_empty = not os.path.isfile(file_name) or os.stat(file_name).st_size == 0
    with open(file_name, 'a', newline='') as f:
        writer = csv.writer(f)
        if file_is_empty:
            writer.writerow(['Protein ID', 'Text ID', 'Prompt', 'Expected'])
        for protein_id, text_id, prompt, gt in zip(protein_ids, text_ids, prompts, ground_truth):
            writer.writerow([protein_id, text_id, prompt, gt])

def main(
    train_args: TrainArgs,
    data_args: DataArgs,
    model_args: ModelArgs,
    encoding: str,
):
    data_args.val_split_type = 'zero_shot'
    data_args.use_caption = True
    data_args.use_entity_compositions = True

    model_args.use_aaseq_embeddings = True

    data_loaders = load_eval_data_loaders(
        data_args,
        model_args,
        train_args.caption_batch_size,
        train_args.num_workers,
    )

    for task, task_loaders in data_loaders.items():
        for dataset_key, data_loader in task_loaders.items():

            if "drugbank" in dataset_key:
                drugbank = False
            else:
                drugbank = False

            # Where the index of the actual text query is depends on whether or not
            # this is a dataset with context augmentation.
            no_context_aug = data_loader.collate_fn._get_input_contexts([0], [0], [[0]], [[0]]) is None
            if no_context_aug:
                query_text_idx = -1
            else:
                query_text_idx = -2

            all_texts = []
            all_gt = []
            all_protein_ids = []
            all_text_ids = []
            for model_input in data_loader:
                texts, _ = construct_text_input(
                    model_input,
                    encoding_name=encoding,
                    drugbank=drugbank,
                )

                protein_ids = [id_to_upid[x[-1]] for x in model_input["reference_indices"]["input"]["seq"]]
                text_ids = [x[query_text_idx] for x in model_input["reference_indices"]["input"]["text"]]

                all_texts.extend(texts)
                all_protein_ids.extend(protein_ids)
                all_text_ids.extend(text_ids)
                all_gt.extend(model_input["target"]["text"])

            if task == "qa":
                write_qa_prompts_to_csv(
                    all_protein_ids,
                    all_text_ids,
                    all_texts,
                    all_gt,
                    f"{dataset_key}.{task}.{encoding}.gpt_prompts.csv",
                )
            else:
                write_prompts_to_csv(
                    all_protein_ids,
                    all_texts,
                    f"{dataset_key}.{task}.{encoding}.gpt_prompts.csv",
                )


encoding_choices = ["gene_name", "protein_name", "aaseq", "uniprot", None]
if __name__ == '__main__':
    parser = HfArgumentParser((TrainArgs, DataArgs, ModelArgs))
    parser.add_argument(
        "--encoding",
        action="store",
        default="gene_name",
        choices=encoding_choices,
    )
    train_args, data_args, model_args, oth_args = parser.parse_args_into_dataclasses()

    if train_args.from_yaml is not None:
        train_args, data_args, model_args = parser.parse_yaml_file(train_args.from_yaml)
    if train_args.from_json is not None:
        train_args, data_args, model_args = parser.parse_json_file(train_args.from_json)

    train_args, data_args, model_args = postprocess_args(train_args, data_args, model_args)

    main(train_args, data_args, model_args, oth_args.encoding)
