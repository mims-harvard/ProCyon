import os, sys, argparse, math
from dataclasses import dataclass
import numpy as np
from datetime import datetime
from time import sleep
import ipdb
from tqdm import trange

import pandas as pd

import torch

from procyon.training.training_args_IT import TrainArgs, DataArgs, ModelArgs, get_hparams, postprocess_args
from procyon.model.model_unified import UnifiedProCyon
from procyon.training.trainIT import TxPLMTrainerIT
from procyon.training.train_utils import get_root_logger, set_seed, get_IT_datasets, get_data_collators_IT, get_data_collators_IT_new, get_all_datasets, get_device
from procyon.training.train_utils import (
    get_root_logger,
    set_seed,
)
from procyon.data.data_utils import DATA_DIR
from procyon.evaluate.general_eval import prepare_inputs

from procyon.data.inference_utils import create_caption_input_simple, uniprot_id_to_index

from torch.utils.data import DataLoader

uniprot_ids = pd.read_pickle(os.path.join(DATA_DIR, "integrated_data/v1/protein/", "protein_info_filtered.pkl"))[["index", "protein_id"]]

def select_print_batch(rdict, i):
    return_dict_sub = {
        "text": rdict["text"][i],
        "input_instructions": rdict["input_instructions"][i],
        "ground_truth_text": rdict["ground_truth_text"][i],
        "uniprot_id": uniprot_ids["protein_id"].iloc[rdict["seq_references"][i]] 
    }
    return return_dict_sub

def print_generate_return(rdict):

    # Get unique values:
    batch_unique_proteins = np.unique(rdict["seq_references"], return_index = True)[1]

    for i in batch_unique_proteins:
        sub_info = select_print_batch(rdict, i)
        print("-" * 50)
        print('UniProt ID: {} ; Internal ID: {}'.format(sub_info["uniprot_id"], rdict["seq_references"][i]))
        print("\nInput instructions")
        print(sub_info["input_instructions"])
        print("\nGT text:")
        print(sub_info["ground_truth_text"])
        print("\nOutput text:")
        for i, t in enumerate(sub_info["text"]):
            print(f"TEXT {i} --------------------------------------------")
            print(t.split("</s>")[0])

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert "drug" not in args.prompt_dataset, "DrugBank not supported in this script"

    # Get data, train, model args:
    data_args, model_args, train_args = UnifiedProCyon.get_checkpoint_configs(resume_from_checkpoint = args.ckpt)
    # Load model:
    model, _ = UnifiedProCyon.from_pretrained(checkpoint_dir=args.ckpt)

    model.to(device)
    model.eval()

    set_seed(1234)

    # Load uniprot file:
    target_uniprot_ids = pd.read_csv(args.uniprot_id_file)

    # Calculate chunks:
    if (args.num_chunks is not None) and (args.chunk_idx is not None):
        fs = target_uniprot_ids.shape[0]
        chunk_indices = np.arange(0, fs, math.ceil(fs / args.num_chunks))
        chunk_start = chunk_indices[args.chunk_idx]

        if args.chunk_idx >= (math.ceil(fs / args.num_chunks) - 1):
            chunk_end = fs

        else:
            chunk_end = chunk_indices[args.chunk_idx + 1]

        # Split by this job's chunk size:
        target_uniprot_ids = target_uniprot_ids.iloc[chunk_start:chunk_end,:]

    uniprot_id_list = target_uniprot_ids["uniprot_id"].tolist()

    results_dict = {}
    results_dict["uniprot_id"] = []
    results_dict.update({f"response{i}":[] for i in range(args.beam_size // 2)})

    save_frequency = 5

    for i in trange(target_uniprot_ids.shape[0]):

        uniprot_id = target_uniprot_ids["uniprot_id"].iloc[i]

        results_dict['uniprot_id'].append(uniprot_id)

        model_input = create_caption_input_simple(
            input_aaseq_ids = [uniprot_id_to_index(uniprot_id)],
            data_args = data_args,
            input_description = None,
            drug_inputs = None,
            task_definition = None,
            instruction_source_dataset = args.prompt_dataset,
            instruction_source_relation = args.prompt_relation,
            aaseq_type = "protein",
            task_type = "caption",
            icl_example_number = 1,
            device = device,
        )
        
        # HARDCODED: Using diverse beam search
        with torch.no_grad():
            out_tokens, log_probs, output_logits, out_text = model.generate(
                inputs = model_input, 
                aaseq_type = "protein", 
                max_len = args.max_len, 
                method = "beam",
                return_all_internals = False,
                beam_size = args.beam_size,
                beam_group_size = 2,
                diversity_penalty = args.diversity_penalty,
            )

        # Get every other beam output:
        for j, t in enumerate(out_text[0]):
            if j % 2 == 1: # bc beam size is 2
                continue

            if "llama-3" in model_args.text_encoder_fname:
                split_str = "<|end_of_text|>"
            else:
                split_str = "</s>"

            sub_text = t.split(split_str)[0]
            results_dict[f'response{j // 2}'].append(sub_text)
        
        if (i % save_frequency == 0) and (i > 0):
            pd.DataFrame(results_dict).to_pickle(args.save_path)

    # Save results_dict to target file
    df = pd.DataFrame(results_dict)

    print(df)

    if args.save_path is not None:
        df.to_csv(args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required = True, help = "Path to ProCyon checkpoint")
    parser.add_argument("--save_path", required = False, default = None, type=str, help = "CSV file name to save captions to")

    parser.add_argument("--chunk_idx", required = False, default = None, type = int, help = "Used for chunking the dataframe into separate pieces")
    parser.add_argument("--num_chunks", required = False, default = None, type = int, help = "Used for chunking the dataframe into separate pieces")

    parser.add_argument("--max_len", default = 200, required = False, type=int, help = "Maximum length of generated text")
    parser.add_argument("--beam_size", default = 10, required = False, type = int, help = "Beam size if using beam search")
    parser.add_argument("--diversity_penalty", default = 0.8, required = False, type = float, help = "Diversity penalty for diverse beam search")

    parser.add_argument("--uniprot_id_file",
        required = True, 
        type=str, 
        help="CSV with uniprot id's to process. UniProt IDs must be contained in a column titled 'uniprot_id'"
    )

    parser.add_argument("--prompt_dataset", default = "uniprot", required = False, 
        help = "Dataset prompt to use for the model. Generates in the style of this dataset. See dataset instructions for more information.")
    parser.add_argument("--prompt_relation", default = "all", required = False,
        help = "Relation for the dataset. Some datasets have more than one, such as GO process, component, and function.")

    args = parser.parse_args()

    # Print metadata:
    print(("-" * 50) + "Metadata" + ("-" * 50))
    print("\tCheckpoint: {}".format(args.ckpt))
    print("\tmax_len: {}".format(args.max_len))
    print(f"\tGeneration method: beam search with beam size = {args.beam_size}, beam_group_size = 2, diversity_penalty = {args.diversity_penalty}")
    print(("-" * 50) + ("-" * 50))

    main(args)