import torch, os, pickle
import torch.nn.functional as F
from procyon.model.model_unified import UnifiedProCyon

from procyon.data.inference_utils import ProCyonQAInference

from tqdm import trange, tqdm
from typing import List
import numpy as np
from procyon.data.data_utils import DATA_DIR, get_text_sequences_compositions
from procyon.data.it_collator import construct_task_id
from procyon.data.instruct_tune.instruct_constructor import get_prompt, get_prompt_open_def
from procyon.data.data_utils import convert_batch_protein

from procyon.data.constants import QA_SUBSETS, CAPTION_SUBSETS

from esm.data import Alphabet, BatchConverter

from Bio import SeqIO

HOME_DIR = os.getenv("HOME_DIR")

PROTEIN_SEQS = [str(seq.seq) for seq in SeqIO.parse(os.path.join(DATA_DIR, f"integrated_data/v1/protein/protein_sequences.fa"), "fasta")]
DOMAIN_SEQS = [str(seq.seq) for seq in SeqIO.parse(os.path.join(DATA_DIR, f"integrated_data/v1/domain/domain_sequences.fa"), "fasta")]
PEPTIDE_SEQS = [str(seq.seq) for seq in SeqIO.parse(os.path.join(DATA_DIR, f"integrated_data/v1/peptide/peptide_sequences.fa"), "fasta")]

import json
import pandas as pd
from typing import List
from procyon.data.data_utils import DATA_DIR, get_text_sequences_compositions
from procyon.data.it_collator import construct_task_id
from procyon.data.instruct_tune.instruct_constructor import get_prompt, get_prompt_open_def

from procyon.data.constants import RETRIEVAL_SUBSETS

UNIPROT_IDS = pd.read_pickle(os.path.join(DATA_DIR, "integrated_data/v1/protein/", "protein_info_filtered.pkl"))[["index", "protein_id", "name"]]

def uniprot_id_to_index(uniprot_id):
    assert (UNIPROT_IDS["protein_id"] == uniprot_id).sum() == 1, "ID {} not found in internal database".format(uniprot_id)
    i = UNIPROT_IDS["index"].loc[UNIPROT_IDS["protein_id"] == uniprot_id].item()
    return i

def index_to_uniprot_id(i):
    uniprot_id = UNIPROT_IDS["protein_id"].loc[UNIPROT_IDS["index"] == i].item()
    return uniprot_id 

# Protein-peptide QA:
from typing import Union, List

def main(out_path):

    CKPT_NAME = "/n/holylfs06/LABS/mzitnik_lab/Lab/PLM/REPRO/ProCyon-Bind"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = UnifiedProCyon.from_pretrained(checkpoint_dir=CKPT_NAME, load_plm_directly = True, protein_pooling_correction_option = True)
    model.config.filter_negatives_by_id_contrastive = False # Need to override bc it can cause problems later - not necessary for inference
    model.to(device)
    model.eval()

    data_args = torch.load(os.path.join(CKPT_NAME, "data_args.pt"))
    train_args = torch.load(os.path.join(CKPT_NAME, "training_args.pt"))

    model_args = torch.load(os.path.join(CKPT_NAME, "model_args.pt"))
    aaseq_tokenizer = Alphabet.from_architecture(model_args.protein_tokenizer_name)
    BATCH_CONVERTER = BatchConverter(
                    aaseq_tokenizer,
                    truncation_seq_length = model_args.max_aaseq_len if (model_args.long_protein_strategy == "truncate") else None
                )
    # To tokenize sequences:
    def convert_batch_sequences(seqs):
        unique_indices = list(range(len(seqs)))
        batch_toks = convert_batch_protein(
            ids = unique_indices,
            is_protein_tokenized = False,
            batch_converter = BATCH_CONVERTER,
            protein_sequences = seqs,
            protein_tokens = None,
            protein_tokenizer = aaseq_tokenizer,
            max_protein_len = model_args.max_aaseq_len,
        )
        return batch_toks

    def get_aaseq_library(seqs, batch_size = 2):
        batch_toks = convert_batch_sequences(seqs).to(device)
        # Break into batch_size to avoid OOM's:
        batch_idx = torch.arange(0, len(seqs), step=batch_size)
        batch_gather = []
        for i in trange(batch_idx.shape[0]):
            if i == (batch_idx.shape[0] - 1):
                sub_batch_toks = batch_toks[batch_idx[i]:,:]
            else:
                sub_batch_toks = batch_toks[batch_idx[i]:batch_idx[i+1],:]
            # Call forward_sequences:
            out = model.forward_sequences(seq_input = sub_batch_toks, get_soft_tokens = False)
            batch_gather.append(out["shared"])
        return torch.cat(batch_gather, dim=0)

    def create_prot_pep_input_qa(
            input_aaseq1_seq: str,
            input_aaseq2_seq: str,
            task_definition: str = None,
            icl_example_number: int = 1,
            positive_icl_seqs1: List[int] = None,
            positive_icl_seqs2: List[str] = None,
            negative_icl_seqs1: List[int] = None,
            negative_icl_seqs2: List[str] = None,
        ):

        task_type = "qa"
        instruction_source_dataset = "peptide"

        icl_binding_seqs1 = None
        icl_binding_seqs2 = None

        example_descriptions = []
        
        # Load instructions from instruction constructor
        task_id = construct_task_id(aaseq_type = "peptide", 
                                    text_type = instruction_source_dataset, 
                                    relation_type = "all", 
                                    task_type = 'qa')
        fpath = os.path.join(HOME_DIR, f"procyon/data/instruct_tune/tasks/{task_id}.json")
        
        with open(fpath, "r") as j:
            task_json = json.loads(j.read())

        get_prompt_fn = get_prompt if (task_definition is None) else get_prompt_open_def

        instruction, _, _, example_text_ids, example_aaseq_ids = get_prompt(task = task_json, 
                num_examples = icl_example_number,
                is_special_definition = False,
                is_ppi = True,
                aaseq_type = "peptide",
            )
        
        icl_seqs = None
        if icl_example_number > 0:
            icl_seqs = [PEPTIDE_SEQS[i] for i in example_aaseq_ids]

        L = example_aaseq_ids
        L += [0, 0]
        if icl_seqs is not None:
            all_seqs = icl_seqs + [input_aaseq1_seq, input_aaseq2_seq]
        else:
            all_seqs = [input_aaseq1, input_aaseq2]

        unique_aaseq_indices = torch.LongTensor(L).to(device)
        input_seq_tokens = convert_batch_sequences(all_seqs).to(device)

        instruction = instruction.replace("[CONTEXT]", "")

        model_input = {
                "data": {
                    "seq": input_seq_tokens,
                    "seq_idx": unique_aaseq_indices,
                    "text": [],
                    "drug": None, # Optional
                },
                "input": {
                    "seq": torch.arange((len(all_seqs))).unsqueeze(0).tolist(),
                    "text": [[]], # List not tensor
                    "drug": None,
                },
                "target": { # This is only used for training
                    "seq": None,
                    "text": None,
                    "drug": None,
                },
                "instructions": [instruction],
            }

        # return model_input
        return model_input

    ace2_peptides = pd.read_csv(os.path.join(DATA_DIR, "experimental_data/ProteinPeptideBinding/ace2_peptides.csv"))

    ace2_seqs = ace2_peptides["seq"].tolist()
    ace2_lib = get_aaseq_library(ace2_seqs)

    qa_model = ProCyonQAInference(model)

    ACE2_SEQ = PROTEIN_SEQS[uniprot_id_to_index("Q9BYF1")]

    preds_positive = []
    for i, s in enumerate(tqdm(ace2_peptides.seq.loc[ace2_peptides['binder'] == 1])):
        prot_peptide_input_qa = create_prot_pep_input_qa(
            input_aaseq1_seq = ACE2_SEQ,
            input_aaseq2_seq = s,
            task_definition = None,
        )
        model_out = qa_model(prot_peptide_input_qa, aaseq_type = 'peptide')
        pred_v = [model_out['pred'][0,qa_model.yes_token].item(), model_out['pred'][0,qa_model.no_token].item()]
        preds_positive.append(pred_v)

    preds_negative = []
    neg_seqs = ace2_peptides.loc[ace2_peptides["binder"] == 0,:].seq.tolist()

    for i, s in enumerate(tqdm(neg_seqs)):
        prot_peptide_input_qa = create_prot_pep_input_qa(
            input_aaseq1_seq = ACE2_SEQ,
            input_aaseq2_seq = s,
            task_definition = None,
        )
        model_out = qa_model(prot_peptide_input_qa, aaseq_type = 'peptide')
        pred_v = [model_out['pred'][0,qa_model.yes_token].item(), model_out['pred'][0,qa_model.no_token].item()]
        preds_negative.append(pred_v)

    preds_np = np.array(preds_positive + preds_negative)
    labels_np = np.array(np.ones(len(preds_positive)).tolist() + np.zeros(len(preds_negative)).tolist())

    with open(out_path, "wb") as outfile:
        pickle.dump((preds_np, labels_np), outfile)


if __name__ == '__main__':
    main(out_path = "./ace2_preds.pickle")