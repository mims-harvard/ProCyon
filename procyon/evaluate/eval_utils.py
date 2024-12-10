import csv
import os
import pickle

from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Bio import SeqIO
from tqdm import tqdm
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    top_k_accuracy_score,
)

from procyon.data.data_utils import DATA_DIR
from procyon.evaluate.metrics import fmax_score

def precision_recall_topk(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    k: int,
    return_all_vals: bool = False,
):
    """
    Calculate precision and recall for top-k accuracy in multi-label classification.
    Parameters:
        y_true (array-like): True binary labels (shape: [n_samples, n_classes]).
        y_pred (array-like): Predicted probabilities for each class (shape: [n_samples, n_classes]).
        k (int): The value of k for top-k accuracy.
    Returns:
        precision (float): Precision for top-k accuracy.
        recall (float): Recall for top-k accuracy.
    """
    # Some of the following code doesn't work with numpy arrays, so just
    # standardize to torch.Tensor
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)

    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred)
    else:
        y_pred = y_pred.clone()

    # Make sure labels are actually binary.
    non_nan_labels = y_true[~torch.isnan(y_true)]
    if not np.isin(non_nan_labels, [0, 1]).all():
        raise ValueError(f"expected labels to be 0 or 1, got: {y_true}")

    num_samples, num_classes = y_true.shape
    if k > num_classes:
        print(f"Provided value of k greater than number of items ({k} > {num_classes}), "
              "padding with neg inf.")
        pad_len = k - num_classes
        y_pred = torch.cat((
            y_pred,
            torch.full((num_samples, pad_len), -float("inf"))
        ), dim=1)

    # Convert any NaN predictions to neg inf for better behavior during sorting.
    # Positions where label is NaN corresponds to pairs we want to ignore,
    # so we also set those predictions to NaN.
    y_pred[torch.isnan(y_true) | torch.isnan(y_pred)] = -float("inf")
    topk_vals, topk_idxs = y_pred.topk(k=k)

    precisions = []
    recalls = []
    fmaxes = []

    any_nan = False
    for i in range(num_samples):
        true_labels = y_true[i]
        preds = y_pred[i]

        sorted_indices = topk_idxs[i]
        topk_preds = topk_vals[i]
        is_neginf = torch.isneginf(topk_preds)
        if is_neginf.any().item():
            any_nan = True
            first_nan = torch.nonzero(is_neginf, as_tuple=True)[0][0].item()
            sorted_indices = sorted_indices[:first_nan]

        true_labels_k = true_labels[sorted_indices]

        # Calculate true positives, relevant items, and retrieved items
        query_true_pos = true_labels_k.nansum().item()
        query_relevant = true_labels.nansum().item()
        query_retrieved = len(sorted_indices)

        if query_retrieved > 0:
            precisions.append(query_true_pos / query_retrieved)
        else:
            precisions.append(0.)

        if query_relevant > 0:
            recalls.append(query_true_pos / query_relevant)
        else:
            recalls.append(0.)

        want_idxs = ~true_labels.isnan() & ~preds.isnan()
        want_labels = true_labels[want_idxs]
        want_preds = preds[want_idxs]

        fmaxes.append(fmax_score(want_labels, want_preds)[0])

    if any_nan:
        print("NaNs found when calculating top-k precision/recall. Results truncated to number of non-NaN items "
              "(this may be expected for some models, e.g. BLAST)")

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_fmax = np.mean(fmaxes)
    if return_all_vals:
        return avg_precision, avg_recall, avg_fmax, precisions, recalls, fmaxes
    else:
        return avg_precision, avg_recall

def get_goid_text_mapping(text_file):
    """generate the mapping dict between go-id and text

    TODO: Replace with get_text_sequences
    Args:
        text_file (str/pathlib.Path): mapping file (*.pkl)
    Returns:
        dict: mapping dict
    """
    goid2text = {}
    with open(text_file, 'rb') as fin:
        content = pickle.load(fin)
        for i in range(len(content)):
            goid2text[content.go_id[i]] = content.go_name[i] + '; ' + content.go_def[i]
    return goid2text
def get_seqid_seq_mapping(protein_file):
    """
    generate the mapping dict between sequence-id and sequence
    Args:
        protein_file (str/pathlib.Path): mapping file (*.fa)
    Returns:
        dict: mapping dict
    """
    id2seq = {}
    for record in SeqIO.parse(protein_file, 'fasta'):
        id2seq[record.id] = record.seq.__str__()
    return id2seq

def get_text_aaseq_mapping(relation_file, max_num = None):
    """
    TODO: Do we need?

    generate the mapping dict between protein-id to multiple go-ids
    Args:
        relation_file (str/pathlib.Path): mapping file (*.csv)
    Returns:
        _type_: dict(std: [])
    """
    test_csv = relation_file
    csv_mode = 1
    txtid_to_seqid = {}
    all_seq_ids, all_go_ids = [], []

    max_num = 1e9 if max_num is None else max_num

    with open(test_csv, 'r') as fin:
        reader = csv.reader(fin)
        header = next(reader) # skip the first line
        print(header)
        seq_line_id = header.index('seq_id')
        go_line_id = header.index('text_id')
        for i, line in enumerate(reader):
            if i > max_num:
                break
            seq_id = line[seq_line_id]
            go_id = int(line[go_line_id])
            if go_id not in txtid_to_seqid:
                txtid_to_seqid[go_id] = []
            txtid_to_seqid[go_id].append(int(seq_id))
            all_seq_ids.append(int(seq_id))
            all_go_ids.append(go_id)

    # Make unique mappings:
    unique_seqs, seq_inverse = np.unique(all_seq_ids, return_inverse = True)
    unique_texts, text_inverse = np.unique(all_go_ids, return_inverse = True)

    text_to_unique_map = {goid: i for i, goid in enumerate(unique_texts)}
    seq_to_unique_map = {seqid: i for i, seqid in enumerate(unique_seqs)}

    # text_to_unique_map = {goid: uni for goid, uni in zip(all_go_ids, text_inverse)}
    # seq_to_unique_map = {proid: uni for proid, uni in zip(all_seq_ids, seq_inverse)}

    return txtid_to_seqid, (unique_texts, text_inverse, text_to_unique_map), (unique_seqs, seq_inverse, seq_to_unique_map)

def get_text_aaseq_mapping_PD(dataframe, max_num=None):
    """
    Generate the mapping dict between protein-id to multiple go-ids

    Args:
        dataframe (pd.DataFrame): Input DataFrame with columns 'seq_id' and 'text_id'
        max_num (int, optional): Maximum number of rows to process

    Returns:
        tuple: (
            dict: {text_id: [seq_ids]},
            tuple: (unique_texts, text_inverse, text_to_unique_map),
            tuple: (unique_seqs, seq_inverse, seq_to_unique_map)
        )
    """
    txtid_to_seqid = {}
    all_seq_ids, all_go_ids = [], []

    max_num = 1e9 if max_num is None else max_num

    seq_line_id = 'seq_id'
    go_line_id = 'text_id'
    for i, row in dataframe.iterrows():
        if i > max_num:
            break
        seq_id = row[seq_line_id]
        go_id = int(row[go_line_id])
        if go_id not in txtid_to_seqid:
            txtid_to_seqid[go_id] = []
        txtid_to_seqid[go_id].append(int(seq_id))
        all_seq_ids.append(int(seq_id))
        all_go_ids.append(go_id)

    # Make unique mappings:
    unique_seqs, seq_inverse = np.unique(all_seq_ids, return_inverse=True)
    unique_texts, text_inverse = np.unique(all_go_ids, return_inverse=True)

    text_to_unique_map = {goid: i for i, goid in enumerate(unique_texts)}
    seq_to_unique_map = {seqid: i for i, seqid in enumerate(unique_seqs)}

    return txtid_to_seqid, (unique_texts, text_inverse, text_to_unique_map), (unique_seqs, seq_inverse, seq_to_unique_map)

def get_protein_feature(model: nn.Module, seq: str):
    """extract protein feature from model (CLIP based retrieval model)
    Args:
        model (nn.Module): CLIP model
        seq (str): input protein sequence data
    Returns:
        torch.tensor: protein feature
    """
    if hasattr(model, 'device'):
        device = model.device
    else:
        device = torch.device('cuda')
    protein = torch.from_numpy(one_hot(seq)).to(device).unsqueeze(0).float()
    text = 'Putative transcription factor'
    text_ids = tokenizer([text])[0].unsqueeze(0).to(device)
    with torch.no_grad():
        prot_feat, _, _ = model(protein, text_ids)
    return prot_feat
def get_text_feature(model: nn.Module, text: str):
    """extract text feature from model (CLIP based retrieval model)
    Args:
        model (nn.Module): CLIP model
        text (str): input data
    Returns:
        torch.tensor: text feature
    """
    if hasattr(model, 'device'):
        device = model.device
    else:
        device = torch.device('cuda')
    protein = 'MAF'
    # encode protein using one-hot encoding
    protein = torch.from_numpy(one_hot(protein)).to(device).unsqueeze(0).float()
    text_ids = tokenizer([text])[0].unsqueeze(0).to(device)
    with torch.no_grad():
        _, text_feat, _ = model(protein, text_ids)
    return text_feat
def protein_go_retrieval(
    model,
    protein_feature_func,
    text_feature_func,
    protein_file,
    text_file,
    relation_file,
    max_sep_topk = 25,
    goid_2_text_func = get_goid_text_mapping,
    seqid_2_seq_func = get_seqid_seq_mapping,
    go_2_protein_func = get_text_aaseq_mapping
):
    """Protein and go text description retrieval
    Args:
        model (nn.Module): Eval model
        protein_feature_func (function): function that gets a protein feature of a protein sequence (string type)
            Example:
                protein_feature = protein_feature_func(model, protein)
        text_feature_func (function): function that gets a text feature of a text
            Example:
                text_feature = text_feature_func(model, text)
        protein_file (str/pathlib.Path): path of *.fa file
        text_file (str/pathlib.Path): path of *.pkl file which contains go descriptions
        relation_file (str/pathlib.Path): path of *.csv file which contains the relationship between protein id and go id
        max_sep_topk (int, optional): hyper-parameter of maximum seperation algorithm for multi-modal retrieval. Defaults to 30.
            this value is dataset specific, implying that a protein is related to at most how many texts (go-ids)
        goid_2_text_func (function): function that recieve text_file return the go-id to text mapping
            Example:
                goid2text = goid_2_text_func(text_file)
        seqid_2_seq_func (function): function that recieve protein_file return the seq-id to protein-sequence (str) mapping
            Example:
                seqid2seq = get_seqid_seq_mapping(protein_file)
        go_2_protein_func (function): function that recieve relation_file (sub-dataset we want evaluate) return the go to protein id mapping.
            Example:
                id2seq = protein_2_go_func(relation_file)
    """
    # loading data
    goid2text:dict = goid_2_text_func(text_file)
    id2seq:dict = seqid_2_seq_func(protein_file)
    txtid_to_seqid:dict = go_2_protein_func(relation_file)
    # extract text feature
    prot_feat_list = []
    prot_keys = list(id2seq.keys())
    prot_keys.sort()
    text = 'Putative transcription factor'
    text_ids = tokenizer([text])[0].unsqueeze(0).cuda()
    bar = tqdm(prot_keys)
    for tk in bar:
        prot = id2seq[tk]
        prot_feat = protein_feature_func(model, prot)
        prot_feat_list.append(prot_feat)

    prot_feat_tensor = torch.concat(prot_feat_list)
    # extract protein feature and evaluation
    preds = []
    labels = []
    n_prot = len(prot_keys)
    st_bar = tqdm(txtid_to_seqid)
    for item in st_bar:
        # seq
        go_id = item
        seq_id_list = txtid_to_seqid[go_id]
        label = None
        label_index = None
        for seqid in seq_id_list:
            if seqid in prot_keys:
                seqid_index = prot_keys.index(seqid)
                seqid_tensor = torch.tensor(seqid_index)
                label_index = seqid_index
                if label is None:
                    label = F.one_hot(seqid_tensor, num_classes=n_prot)
                else:
                    label += F.one_hot(seqid_tensor, num_classes=n_prot)
        if label is None:
            continue


        text = goid2text[go_id]
        text_feat = text_feature_func(model, text)
        dot_similarity_text = text_feat @ prot_feat_tensor.T

        # print(dot_similarity_text.shape, label.shape)
        # break
        labels.append(label.numpy())
        preds.append(dot_similarity_text.cpu().numpy())
        st_bar.set_description(desc=f"Evaluation")

    preds_np = np.concatenate(preds)
    labels_np = np.stack(labels)
    precision, recall = precision_recall_topk(labels_np, preds_np, max_sep_topk)
    F_max = 2 * precision * recall  / (precision + recall)

    lables_list = []
    for ll in labels_np:
        lables_list.append(ll.argmax())
    topkacc = top_k_accuracy_score(
        np.stack(lables_list), preds_np, k=25, labels=np.arange(0, preds_np.shape[1])
    )
    roc_auc = roc_auc_score(labels_np, preds_np, average='micro')
    auprc = average_precision_score(labels_np, preds_np, average='micro')
    return {"Fmax": F_max, "AUPRC": auprc, "AUROC": roc_auc, 'TOPK_ACC': topkacc}

def first_occurrence_indices(input_list):
    # Util function to filter non-unique embeddings

    num_unique = len(np.unique(input_list))
    assert num_unique == (max(input_list)+1)
    result = np.full(shape=(num_unique,), fill_value = -1)      # List to store the output indices

    for i, num in enumerate(input_list):
        if result[num] == -1:
            result[num] = i

    assert (result == -1).sum() == 0

    return result

def get_predictions_protein_retrieval(
        text_embeds,
        prot_embeds,
        relation_file,
        protein_file = "integrated_data/v1/protein/protein_sequences.fa",
        text_file = "integrated_data/v1/go/go_info_filtered.pkl",
    ):
    '''
    Assumes cosine similarity for similarity score between text embeddings and protein embeddings

    Assumes text and protein embeddings are sorted according to relations found in relation_file
        - Note: cannot make this assumption, need to re-index

    Can bypass this if text_embeds and prot_embeds are already aligned by calling compute_cosine_sim_preds
    '''

    # First get labels, as per algorithm above:
    #id2seq:dict = get_seqid_seq_mapping(protein_file)

    max_num = text_embeds.shape[0]
    assert max_num == prot_embeds.shape[0]

    txtid_to_seqid, (unique_texts, text_inverse, text_to_unique_map), (unique_seqs, seq_inverse, seq_to_unique_map) = get_text_aaseq_mapping(relation_file, max_num = max_num)

    n_text = len(unique_texts)
    n_prot = len(unique_seqs)

    labels = []
    #st_bar = tqdm(txtid_to_seqid)
    st_bar = txtid_to_seqid
    #filled_gos = np.zeros(unique_texts, dtype = bool)
    labels = np.zeros((n_text, n_prot))
    for item in st_bar:
        # seq
        go_id = item
        seq_id_list = txtid_to_seqid[go_id]
        label, label_index = None, None
        for seqid_index in seq_id_list:
            #if seqid in prot_keys:
            seqid_tensor = torch.tensor(seq_to_unique_map[seqid_index])
            #label_index = seqid_index
            if label is None:
                label = F.one_hot(seqid_tensor, num_classes=n_prot)
            else:
                label += F.one_hot(seqid_tensor, num_classes=n_prot)
        if label is None:
            continue

        labels[text_to_unique_map[go_id],:] = label.numpy()

    # Can assume text and prot embeds are both indexed to txtid_to_seqid dataframe

    # Key: assume there are repeat elements in the prot_embeds (i.e. repeat elements in seq_inverse)
    text_occurence = torch.from_numpy(first_occurrence_indices(text_inverse)).long()
    prot_occurence = torch.from_numpy(first_occurrence_indices(seq_inverse)).long()

    preds_np = compute_cosine_sim_preds(text_embeds[text_occurence,:], prot_embeds[prot_occurence,:])

    return preds_np, labels

def get_predictions_protein_retrieval_unique(
        text_embeds,
        prot_embeds,
        relation_file,
        text_alignment_relations,
        prot_alignment_relations,
        extra_prot_embeds = None,
    ):
    '''
    text_alignment_relations: provides reference to the ordering of text_embeds
    prot_alignment_relations: provides reference to the ordering of prot_embeds
    '''

    # First, need to get labels
    #max_num = text_embeds.shape[0]
    # assert max_num == prot_embeds.shape[0]

    # NOTE: This can be made more efficient by saving the below values and just doing this once, but would require refactoring the code.
    #   It's fairly fast for protein-go (probably our largest dataset), so I'm (Owen) leaving it for now
    if isinstance(relation_file, str):
        txtid_to_seqid, (unique_texts, text_inverse, text_to_unique_map), (unique_seqs, seq_inverse, seq_to_unique_map) = get_text_aaseq_mapping(relation_file, max_num = None)
    else:
        txtid_to_seqid, (unique_texts, text_inverse, text_to_unique_map), (unique_seqs, seq_inverse, seq_to_unique_map) = get_text_aaseq_mapping_PD(relation_file, max_num = None)

    n_text = len(unique_texts)
    n_prot = len(unique_seqs)

    labels = []
    st_bar = tqdm(txtid_to_seqid)
    #filled_gos = np.zeros(unique_texts, dtype = bool)
    labels = np.zeros((n_text, n_prot))
    for item in st_bar:
        # seq
        go_id = item
        seq_id_list = txtid_to_seqid[go_id]
        label, label_index = None, None
        for seqid_index in seq_id_list:
            #if seqid in prot_keys:
            seqid_tensor = torch.tensor(seq_to_unique_map[seqid_index])
            #label_index = seqid_index
            if label is None:
                label = F.one_hot(seqid_tensor, num_classes=n_prot)
            else:
                label += F.one_hot(seqid_tensor, num_classes=n_prot)
        if label is None:
            continue

        labels[text_to_unique_map[go_id],:] = label.numpy()

    # Need to align unique text_embeds and unique prot_embeds
    # TODO: Make this more efficient if we need to
    ta = list(text_alignment_relations)
    sa = list(prot_alignment_relations)
    text_alignment_ind = [ta.index(i) for i in unique_texts]
    prot_alignment_ind = [sa.index(i) for i in unique_seqs]
    # text_alignment_ind = torch.LongTensor([text_to_unique_map[i] for i in text_alignment_relations])
    # prot_alignment_ind = torch.LongTensor([seq_to_unique_map[i] for i in prot_alignment_relations])

    text_embeds = text_embeds[text_alignment_ind,:]
    prot_embeds = prot_embeds[prot_alignment_ind,:]

    if extra_prot_embeds is not None:
        # Need to concatenate onto prot_embeds which is size (n_p, d_z)
        prot_embeds = torch.cat([prot_embeds, extra_prot_embeds], dim=0)
        # Labels get all zero's attached where all the extra prot embeddings go, but need to calculate size
        # labels size (text num, prot_num)
        extra_labels = np.zeros((text_embeds.shape[0], extra_prot_embeds.shape[0])) # MUST be all zeros
        labels = np.concatenate([labels, extra_labels], axis=1)

    preds_np = compute_cosine_sim_preds(text_embeds, prot_embeds)

    return preds_np, labels


def compute_cosine_sim_preds(text_embeds, prot_embeds):
    # Normalize text, prot embeds:
    text_embeds_fp = text_embeds.to(dtype=torch.float32)
    prot_embeds_fp = prot_embeds.to(dtype=torch.float32)

    text_embeds = F.normalize(text_embeds_fp, dim=-1)
    prot_embeds = F.normalize(prot_embeds_fp, dim=-1)

    # Compute similarity:
    preds = torch.matmul(text_embeds, prot_embeds.transpose(0,1))
    preds_np = preds.detach().clone().cpu().numpy()
    #labels_np = np.stack(labels)

    return preds_np

def protein_retrieval_eval_from_embeddings(
        text_embeds,
        prot_embeds,
        relation_file,
        extra_prot_embeds = None,
        protein_file = "integrated_data/v1/protein/protein_sequences.fa",
        text_file = "integrated_data/v1/go/go_info_filtered.pkl",
        text_alignment_relations = None,
        prot_alignment_relations = None,
        max_sep_topk = 25
    ):
    '''
    Computes protein retrieval evaluation from predictions already generated
    '''

    if isinstance(relation_file, str):
        relation_file = os.path.join(DATA_DIR, relation_file)

    if (prot_alignment_relations is None) and (text_alignment_relations is None):
        preds_np, labels_np = get_predictions_protein_retrieval(
            text_embeds = text_embeds,
            prot_embeds = prot_embeds,
            relation_file = relation_file,
            protein_file = os.path.join(DATA_DIR, protein_file),
            text_file = os.path.join(DATA_DIR, text_file)
        )
    else:
        preds_np, labels_np = get_predictions_protein_retrieval_unique(
            text_embeds = text_embeds,
            prot_embeds = prot_embeds,
            relation_file = relation_file,
            text_alignment_relations = text_alignment_relations,
            prot_alignment_relations = prot_alignment_relations,
            extra_prot_embeds = extra_prot_embeds,
        )

    precision, recall = precision_recall_topk(labels_np, preds_np, max_sep_topk)
    F_max = 2 * precision * recall  / (precision + recall)

    lables_list = []
    for ll in labels_np:
        lables_list.append(ll.argmax())
    topkacc = top_k_accuracy_score(
        np.stack(lables_list), preds_np, k=max_sep_topk, labels=np.arange(0, preds_np.shape[1])
    )
    if not np.any(labels_np.sum(axis=1) > 1): # Multiclass output detected
        roc_auc = roc_auc_score(labels_np, preds_np, average='micro', multi_class = 'ovr')
        auprc = average_precision_score(labels_np, preds_np, average='micro')
    else: # Not a multiclass output detected
        roc_auc = -1
        auprc = -1
    return {"Fmax": F_max, "AUPRC": auprc, "AUROC": roc_auc, 'TOPK_ACC': topkacc}

if '__main__' == __name__:
    protein_file = data_root.joinpath('protein_sequences.fa')
    text_file = data_root.joinpath("go_info_filtered.pkl")
    test_csv = data_root.joinpath('/workspace/BioTranslatorProject/data/protein_go_relations_eval_five_shot.csv')
    result_dict = protein_go_retrieval(
        model,
        get_protein_feature,
        get_text_feature,
        protein_file,
        text_file,
        test_csv,
    )
    print(result_dict)


def get_closest_uniprotIDs(
        protein_path = 'integrated_data/v1/'
    ):
    pass
