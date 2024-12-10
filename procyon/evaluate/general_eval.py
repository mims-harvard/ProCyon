import os, argparse
from typing import List
import pandas as pd
import numpy as np
from tqdm import trange, tqdm

import json
import torch

from torch.utils.data import DataLoader

from procyon.model.model_unified import UnifiedProCyon
from procyon.training.train_utils import get_qa_scores, get_qa_metrics_from_preds

from procyon.data.data_utils import DATA_DIR
from procyon.data.dataset import ProteinEvalDataset

from procyon.training.train_utils import (
    get_cl_metrics,
    get_qa_metrics,
    get_qa_metrics_from_preds,
    get_caption_pairs,
    get_caption_metrics_from_preds,
    get_retrieval_scores_inbatch,
    get_IT_datasets,
    get_data_collators_IT,
    get_data_collators_IT_new
)

from procyon.evaluate.eval_utils import protein_retrieval_eval_from_embeddings
from procyon.data.data_utils import get_relation_fname

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_inputs(inputs):
    # TODO: Move this to a utils file
    for k, v in inputs.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, torch.Tensor):
                    inputs[k][k2] = v2.to(device)
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    return inputs

def make_eval_argparser_OLD():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shot_level', type=str, required=True, choices=['pt_ft', 'zero_shot', 'five_shot'])
    #parser.add_argument('--text_variant_type', type=str, default='standard')
    #parser.add_argument('--relation_type', type=str, default=None, choices=[None, 'drug_target', 'drug_transporter', 'drug_carrier', 'drug_enzyme'])
    parser.add_argument('--split_method', type=str, required=True, help="how the data is split",
        choices=[ # TODO: add to options
            "sample_aware_ontology_go_centric",
            "oott_eval",
        ])
    parser.add_argument('--model_dir', type=str, required=True, help="Path to model directory")
    #parser.add_argument('--num_negatives', type=int, default=None, help="Number of negative samples to sample for each positive sample. Default is to use in-batch negative samples")
    parser.add_argument('--text_col_name', type=str, default="default", help="name of text col (to save in output fname)")
    parser.add_argument('--batch_size', type=int, default=8, help="Size of largest number of positive samples to feed through model at one time")
    parser.add_argument('--k', type=int, default=25)
    parser.add_argument('--task_type', type = str, default="qa", choices=["qa","retrieval","caption"])
    parser.add_argument('--device', type=str, default='cuda')

    # Debugging:
    parser.add_argument('--use_val_split', action = 'store_true', help = 'If True, runs the evaluation with validation data in same pipeline as training would be')
    parser.add_argument('--max_num_pos_samples', type = int, default = None, help = "Maximum number of positive samples on which to evaluate. For quicker evaluations.")

    # For score saving:
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--scores_subdir', type=str, default=None)
    parser.add_argument('--regenerate', default=True, help="Whether to recalculate results if file with same name exists")

    # For separately loading aaseq embeddings:
    parser.add_argument('--fixed_aaseq_embed_path', type=str, default=None)
    parser.add_argument('--fixed_aaseq_idmap_path', type=str, default=None)

    parser.add_argument('--old_eval', action='store_true')
    parser.add_argument('--disable_all_eval', action='store_true')

    return parser

def create_eval_argparser():
    # From GPT-4

    parser = argparse.ArgumentParser(description="Arguments used for benchmark evaluation")

    # Model
    parser.add_argument("--model_dir", default=None,
                        help="Checkpoint directory of model to evaluate")

    # Processing arguments
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Maximum size of batch to feed into model during evaluation - based on memory constraints")
    parser.add_argument("--max_num_positive_qa_samples", type=int, default=None,
                        help="Maximum number of (positive) samples on which to evaluate QA")
    parser.add_argument("--max_num_captioning_samples", type=int, default=None,
                        help="Maximum number of samples on which to evaluate captioning")

    # Evaluation task options
    parser.add_argument("--evaluate_qa", action="store_true",
                        help="Evaluate model on QA task")
    parser.add_argument("--evaluate_retrieval", action="store_true",
                        help="Evaluate model on retrieval task")
    parser.add_argument("--evaluate_caption", action="store_true",
                        help="Evaluate model on captioning task")

    # Dataset options
    parser.add_argument("--shot_level", default=False, choices=["pt_ft", "five_shot", "zero_shot"],
                        help="Shot level for datasets")
    parser.add_argument("--aaseq_type", default="protein", choices=["protein", "domain"],
                        help="Type of amino acid sequence in the dataset")
    parser.add_argument("--text_type", default="go",
                        help="Type of text in the dataset desired")
    parser.add_argument("--relation_type", default="all",
                        help="Type of relation to consider for the evaluation")
    parser.add_argument("--text_variant_type", default="standard",
                        help="The type of description to use for text.")

    # Metric parameters
    parser.add_argument("--eval_k", type=int, default=25,
                        help="k number for retrieval evaluation calculation")
    parser.add_argument("--num_neg_samples_qa", type=int, default=1,
                        help="Number of negative examples for QA evaluation")

    # GO-specific arguments
    parser.add_argument("--go_def_col", default="standard", choices=["description_combined", "standard", "name_def", "def_only"],
                        help="The name of the text to use for GO descriptions during training.")
    parser.add_argument("--go_split_method", default="sample_aware_ontology_go_centric",
                        help="The method to split GO terms into CL train, CL val, eval pt-ft, eval few-shot, and eval zero-shot sets.")

    return parser


@torch.no_grad()
def get_validation_scores(
        model: UnifiedProCyon,
        dataloader,
        batch_size: int,
        task_type: List[str],
        max_num_pos_samples: int = None,
    ):
    # Calculate higher end of iteration numbers:
    max_num_pos_samples = 1e9 if max_num_pos_samples is None else max_num_pos_samples

    metrics = []

    for i, model_inputs in enumerate(dataloader):
        # Accumulate predictions:
        if (i * batch_size) > max_num_pos_samples:
            break

        # TODO: Below is kind of a hacky fix - could be better
        model_inputs = prepare_inputs(model_inputs)

        if task_type == 'qa' or task_type == 'caption':
            out = model(model_inputs, retrieval = False, get_full_labels = True)
            acc, f1 = get_qa_metrics(out, yes_token = model.yes_token, no_token = model.no_token, padding_token = model.tokenizer.pad_token_id) #answer_token = model.answer_idx)
            met_i = (acc, f1)

        elif task_type == 'retrieval':
            out = model(model_inputs, retrieval = True)
            pos_scores, neg_scores = get_retrieval_scores_inbatch(out['contrastive_out'])
            pos_count, neg_count, auroc, auprc = get_cl_metrics(pos_scores, neg_scores)
            print('Auroc', auroc, 'auprc', auprc)
            met_i = (auroc, auprc)

        else:
            raise ValueError('Evaluation type {} not recognized'.format(task_type))

        metrics.append(met_i)

    return np.array(metrics).mean(axis=0)

@torch.no_grad()
def get_embeddings_retrieval(
        model: UnifiedProCyon,
        protein_dataloader,
        text_dataloader,
        protein_batch_size: int = None,
        text_batch_size: int = None,
        max_num_pos_samples: int = None,
    ):
    '''
    Performs a more efficient version of retrieval than previous version
        - Key is that we only want to forward pass each text and protein ONCE
        - Need to use separate functions
    '''
    model_device = model.device # NOTE: This won't work for general modules, only UnifiedTxPLM

    # Note that both functions call independent modules, so having two for loops is fine

    # Passing proteins via dataloader:
    protein_embeddings = []
    protein_ids = []
    for i, model_inputs in enumerate(protein_dataloader):
        if isinstance(model_inputs, dict):
            model_inputs["data"] = model_inputs["data"].to(model_device)
            protein_inputs = model_inputs
            protein_ids += model_inputs["indices"]
        else:
            protein_inputs = model_inputs.to(model_device)
            protein_ids += model_inputs.detach().clone().cpu().tolist()

        out = model.forward_sequences(protein_inputs)
        protein_embeddings.append(out["shared"].detach().clone().cpu())

    #     protein_ids += model_inputs.detach().clone().cpu().tolist()

    prot_indices = torch.LongTensor(protein_ids)

    protein_embeddings = torch.cat(protein_embeddings, dim = 0)

    # Text inputs:
    # Still need to pass through collator, e.g. loader
    text_embeddings = []
    text_indices = []

    for i, model_inputs in enumerate(text_dataloader):
        model_inputs = prepare_inputs(model_inputs)

        if len(np.array(model_inputs['input']['text']).shape) == 1: # Lazy way to recognize old method of retrieval
            unique_inds = model_inputs['reference_indices']['text'].detach().clone().cpu().tolist()
            select_model_inds = model_inputs['input']['text']
        else:
            unique_inds = model_inputs['reference_indices']['text']
            select_model_inds = [model_inputs['input']['text'][i][-1] for i in range(len(model_inputs['input']['text']))]
            #unique_inds = [unique_inds[i][-1] for i in range(len(unique_inds))] # Get last element of each sub-list

        reshuffled = [unique_inds[i] for i in select_model_inds]
            # Must reshuffle by input text, this is for if there are multiple of one text in input
        text_indices += reshuffled
        #text_indices.append(unique_inds)

        out = model(model_inputs, retrieval = True)
        text_ret_embs = out['contrastive_out']['positive']['text'].detach().clone().cpu()
        text_embeddings.append(text_ret_embs)

    text_embeddings = torch.cat(text_embeddings, dim = 0)
    #text_indices = torch.LongTensor(text_indices)
    #text_indices = torch.cat(text_indices)

    return text_embeddings, protein_embeddings, text_indices, prot_indices

@torch.no_grad()
def get_testing_predictions(
        model: UnifiedProCyon,
        dataloader,
        batch_size: int,  # Only used to calculate bound for max_num_pos_samples - TODO: could extract this from dataloader
        task_type: List[str],
        protein_dataloader = None,
        max_num_pos_samples: int = None,
        old_eval = False,
    ):
    '''
    Get testing predictions for a given datalaoder
    '''
    task_type = task_type.lower()
    max_num_pos_samples = 1e9 if max_num_pos_samples is None else max_num_pos_samples

    predictions = []
    reference_indices = []

    if batch_size is None:
        batch_size = 0 # Makes sure it's always below max_num_pos_samples

    # TODO: make progress bar instead of running iteration counter in tqdm
    if task_type != 'retrieval' or old_eval:
        for i, model_inputs in tqdm(enumerate(dataloader), total = min(len(dataloader.dataset) // batch_size, max_num_pos_samples // batch_size)):

            if (i * batch_size) > max_num_pos_samples:
                break

            model_inputs = prepare_inputs(model_inputs)

            if task_type == 'qa':
                out = model(model_inputs, retrieval = False, get_full_labels = True)
                # Get QA scores:
                pred_toks, y_toks = get_qa_scores(out, answer_token = model.answer_idx) #padding_token = model.tokenizer.pad_token_id)

                predictions.append((pred_toks, y_toks))

            elif (task_type == 'retrieval' and old_eval):
                out = model(model_inputs, retrieval = True)

                text_ret_embs = out['contrastive_out']['positive']['text'].detach().clone().cpu()
                prot_ret_embs = out['contrastive_out']['positive']['sequence'].detach().clone().cpu()

                predictions.append((text_ret_embs, prot_ret_embs))

                # Get protein and text id's:
                text_idx, seq_idx = model_inputs["reference_indices"]["text"], model_inputs["reference_indices"]["seq"]
                unrolled_ref_idx = list(zip(text_idx, seq_idx))
                reference_indices += unrolled_ref_idx

            elif task_type == 'caption':
                out = model(inputs=model_inputs,
                            return_mlm=False,
                            retrieval=False,
                            get_full_labels=True)

                # Get list of (generated_caption, reference_caption) string pairs
                caption_pairs = get_caption_pairs(out, model.tokenizer)
                predictions.extend(caption_pairs)
            else:
                raise ValueError('Evaluation type {} not recognized'.format(task_type))

        LP = len(predictions)

    if task_type == 'qa':
        all_preds = torch.cat([predictions[i][0] for i in range(LP)])
        all_y = torch.cat([predictions[i][1] for i in range(LP)])
        return (all_preds, all_y), reference_indices
    elif task_type == 'retrieval':
        if old_eval:
            all_text_embs = torch.cat([predictions[i][0] for i in range(LP)], dim = 0)
            all_prot_embs = torch.cat([predictions[i][1] for i in range(LP)], dim = 0)
            return (all_text_embs, all_prot_embs), reference_indices
        else:
            all_text_embs, all_prot_embs, text_indices, prot_indices = get_embeddings_retrieval(
                model = model,
                protein_dataloader = protein_dataloader,
                text_dataloader = dataloader,
                protein_batch_size = batch_size,
                text_batch_size = batch_size,
                max_num_pos_samples = max_num_pos_samples,
            )
            return (all_text_embs, all_prot_embs), (text_indices, prot_indices) #, reference_indices
    elif task_type == 'caption':
        # TODO(rcalef): not sure if I should be populating these `reference_indices`
        return predictions, reference_indices
    else:
        raise ValueError(f"Unexpected task type: {task_type}")

def reindex_test_preds(test_preds, reference_indices, relation_file):
    '''
    Re-indexes to the original indices followed in the relation file
        - Allows indexing as if directly indexing the relation file in downstream functions
    '''
    df_rels = pd.read_csv(os.path.join(DATA_DIR, relation_file))

    # Convert to tuple map:
    tmap = {(df_rels['text_id'].iloc[i], df_rels['seq_id'].iloc[i]):i for i in range(df_rels.shape[0])}

    idx_map = torch.LongTensor([tmap[ri] for ri in reference_indices])

    text_embs = test_preds[0][idx_map]
    prot_embs = test_preds[1][idx_map]

    return text_embs, prot_embs

# TODO: Expand the arguments in the function signature - can use for trainIT as well
def eval_validation(
        logger,
        model_dir,
        task_type,
        val_split,
        batch_size,
        max_num_pos_samples
    ):

    # Load model:
    logger.info(f'Loading model from {model_dir}')
    model, model_args = UnifiedProCyon.from_pretrained(pretrained_weights_dir=DATA_DIR+"model_weights/", checkpoint_dir=model_dir, device = device)
    logger.info(f'Loaded model with config: {model_args}')

    model.eval()

    # TODO: Implement loading static protein embeddings here

    # Get data args, train args from checkpoints:
    data_args = torch.load(os.path.join(model_dir, 'data_args.pt'))
    train_args = torch.load(os.path.join(model_dir, 'training_args.pt'))

    data_args.val_split_type = val_split

    # Load collators, datasets:

    _, val_dataset = get_IT_datasets(data_args, task_type = task_type)
    collators = get_data_collators_IT(data_args, model_args)
    protein_mlm_collator, qa_collator, retrieval_collator, retrieval_eval_collator, caption_collator = collators

    #assert task_type == 'retrieval'
    collator = qa_collator

    # Make dataloader for validation:
    # TODO: Add here to make parallelizable
    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        collate_fn = collator,
        num_workers = 4,
        drop_last = True,
        shuffle = False # FALSE FOR NOW CAN CHANGE LATER
    )

    scores = get_validation_scores(
        model = model,
        dataloader = val_loader,
        batch_size = batch_size,
        task_type = task_type,
        max_num_pos_samples = max_num_pos_samples
    )

    if task_type == 'qa':
        logger.info('Scores Acc = {:.4f} F1 = {:.4f}'.format(scores[0], scores[1]))
    elif task_type == 'retrieval':
        logger.info('Scores AUROC = {:.4f} AUPRC = {:.4f}'.format(scores[0], scores[1]))

def eval_OLD(
        logger,
        model_dir,
        text_col_name,
        eval_args,
        task_type: str = 'retrieval',
        shot_level: str = 'pt_ft',
        batch_size: int = 8,
        max_num_pos_samples = None,
        old_eval = False,
    ):
    '''
    Testing evaluation - not to be used in on-the-fly evaluation or evaluation with validation data

    TODO: Make this work with new setup
    '''

    # Load model:
    logger.info(f'Loading model from {model_dir}')
    model, model_args = UnifiedProCyon.from_pretrained(pretrained_weights_dir=DATA_DIR+"model_weights/", checkpoint_dir=model_dir, device = device)
    logger.info(f'Loaded model with config: {model_args}')

    model.eval()

    # Get data args, model_args
    data_args = torch.load(os.path.join(model_dir, 'data_args.pt'))
    train_args = torch.load(os.path.join(model_dir, 'training_args.pt'))

    # Change data args as needed:
    if text_col_name != 'default':
        data_args.go_def_col = text_col_name

    # get_it_datasets
    testing_kwargs = {
        "shot_level": shot_level,
        "use_preset_negatives": False if task_type == 'retrieval' else True,
        "num_negatives": 1 if task_type != 'retreival' else None,
    }
    test_dataset = get_IT_datasets(data_args, task_type = task_type, testing = True, testing_kwargs = testing_kwargs)
    collators = get_data_collators_IT(data_args, model_args)
    protein_mlm_collator, qa_collator, retrieval_collator, retrieval_eval_collator, caption_collator = collators

    if task_type == 'qa':
        collator = qa_collator
    elif task_type == 'retrieval':
        collator = retrieval_eval_collator
    elif task_type == 'caption':
        collator = caption_collator

    # Make dataloader for testing
    loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        collate_fn = collator,
        num_workers = 2,
        drop_last = False,
        shuffle = False, # FALSE FOR NOW CAN CHANGE LATER
        pin_memory = True,
    )

    # Get protein dataloader if needed:
    protein_dataloader = None
    if task_type == "retrieval":
        protein_dataset = ProteinEvalDataset(test_dataset.unique_aaseq)
        protein_dataloader = DataLoader(
            protein_dataset,
            batch_size = batch_size,
            num_workers = 2,
            drop_last = False,
            shuffle = False,
            pin_memory = True,
        )

    # Run inference loop to get testing predictions

    test_preds, reference_indices = get_testing_predictions(
        model = model,
        dataloader = loader,
        protein_dataloader = protein_dataloader,
        batch_size = batch_size,
        task_type = task_type,
        max_num_pos_samples = max_num_pos_samples,
        old_eval = old_eval,
    )

    # Task-specific evaluations
    if task_type == 'retrieval':
        # # Relation file (containing text-prot relation pairs)
        relation_filename = get_relation_fname(go_split_method = data_args.go_split_method, shot_level = shot_level, split = 'test')

        # Retrieval-based evaluation
        if old_eval:
            # Need to re-index test_preds
            test_preds = reindex_test_preds(test_preds, reference_indices, relation_filename)
            metric_dict = protein_retrieval_eval_from_embeddings(
                text_embeds = test_preds[0],
                prot_embeds = test_preds[1],
                relation_file = relation_filename,
                protein_file = "integrated_data/v1/protein/protein_sequences.fa",
                text_file = "integrated_data/v1/go/go_info_filtered.pkl",
                text_alignment_relations = None,
                prot_alignment_relations = None,
                max_sep_topk = 25
            )
        else:
            metric_dict = protein_retrieval_eval_from_embeddings(
                text_embeds = test_preds[0],
                prot_embeds = test_preds[1],
                relation_file = relation_filename,
                protein_file = "integrated_data/v1/protein/protein_sequences.fa",
                text_file = "integrated_data/v1/go/go_info_filtered.pkl",
                text_alignment_relations = reference_indices[0],
                prot_alignment_relations = reference_indices[1].numpy(),
                max_sep_topk = 25
            )
        logger.info('Scores')
        logger.info(metric_dict)
    elif task_type == 'qa':
        # QA-based evaluation, similar to in training
        acc, f1 = get_qa_metrics_from_preds(
            pred_toks = test_preds[0],
            y_toks = test_preds[1],
            yes_token = model.yes_token,
            no_token = model.no_token,
        )
        print('Scores:')
        print(f'Acc = {acc:.4f} F1 = {f1:.4f}')
    elif task_type == 'caption':
        pass
    else:
        raise NotImplementedError

def print_args_values(args, logger):
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

ALL_PROTEINS_FILE = os.path.join(DATA_DIR, "integrated_data/v1/protein/protein_info_filtered.pkl")

def eval(
        logger,
        eval_args,
    ):
    '''
    Testing evaluation - not to be used in on-the-fly evaluation or evaluation with validation data

    TODO: Make more sophisticated logging mechanisms, i.e. saving metrics in files
    '''

    # Load model:
    logger.info(f'Loading model from {eval_args.model_dir}')
    model, model_args = UnifiedProCyon.from_pretrained(pretrained_weights_dir=DATA_DIR+"model_weights/", checkpoint_dir=eval_args.model_dir, device = device)
    logger.info(f'Loaded model with config: {model_args}')

    model.to(device)

    logger.info('Evaluation args')
    print_args_values(eval_args, logger)

    model.eval()

    # Get data args, model_args
    data_args = torch.load(os.path.join(eval_args.model_dir, 'data_args.pt'))
    train_args = torch.load(os.path.join(eval_args.model_dir, 'training_args.pt'))
    logger.info(f"Training args: {train_args}")

    # Change data args as needed to fit evaluation:
    if eval_args.text_variant_type is not None:
        data_args.go_def_col = eval_args.text_variant_type
    if eval_args.num_neg_samples_qa is not None:
        data_args.num_neg_samples_qa = eval_args.num_neg_samples_qa
    if eval_args.go_split_method is not None:
        data_args.go_split_method = eval_args.go_split_method
    if eval_args.go_def_col is not None:
        data_args.go_def_col = eval_args.go_def_col
    if eval_args.shot_level is not None:
        data_args.val_split_type = eval_args.shot_level

    logger.info(f'Data args after modifications: {data_args}')

    task_types_considered = []
    if eval_args.evaluate_qa:
        task_types_considered.append("qa")
    if eval_args.evaluate_retrieval:
        task_types_considered.append("retrieval")
    if eval_args.evaluate_caption:
        task_types_considered.append("caption")

    all_metric_dicts = {}

    for task_type in task_types_considered:

        # get_it_datasets
        testing_kwargs = {
            "shot_level": eval_args.shot_level,
            "use_preset_negatives": False if task_type == 'retrieval' else True,
            "num_negatives": 1 if task_type != 'retreival' else None,
        }

        # Get testing datasets with appropriate eval inputs
        test_dataset = get_IT_datasets(
            data_args = data_args,
            task_type = task_type,
            aaseq_type = eval_args.aaseq_type,
            text_type = eval_args.text_type,
            relation_type = eval_args.relation_type,
            testing = True, # Always true
            testing_kwargs = testing_kwargs
        )

        # Can this be moved out?
        collators = get_data_collators_IT_new(
            data_args,
            model_args,
            aaseq_types = [eval_args.aaseq_type],
            text_types = [eval_args.text_type],
            relation_types = [[eval_args.relation_type]],
        )
        _, qa_collators, retrieval_collators, caption_collators = collators

        # Index each of the dictionaries (set up for multiple evaluations)
        ind_str = eval_args.text_type+'_'+eval_args.relation_type
        if task_type == 'qa':
            collator = qa_collators[ind_str]
        elif task_type == 'retrieval':
            collator = retrieval_collators[ind_str]
        elif task_type == 'caption':
            collator = caption_collators[ind_str]

        # Make dataloader for testing
        loader = DataLoader(
            test_dataset,
            batch_size = eval_args.batch_size,
            collate_fn = collator,
            num_workers = 2,
            drop_last = False,
            shuffle = False, # FALSE FOR NOW CAN CHANGE LATER
            pin_memory = True,
        )

        # Get protein dataloader if needed:
        protein_dataloader = None
        if task_type == "retrieval":
            protein_dataset = ProteinEvalDataset(test_dataset.unique_aaseq)
            protein_dataloader = DataLoader(
                protein_dataset,
                batch_size = eval_args.batch_size,
                num_workers = 2,
                drop_last = False,
                shuffle = False,
                pin_memory = True,
            )

        # Run inference loop to get testing predictions
        test_preds, reference_indices = get_testing_predictions(
            model = model,
            dataloader = loader,
            protein_dataloader = protein_dataloader,
            batch_size = eval_args.batch_size,
            task_type = task_type,
            max_num_pos_samples = eval_args.max_num_positive_qa_samples,
            old_eval = False,
        )

        # Task-specific evaluations
        if task_type == 'retrieval':
            # # Relation file (containing text-prot relation pairs)
            # relation_filename = get_relation_fname(
            #     aaseq_type = eval_args.aaseq_type,
            #     text_type = eval_args.text_type,
            #     go_split_method = eval_args.go_split_method,
            #     shot_level = eval_args.shot_level,
            #     split = 'test'
            # )

            relation_file = pd.DataFrame(test_dataset.aaseq_text_relations)

            if not eval_args.disable_all_eval:
                # Need to get extra sequences:
                df_all_prots = pd.read_pickle(ALL_PROTEINS_FILE)

                set_A = set(relation_file['seq_id'])
                set_B = set(df_all_prots['index'])
                # Union minus intersection
                sym_diff = (set_B - set_A) # all_proteins excluding ones in the evaluation set

                filtered_df = df_all_prots.loc[df_all_prots['index'].isin(sym_diff),:]

                # Get protein embeddings for all those in filtered:
                protein_dataset = ProteinEvalDataset(filtered_df['index'])
                protein_else_dataloader = DataLoader(
                    protein_dataset,
                    batch_size = eval_args.batch_size,
                    num_workers = 2,
                    drop_last = False,
                    shuffle = False,
                    pin_memory = True,
                )

                # Run inference loop:
                # Passing proteins via dataloader:
                model_device = model.device
                extra_protein_embeddings = []
                for i, model_inputs in enumerate(protein_else_dataloader):
                    if isinstance(model_inputs, dict):
                        model_inputs["data"] = model_inputs["data"].to(model_device)
                        protein_inputs = model_inputs
                    else:
                        protein_inputs = model_inputs.to(model_device)

                    out = model.forward_sequences(protein_inputs)
                    extra_protein_embeddings.append(out["shared"].detach().clone().cpu())

                extra_prot_embeds = torch.cat(extra_protein_embeddings, dim = 0)

            else:
                extra_prot_embeds = None

            # Get all extra embeddings

            # Retrieval-based evaluation
            metric_dict = protein_retrieval_eval_from_embeddings(
                text_embeds = test_preds[0],
                prot_embeds = test_preds[1],
                relation_file = relation_file,
                extra_prot_embeds = extra_prot_embeds,
                text_alignment_relations = reference_indices[0],
                prot_alignment_relations = reference_indices[1].numpy(),
                max_sep_topk = eval_args.eval_k,
            )
            logger.info('Scores')
            logger.info(metric_dict)
            all_metric_dicts['retrieval'] = metric_dict
        elif task_type == 'qa':
            # QA-based evaluation, similar to in training
            acc, f1 = get_qa_metrics_from_preds(
                pred_toks = test_preds[0],
                y_toks = test_preds[1],
                yes_token = model.yes_token,
                no_token = model.no_token,
            )
            logger.info('Scores:')
            logger.info(f'Acc = {acc:.4f} F1 = {f1:.4f}')
            all_metric_dicts['qa'] = {'acc': acc, 'f1': f1}
        elif task_type == 'caption':
            metric_dict = get_caption_metrics_from_preds(test_preds)
            logger.info('Scores:')
            logger.info(metric_dict)
        else:
            raise NotImplementedError

    return all_metric_dicts

def eval_wrapper_OLD(args, logger):

    logger.info('args:')
    logger.info(json.dumps(args.__dict__, indent=4))

    if args.use_val_split:
        logger.info('Run validation split')
        eval_validation(
            logger = logger,
            task_type = args.task_type,
            val_split = args.shot_level,
            model_dir = args.model_dir,
            batch_size = args.batch_size,
            max_num_pos_samples = args.max_num_pos_samples
        )
    else:
        logger.info('Run test split')
        eval(
            logger,
            model_dir = args.model_dir,
            text_col_name = args.text_col_name,
            task_type = args.task_type,
            shot_level = args.shot_level,
            batch_size = args.batch_size,
            max_num_pos_samples = args.max_num_pos_samples,
            old_eval = args.old_eval
        )

