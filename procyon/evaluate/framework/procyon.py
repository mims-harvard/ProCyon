import os

from collections import defaultdict
from typing import (
    Callable,
    Dict,
    List,
    Tuple,
)

import torch
import tqdm
import pandas as pd
import numpy as np

import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm

from procyon.data.dataset import (
    AASeqDataset,
    AASeqTextUnifiedDataset,
)

from procyon.model.model_unified import (
    DEFAULT_PRETRAINED_WEIGHTS_DIR,
    UnifiedProCyon,
)

from procyon.training.training_args_IT import ModelArgs

from procyon.evaluate.framework.args import EvalArgs
from procyon.evaluate.framework.caption import AbstractCaptionModel
from procyon.evaluate.framework.qa import AbstractQAModel
from procyon.evaluate.framework.retrieval import (
    AbstractRetrievalModel,
    get_retrieval_target_proteins_loader,
    get_retrieval_target_set,
)
from procyon.evaluate.framework.utils import (
    compare_and_warn_model_args,
    move_inputs_to_device,
)

from procyon.training.train_utils import get_qa_scores


class ProcyonCaptionEval(AbstractCaptionModel):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        checkpoint_dir = model_config["checkpoint_dir"]
        model, checkpoint_model_args = UnifiedProCyon.from_pretrained(
            pretrained_weights_dir=DEFAULT_PRETRAINED_WEIGHTS_DIR,
            checkpoint_dir=checkpoint_dir,
        )
        compare_and_warn_model_args(model_args, checkpoint_model_args)

        model.eval()
        model.bfloat16()
        self.device = device
        self.model = model.to(self.device)
        self.model_args = model_args
        self.checkpoint_dir = checkpoint_dir
        self.max_len = eval_args.caption_max_len
        self.method = model_config.get("generation_method", "beam")
        self.num_captions = model_config.get("num_captions", 5)
        self.beam_group_size = model_config.get("beam_group_size", 2)
        self.beam_size = model_config.get(
            "beam_size", self.num_captions * self.beam_group_size
        )

    @torch.no_grad()
    def get_predictions(
        self,
        data_loader: DataLoader,
    ) -> pd.DataFrame:
        aaseq_indices = []
        generated_captions = []
        for model_inputs in tqdm(data_loader):
            model_inputs = move_inputs_to_device(model_inputs, self.device)
            _, _, _, captions = self.model.generate(
                model_inputs,
                max_len=self.max_len,
                aaseq_type=data_loader.dataset.aaseq_type,
                return_all_internals=False,
                method=self.method,
                beam_size=self.beam_size,
                beam_group_size=self.beam_group_size,
                truncate_on_eos=True,
            )
            for i, indices in enumerate(
                model_inputs["reference_indices"]["input"]["seq"]
            ):
                aaseq_index = indices[-1]

                for j in range(self.num_captions):
                    aaseq_indices.append(aaseq_index)
                    generated_captions.append(captions[i][(j * self.beam_group_size)])

        return pd.DataFrame(
            {
                "seq_id": aaseq_indices,
                "generated_caption": generated_captions,
            }
        )


class ProcyonQAEval(AbstractQAModel):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        checkpoint_dir = model_config["checkpoint_dir"]
        model, checkpoint_model_args = UnifiedProCyon.from_pretrained(
            pretrained_weights_dir=DEFAULT_PRETRAINED_WEIGHTS_DIR,
            checkpoint_dir=checkpoint_dir,
        )
        compare_and_warn_model_args(model_args, checkpoint_model_args)

        model.eval()
        model.bfloat16()
        self.device = device
        self.model = model.to(self.device)
        self.model_args = model_args
        self.checkpoint_dir = checkpoint_dir
        self.num_samples = eval_args.qa_num_samples
        self.rng = np.random.default_rng(seed=eval_args.seed)

        self.yes_token = model.yes_token
        self.no_token = model.no_token

    @torch.no_grad()
    def get_predictions(
        self,
        data_loader: DataLoader,
        aaseq_type: str = "protein",
    ) -> Dict[str, torch.Tensor]:

        results_dict = defaultdict(list)

        samples_to_hit = None
        if self.num_samples is not None:
            if self.num_samples < len(
                data_loader
            ):  # Keep None if we don't need to downsample
                samples_to_hit = self.rng.choice(
                    np.arange(len(data_loader)),
                    size=self.num_samples,
                    replace=False,
                )
                samples_to_hit = set(samples_to_hit)  # Faster lookup

        # Where the index of the actual text query is depends on whether or not
        # this is a dataset with context augmentation.
        no_context_aug = data_loader.collate_fn._get_input_contexts([], []) is None
        if no_context_aug:
            query_text_idx = -1
        else:
            query_text_idx = -2

        for i, model_inputs in enumerate(tqdm(data_loader)):
            if samples_to_hit is not None:
                if not (i in samples_to_hit):
                    continue
            out = self.model(
                move_inputs_to_device(model_inputs, self.device),
                return_mlm=False,
                retrieval=False,
                get_full_labels=True,
                aaseq_type=aaseq_type,
                crop_off=True,  # OWEN: HARDCODED - COULD REMOVE LATER OR MAKE AN OPTION
            )

            seq_ids = [x[-1] for x in model_inputs["reference_indices"]["input"]["seq"]]
            text_ids = [
                x[query_text_idx]
                for x in model_inputs["reference_indices"]["input"]["text"]
            ]

            # Workaround because in eval, we take out the yes/no in the instructions
            y_text = model_inputs["target"]["text"]
            y_toks = torch.LongTensor(
                [(self.yes_token if y == "yes" else self.no_token) for y in y_text]
            )

            pred_toks, _ = get_qa_scores(out, answer_token=self.model.answer_idx)

            results_dict["seq_ids"].extend(seq_ids)
            results_dict["text_ids"].extend(text_ids)
            results_dict["pred"].append(pred_toks)
            results_dict["y"].append(y_toks)

        results_dict["pred"] = torch.cat(results_dict["pred"])
        results_dict["y"] = torch.cat(results_dict["y"])

        return results_dict


class ProcyonRetrievalEval(AbstractRetrievalModel):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        checkpoint_dir = model_config["checkpoint_dir"]
        model, checkpoint_model_args = UnifiedProCyon.from_pretrained(
            pretrained_weights_dir=DEFAULT_PRETRAINED_WEIGHTS_DIR,
            checkpoint_dir=checkpoint_dir,
            strict_load=False,  # Must set to false because we don't store non-tuned weights
        )
        compare_and_warn_model_args(model_args, checkpoint_model_args)

        model.eval()
        model.bfloat16()
        self.device = device
        self.model = model.to(self.device)
        self.model_args = model_args
        self.batch_size = eval_args.batch_size
        self.checkpoint_dir = checkpoint_dir
        self.use_cached_target_embeddings = (
            eval_args.retrieval_use_cached_target_embeddings
        )

    @torch.no_grad()
    def _get_query_embeddings(
        self,
        query_loader: DataLoader,
        query_order: List,
    ) -> torch.Tensor:
        if isinstance(query_loader.dataset, AASeqTextUnifiedDataset):
            is_ppi = False
        elif isinstance(query_loader.dataset, AASeqDataset):
            is_ppi = True
        else:
            raise ValueError(f"unexpected dataset type: {type(query_loader.dataset)}")

        query_embeddings = []
        query_ids = []
        i = 0
        for model_inputs in tqdm(query_loader):
            # Zero out seq targets to prevent calculating unnecessary target
            # embeddings (only used for loss calculation at train time)
            model_inputs["target"]["seq"] = None
            model_inputs = move_inputs_to_device(model_inputs, self.device)

            # Where the query ID is stored depending on whether the query
            # is text or sequence (i.e. PPI)
            if is_ppi:
                query_ids += [
                    x[-1] for x in model_inputs["reference_indices"]["input"]["seq"]
                ]
            else:
                query_ids += [
                    x[-1] for x in model_inputs["reference_indices"]["input"]["text"]
                ]

            out = self.model(
                model_inputs,
                retrieval=True,
                aaseq_type=query_loader.dataset.aaseq_type,
            )
            query_embeddings.append(
                out["contrastive_out"]["positive"]["text"].detach().clone().cpu()
            )

            # i += 1
            # if i > 20:
            #     break

        # NOTE: Since the dataloaders are operating on (query, target) relations
        # rather than just queries, there may be multiple examples for a given query
        # (i.e. if a query has multiple relations). The below code has the effect
        # of just using the embedding from the last occurence of each query.
        # Theoretically the embeddings should be similar for each occurence, but
        # likely differ slightly due to potentially different in-context examples
        # provided.
        query_idxs = {query_id: idx for idx, query_id in enumerate(query_ids)}
        rearrange_idxs = [query_idxs[query_id] for query_id in query_order]
        query_embeddings = torch.cat(query_embeddings, dim=0)[rearrange_idxs]
        return query_embeddings

    @torch.no_grad()
    def _calculate_target_embeddings(
        self,
        target_loader: DataLoader,
        collate_fn: Callable,
        aaseq_type="protein",
    ) -> Tuple[torch.Tensor, List]:
        target_protein_embeddings = []
        target_ids = []
        # This is a loader that just provides a list of integer IDs
        # for each protein.
        for protein_ids in tqdm(target_loader):
            # In the case where the model is expecting the tokenized protein sequence
            # (instead of just the ID), we need to perform that conversion.
            # Somewhat hacky using the collator in the query loader,
            # can we move this out to a separate object at some point?
            # I think this may also mean we're storing all the raw protein sequences
            # once for each instantiated collator fxn.
            if not self.model.config.use_aaseq_embeddings:
                model_inputs = collate_fn._convert_batch("sequence", protein_ids)
            else:
                model_inputs = protein_ids
            model_inputs = move_inputs_to_device(model_inputs, self.device)
            target_ids += protein_ids.tolist()

            out = self.model.forward_sequences(model_inputs, aaseq_type=aaseq_type)
            target_protein_embeddings.append(out["shared"].detach().clone().cpu())

        target_protein_embeddings = torch.cat(target_protein_embeddings, dim=0)
        return target_protein_embeddings, target_ids

    def _get_cached_target_embeddings(
        self,
        collate_fn: Callable,
        aaseq_type: str,
    ):
        print("loading cached target embeddings")
        target_embeddings_path = os.path.join(
            self.checkpoint_dir, f"{aaseq_type}_target_embeddings.pkl"
        )
        if not os.path.exists(target_embeddings_path):
            print(
                f"retrieval_use_cached_target_embeddings is set to True but cached "
                f"embeddings not found, calculating and writing to: {target_embeddings_path}"
            )
            all_targets = get_retrieval_target_set(
                None,
                {},
                EvalArgs(retrieval_eval_all_aaseqs=True),
                aaseq_type=aaseq_type,
            )
            target_loader = get_retrieval_target_proteins_loader(
                all_targets,
                self.batch_size,
            )

            target_protein_embeddings, target_ids = self._calculate_target_embeddings(
                target_loader, collate_fn, aaseq_type=aaseq_type
            )
            with open(target_embeddings_path, "wb") as fh:
                torch.save((target_protein_embeddings, target_ids), fh)
        else:
            target_protein_embeddings, target_ids = torch.load(target_embeddings_path)
        return target_protein_embeddings, target_ids

    @torch.no_grad()
    def _get_target_embeddings(
        self,
        target_loader: DataLoader,
        target_order: List,
        collate_fn: Callable,
        aaseq_type: str,
    ) -> torch.Tensor:
        if self.use_cached_target_embeddings:
            target_protein_embeddings, target_ids = self._get_cached_target_embeddings(
                collate_fn, aaseq_type
            )
        else:
            target_protein_embeddings, target_ids = self._calculate_target_embeddings(
                target_loader, collate_fn, aaseq_type=aaseq_type
            )

        # Rearrange to expected order and/or subset down to just the targets of interest.
        target_idxs = {target_id: idx for idx, target_id in enumerate(target_ids)}
        rearrange_idxs = [target_idxs[target_id] for target_id in target_order]
        target_protein_embeddings = target_protein_embeddings[rearrange_idxs]
        return target_protein_embeddings

    @torch.no_grad()
    def get_predictions(
        self,
        query_loader: DataLoader,
        target_loader: DataLoader,
        query_order: List,
        target_order: List,
    ) -> torch.Tensor:
        # Get embeddings of queries (could be text or proteins).
        query_embeddings = self._get_query_embeddings(query_loader, query_order)

        # Get embeddings of set of target proteins.
        target_embeddings = self._get_target_embeddings(
            target_loader,
            target_order,
            query_loader.collate_fn,
            query_loader.dataset.aaseq_type,
        )

        # Normalize embeddings and calculate cosine similarities.
        query_embs_normalized = F.normalize(query_embeddings)
        target_embs_normalized = F.normalize(target_embeddings)
        # Dims like: num_queries X num_targets
        sims = query_embs_normalized @ target_embs_normalized.T

        return sims.detach().cpu().to(torch.float64)
