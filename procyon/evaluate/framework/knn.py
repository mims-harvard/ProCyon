import gzip
import os
import pickle
from typing import (
    Dict,
    List,
    Union,
)

import torch

import numpy as np
import pandas as pd
import torch.nn.functional as F


from torch.utils.data import DataLoader
from tqdm import tqdm

from procyon.data.data_utils import DATA_DIR
from procyon.data.dataset import (
    AASeqDataset,
    AASeqTextUnifiedDataset,
)
from procyon.evaluate.framework.args import EvalArgs
from procyon.evaluate.framework.retrieval import prep_for_retrieval_eval
from procyon.evaluate.framework.utils import (
    extract_qa_data,
    get_dataset_alternate_splits,
    optimal_qa_thresh_acc,
)
from procyon.training.training_args_IT import ModelArgs

embedding_map = {
    "esm2": "esm2-3b_mean.pt",
    "esm2-650m": "esm2-650m_mean.pt",
    "esm2-3b": "esm2-3b_mean.pt",
    "esm3": "esm3-sm-open-v1_mean.pt",
    "gearnet": "gearnet.pt",
}


class BaseKnnModel:
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        self.train_splits = model_config.get("train_splits", ["CL_train"])
        self.embed_type = model_config["embed_type"]
        assert self.embed_type in [
            "esm2",
            "esm2-650m",
            "esm2-3b",
            "esm3",
            "gearnet",
            "blast",
        ]
        if self.embed_type == "esm2":
            self.embed_type = "esm2-3b"
        # For PPI queries, whether or not to disallow a protein
        # for having itself as one of its nearest neighbors.
        # (Probably the only reason to turn this off would be
        # something like protein <-> domain relations)
        self.remove_self = model_config.get("remove_self", True)

        # Whether to raise an exception if the test set contains texts that
        # weren't observed in the train set. Since the kNN can't perform zero-shot
        # inference, the alternative is setting these to false and ignoring these
        # texts.
        self.filter_zero_shot = model_config.get("filter_zero_shot", False)

        # Note that currently all of our metrics (cosine sim or blast score) want the
        # largest value to get the nearest neighbors, but this may change in the
        # future (e.g. Euclidean distance)
        self.want_largest = True

        self.top_k = model_config.get("k", 10)
        self.loaded = False

        self.yes_token = 1
        self.no_token = 0

    def _init_label_mat(
        self,
        query_dataset: Union[AASeqTextUnifiedDataset, AASeqDataset],
    ):
        # First need to construct the corresponding train dataset for
        # the given query set.
        train_dataset = get_dataset_alternate_splits(query_dataset, self.train_splits)
        if isinstance(query_dataset, AASeqTextUnifiedDataset):
            target_ids = train_dataset.unique_aaseq

        elif isinstance(query_dataset, AASeqDataset):
            target_ids = train_dataset.all_aaseqs

        else:
            raise ValueError(f"unexpected dataset type: {type(query_dataset)}")
        label_matrix, query_order, target_order = prep_for_retrieval_eval(
            train_dataset,
            target_ids=target_ids,
            filter_training=False,
        )
        # Want to transpose: (texts X proteins) -> (proteins X texts)
        self.label_matrix = label_matrix.T
        self.aaseq_id_order = target_order
        self.aaseq_id_to_idx = {aaseq_id: i for i, aaseq_id in enumerate(target_order)}

        self.text_id_to_idx = {text_id: i for i, text_id in enumerate(query_order)}

        self.aaseq_type = query_dataset.aaseq_type

    def _init_embeds(
        self,
    ):
        if self.embed_type == "blast":
            scores_path = os.path.join(
                DATA_DIR,
                "generated_data",
                "baselines",
                "blast_scores",
                "blastp_defaults_max_ev10.pkl.gz"
            )
            with gzip.open(scores_path, "rb") as fh:
                # torch.topk treats NaNs as greater than all other values,
                # so want to set to -1 so we can still get the highest EV hits.
                self.scores = torch.nan_to_num(
                    torch.tensor(pd.read_pickle(fh).values), nan=-1
                )
            self.scores = self.scores[:, self.aaseq_id_order]
        else:
            aaseq_info_path = os.path.join(
                DATA_DIR,
                "integrated_data",
                "v1",
                self.aaseq_type,
                f"{self.aaseq_type}_info_filtered.pkl",
            )
            embeds_path = os.path.join(
                DATA_DIR,
                "generated_data",
                "node_embeddings",
                self.aaseq_type,
                f"{self.aaseq_type}_{embedding_map[self.embed_type]}",
            )
            id_map_path = os.path.join(
                DATA_DIR,
                "generated_data",
                "node_embeddings",
                self.aaseq_type,
                f"{self.aaseq_type}_{embedding_map[self.embed_type].replace('.pt', '.pkl')}",
            )

            self.embeds = F.normalize(torch.load(embeds_path))
            with open(id_map_path, "rb") as fh:
                id_order_map = {x.split()[0]: i for i, x in enumerate(pickle.load(fh))}
            expected_aaseq_order = pd.read_pickle(aaseq_info_path)[
                f"{self.aaseq_type}_id"
            ].tolist()
            reorder_idxs = [id_order_map[x] for x in expected_aaseq_order]
            self.embeds = self.embeds[reorder_idxs]

            # targets X embed
            self.train_embeds = self.embeds[self.aaseq_id_order]

    def load_data(
        self,
        dataset: Union[AASeqTextUnifiedDataset, AASeqDataset],
    ):
        self._init_label_mat(dataset)
        self._init_embeds()
        self.loaded = True

    def get_knns(
        self,
        query_protein_idxs: List[int],
    ):
        if not self.loaded:
            raise ValueError("need to call `load_data` before `get_knns`")

        if self.embed_type == "blast":
            distances = self.scores[query_protein_idxs]
        else:
            # query X embed
            query_embeds = self.embeds[query_protein_idxs]
            # query X target
            distances = query_embeds @ self.train_embeds.T
        if self.remove_self:
            for i, query_idx in enumerate(query_protein_idxs):
                if query_idx in self.aaseq_id_to_idx:
                    distances[i, self.aaseq_id_to_idx[query_idx]] = -1

        return distances.topk(k=self.top_k, dim=-1, largest=self.want_largest)


class KnnQAEval(BaseKnnModel):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        super(KnnQAEval, self).__init__(model_config, eval_args, model_args, device)

    def calc_results(
        self,
        data_loader: DataLoader,
    ):
        # Given the full set of query <-> target pairs, we can then
        # compute the query (text) probs for each target (protein)
        # using the kNN, and extract the appropriate prob for each
        # query <-> target pair.
        results_dict = {"pred": [], "y": []}

        aaseq_text_pairs, ground_truth = extract_qa_data(data_loader)
        all_aaseqs = [x[0] for x in aaseq_text_pairs]

        # nn_idxs is (num_proteins X k), giving idxs of each protein's top-k nearest neighbors
        _, nn_idxs = self.get_knns(all_aaseqs)
        # inferred_labels goes (num_proteins X k X num_texts) -> (num_proteins X num_texts) giving
        # the average over the one-hot encoded relations of each protein's neighbors
        inferred_labels = self.label_matrix[nn_idxs].mean(dim=1)

        mapped_text_idxs = []
        ground_truth_filt = []
        for i, (_, text_id) in enumerate(aaseq_text_pairs):
            if text_id in self.text_id_to_idx:
                mapped_text_idxs.append(self.text_id_to_idx[text_id])
                ground_truth_filt.append(ground_truth[i])
            else:
                if not self.filter_zero_shot:
                    raise ValueError(
                        f"KnnQAEval: test set contained text ID not observed in train set: {text_id}"
                    )
                # else do nothing, i.e. skip this pair
        print(
            f"KnnQaEval: filtered {len(ground_truth) - len(ground_truth_filt)} / {len(ground_truth)} "
            "due to absence from train set."
        )
        preds = np.array(
            [
                inferred_labels[i][mapped_text_idx]
                for i, (mapped_text_idx) in enumerate(mapped_text_idxs)
            ]
        )
        ground_truth = np.array(ground_truth_filt)

        best_thresh, best_acc = optimal_qa_thresh_acc(preds, ground_truth)
        print(
            f"KnnQAEval: best thresh of {best_thresh:0.3f} gives acc of {best_acc:0.3f}"
        )

        results_dict["pred"] = torch.tensor(
            np.where(preds >= best_thresh, self.yes_token, self.no_token)
        )
        results_dict["y"] = torch.tensor(
            np.where(ground_truth == "yes", self.yes_token, self.no_token)
        )
        return results_dict

    def get_predictions(
        self,
        data_loader: DataLoader,
        aaseq_type: str = "protein",
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(data_loader.dataset, AASeqTextUnifiedDataset):
            raise ValueError(f"unexpected dataset type: {type(data_loader.dataset)}")

        self.load_data(data_loader.dataset)
        return self.calc_results(data_loader)


class KnnRetrievalEval(BaseKnnModel):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        super(KnnRetrievalEval, self).__init__(
            model_config, eval_args, model_args, device
        )

    def get_predictions(
        self,
        query_loader: DataLoader,
        target_loader: DataLoader,
        query_order: List,
        target_order: List,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(query_loader.dataset, AASeqTextUnifiedDataset):
            input_key = "text"
        elif isinstance(query_loader.dataset, AASeqDataset):
            input_key = "seq"
        else:
            raise ValueError(f"unexpected dataset type: {type(query_loader.dataset)}")

        self.load_data(query_loader.dataset)

        query_ids = []
        for batch in tqdm(query_loader):
            input_idxs = [x[-1] for x in batch["reference_indices"]["input"][input_key]]
            query_ids += input_idxs

        target_ids = []
        # This is a loader that just provides a list of integer IDs
        # for each protein.
        for protein_ids in tqdm(target_loader):
            target_ids += protein_ids.tolist()

        # nn_idxs is (num_proteins X k), giving idxs of each protein's top-k nearest neighbors
        _, nn_idxs = self.get_knns(target_ids)
        # inferred_labels goes (num_proteins X k X num_texts) -> (num_proteins X num_texts) giving
        # the average over the one-hot encoded relations of each protein's neighbors, then transpose
        # to get (num_texts X num_proteins), i.e. queries X targets
        inferred_labels = self.label_matrix[nn_idxs].mean(dim=1).T

        aaseq_id_to_idx = {aaseq_idx: i for i, aaseq_idx in enumerate(target_ids)}
        aaseq_order = [aaseq_id_to_idx[idx] for idx in target_order]

        text_order = []
        nan_rows = []
        for i, query_id in enumerate(query_order):
            if query_id in self.text_id_to_idx:
                text_order.append(self.text_id_to_idx[query_id])
            else:
                if not self.filter_zero_shot:
                    raise ValueError(
                        f"KnnRetrievalEval: test set contained query ID not observed in train set: {query_id}"
                    )
                # Else we need to put in a placeholder row, and mark this row for filling with NaNs
                text_order.append(0)
                nan_rows.append(i)
        print(
            f"KnnRetrievalEval: filtered {len(nan_rows)} / {len(query_order)} queries due to absence from train set"
        )
        ret = inferred_labels[text_order][:, aaseq_order]
        ret[nan_rows] = np.nan

        return ret


class ESMKnnQAEval(KnnQAEval):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        model_config["embed_type"] = "esm2"
        super(ESMKnnQAEval, self).__init__(model_config, eval_args, model_args, device)


class ESM3KnnQAEval(KnnQAEval):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        model_config["embed_type"] = "esm3"
        super(ESM3KnnQAEval, self).__init__(model_config, eval_args, model_args, device)


class GearNetKnnQAEval(KnnQAEval):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        model_config["embed_type"] = "gearnet"
        super(GearNetKnnQAEval, self).__init__(
            model_config, eval_args, model_args, device
        )


class BlastKnnQAEval(KnnQAEval):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        model_config["embed_type"] = "blast"
        super(BlastKnnQAEval, self).__init__(
            model_config, eval_args, model_args, device
        )


class ESMKnnRetrievalEval(KnnRetrievalEval):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        model_config["embed_type"] = "esm2"
        super(ESMKnnRetrievalEval, self).__init__(
            model_config, eval_args, model_args, device
        )


class ESM3KnnRetrievalEval(KnnRetrievalEval):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        model_config["embed_type"] = "esm3"
        super(ESM3KnnRetrievalEval, self).__init__(
            model_config, eval_args, model_args, device
        )


class GearNetKnnRetrievalEval(KnnRetrievalEval):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        model_config["embed_type"] = "gearnet"
        super(GearNetKnnRetrievalEval, self).__init__(
            model_config, eval_args, model_args, device
        )


class BlastKnnRetrievalEval(KnnRetrievalEval):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        model_config["embed_type"] = "blast"
        super(BlastKnnRetrievalEval, self).__init__(
            model_config, eval_args, model_args, device
        )
