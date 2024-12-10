import os
import gzip

from typing import (
    Dict,
    List,
)

import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from procyon.data.data_utils import DATA_DIR
from procyon.data.dataset import (
    AASeqDataset,
)
from procyon.evaluate.framework.args import EvalArgs
from procyon.evaluate.framework.retrieval import (
    AbstractRetrievalModel,
)
from procyon.training.training_args_IT import ModelArgs


class BlastRetrievalEval(AbstractRetrievalModel):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        max_ev = model_config.get("max_ev", 10)
        blast_args = model_config.get("blast_args", "defaults")

        scores_path = os.path.join(
            DATA_DIR,
            "integrated_data",
            "blast",
            "baselines",
            "blast_scores",
            f"blastp_{blast_args}_max_ev{max_ev:d}.pkl.gz"
        )
        print(f"Loading BLAST scores from: {scores_path}")

        with gzip.open(scores_path, "rb") as fh:
            self.scores = pd.read_pickle(fh)

        # Whether to remove a protein's alignment to itself (otherwise top scoring hit
        # is always to itself).
        if model_config.get("remove_self", True):
            np.fill_diagonal(self.scores.values, np.nan)

    # Return a tensor of size num_queries x num_targets
    # where each value (i, j) is the score of target j
    # given query i.
    #
    # query_loader - DataLoader for queries (either text
    #                or protein)
    # target_loader - DataLoader for targets (proteins)
    # query_order: - List giving the expected order of
    #                queries in the returned tensor (i.e.
    #                row i of the returned tensor should
    #                correspond to query at query_order[i]).
    # target_order: - List giving the expected order of
    #                 targets in the returned tensor
    #                 (analogous to above but for columns and
    #                 targets).
    def get_predictions(
        self,
        query_loader: DataLoader,
        target_loader: DataLoader,
        query_order: List,
        target_order: List,
    ) -> torch.Tensor:
        if not isinstance(query_loader.dataset, AASeqDataset):
            raise ValueError("BlastRetrievalEval only works for PPI-style "
                             f"retrieval, got: {type(query_loader.dataset)}")
        query_order = pd.Series(query_order)
        target_order = pd.Series(target_order)

        missing_queries = query_order[~query_order.isin(self.scores.index)]
        missing_targets = target_order[~target_order.isin(self.scores.columns)]
        if len(missing_queries) != 0 or len(missing_targets) != 0:
            raise ValueError("Missing queries or targets in BLAST scores: \n"
                             f"queries: {missing_queries}\ntargets:{missing_targets}")

        return torch.tensor(self.scores.loc[query_order, target_order].values)
