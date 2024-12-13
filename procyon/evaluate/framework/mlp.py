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

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from procyon.data.data_utils import DATA_DIR
from procyon.data.dataset import (
    AASeqDataset,
    AASeqTextUnifiedDataset,
)
from procyon.model.model_utils import create_mlp
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

def has_validation_split(
    dataset: Union[AASeqTextUnifiedDataset, AASeqDataset],
    validation_splits: List[str],
) -> bool:
    all_relations = pd.read_csv(os.path.join(DATA_DIR,
                                            "integrated_data",
                                            "v1",
                                            f"{dataset.aaseq_type}_{dataset.text_type}",
                                            dataset.text_split_method,
                                            f"{dataset.aaseq_type}_{dataset.text_type}_relations_indexed.unified.csv"))
    obs_splits = all_relations.split.unique()
    return pd.Series(validation_splits).isin(obs_splits).all()

class BaseMLPModel:
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        self.device = device
        self.train_splits = model_config.get("train_splits", ["CL_train"])
        self.val_splits = model_config.get("train_splits", ["CL_val_pt_ft"])

        self.embed_type = model_config["embed_type"]
        assert self.embed_type in ["esm2", "esm2-650m", "esm2-3b", "esm3", "gearnet"]
        if self.embed_type == "esm2":
            self.embed_type = "esm2-3b"

        # Whether to raise an exception if the test set contains texts that
        # weren't observed in the train set. Since the kNN can't perform zero-shot
        # inference, the alternative is setting these to false and ignoring these
        # texts.
        self.filter_zero_shot = model_config.get("filter_zero_shot", False)

        self.loaded = False

        self.checkpoint_dir = model_config["checkpoint_dir"]
        self.num_layers = model_config.get("num_layers", 2)
        self.hidden_dim = model_config.get("hidden_dim", 256)
        self.dropout_rate = model_config.get("dropout_rate", 0.25)
        self.learning_rate = model_config.get("learning_rate", 5e-4)
        self.batch_size = model_config.get("batch_size", 64)
        self.num_steps = model_config.get("num_steps", 2000)
        self.validation_steps = model_config.get("validation_steps", 50)
        self.num_steps_no_validation = model_config.get("num_steps_no_validation", 300)

        self.pos_weight = model_config.get("pos_weight", 1000)

        self.embeds = None

    def _init_data(
        self,
        query_dataset: Union[AASeqTextUnifiedDataset, AASeqDataset],
    ):
        assert self.embeds is not None, "Must init embeddings before training data"
        # First need to construct the corresponding train and validation datasets for
        # the given query set.
        train_dataset = get_dataset_alternate_splits(query_dataset, self.train_splits)
        if isinstance(query_dataset, AASeqTextUnifiedDataset):
            target_ids = train_dataset.unique_aaseq
        elif isinstance(query_dataset, AASeqDataset):
            target_ids = train_dataset.all_aaseqs
        else:
            raise ValueError(f"unexpected dataset type: {type(query_dataset)}")

        # Get training label matrix: texts X proteins
        label_matrix, query_order, target_order = prep_for_retrieval_eval(
            train_dataset,
            target_ids=target_ids,
            filter_training=False,
        )
        self.aaseq_id_to_idx = {aaseq_id: i for i, aaseq_id in enumerate(target_order)}
        self.text_id_to_idx = {text_id: i for i, text_id in enumerate(query_order)}

        self.train_label_matrix = label_matrix.T
        self.train_embeds = self.embeds[target_order]

        if has_validation_split(query_dataset, self.val_splits):
            val_dataset = get_dataset_alternate_splits(query_dataset, self.val_splits)
            joint_dataset = get_dataset_alternate_splits(query_dataset, self.train_splits + self.val_splits)
            if isinstance(query_dataset, AASeqTextUnifiedDataset):
                joint_target_ids = joint_dataset.unique_aaseq
            elif isinstance(query_dataset, AASeqDataset):
                joint_target_ids = joint_dataset.all_aaseqs
            else:
                raise ValueError(f"unexpected dataset type: {type(query_dataset)}")


            # To construct validation label matrix, we need to make sure all the same
            # labels are present, which isn't guaranteed since not all train texts are
            # also in the validation set.
            #
            # Workaround is to get the "joint" label matrix, which has all the same texts
            # (since we're not including zero shot validation) and a superset of proteins.
            # We then subset down to validation proteins, and zero out the positive labels
            # from the train set.
            joint_label_matrix, joint_query_order, joint_target_order = prep_for_retrieval_eval(
                joint_dataset,
                target_ids=joint_target_ids,
                filter_training=False,
            )
            val_aaseq_id_map = {aaseq_id: i for i, aaseq_id in enumerate(joint_target_order)}
            # Some validation texts may technically be zero shot, i.e. not in train, once subset to
            # a specfic relation, e.g. DrugBank drug carrier. Need to remove these and put the remaining
            # ones into the same order.
            joint_query_idx_map = {text_id : i for i, text_id in enumerate(joint_query_order)}
            reordered_joint_queries = [joint_query_idx_map[x] for x in query_order]
            joint_label_matrix = joint_label_matrix[reordered_joint_queries]

            _, _, val_target_order = prep_for_retrieval_eval(
                val_dataset,
                target_ids=joint_target_ids,
                filter_training=False,
            )

            # Get sets of proteins that overlap train set and those that don't.
            train_targets = [x for x in joint_target_order if x in target_order and x in val_target_order]
            val_targets = [x for x in joint_target_order if x not in target_order]

            # Indexes of validation proteins in the "joint" matrix, grab those rows.
            val_joint_idxs = [val_aaseq_id_map[x] for x in val_targets]
            pure_val_labels = joint_label_matrix[:, val_joint_idxs]

            # Indexes of validation proteins in the "joint" matrix that overlap the trainset,
            # grab those rows and zero out the training positive relations.
            val_train_idxs = [self.aaseq_id_to_idx[val_id] for val_id in train_targets]
            overlap_val_labels = (joint_label_matrix[:, val_train_idxs].bool() &
                                ~label_matrix[:, val_train_idxs].bool()).float()

            # Concat the two sets of validation labels constructed above.
            val_target_order = val_targets + train_targets
            val_label_matrix = torch.cat((pure_val_labels, overlap_val_labels), dim=1)[:, ]

            # Want to transpose: (texts X proteins) -> (proteins X texts)

            self.val_label_matrix = val_label_matrix.T

            self.val_embeds = self.embeds[val_target_order]
        else:
            # No validation splits.
            self.val_embeds = None
            self.num_steps = self.num_steps_no_validation
            print(f"dataset {query_dataset.name()} has no validation splits, "
                  "make sure to set `num_steps_no_validation` steps accordingly")

    def _init_embeds(
        self,
        query_dataset: Union[AASeqTextUnifiedDataset, AASeqDataset],
    ):
        aaseq_info_path = os.path.join(
            DATA_DIR,
            "integrated_data",
            "v1",
            query_dataset.aaseq_type,
            f"{query_dataset.aaseq_type}_info_filtered.pkl",
        )
        embeds_path = os.path.join(
            DATA_DIR,
            "generated_data",
            "node_embeddings",
            query_dataset.aaseq_type,
            f"{query_dataset.aaseq_type}_{embedding_map[self.embed_type]}",
        )
        id_map_path = os.path.join(
            DATA_DIR,
            "generated_data",
            "node_embeddings",
            query_dataset.aaseq_type,
            f"{query_dataset.aaseq_type}_{embedding_map[self.embed_type].replace('.pt', '.pkl')}",
        )

        self.embeds = torch.load(embeds_path)
        # Some embeddings were saved as tensors with requires_grad = True
        self.embeds.requires_grad_(False)
        self.embeds = F.normalize(self.embeds)
        with open(id_map_path, "rb") as fh:
            id_order_map = {x.split()[0]: i for i, x in enumerate(pickle.load(fh))}
        expected_aaseq_order = pd.read_pickle(aaseq_info_path)[f"{query_dataset.aaseq_type}_id"].tolist()
        reorder_idxs = [id_order_map[x] for x in expected_aaseq_order]
        self.embeds = self.embeds[reorder_idxs]

        self.aaseq_type = query_dataset.aaseq_type

    def _init_model(
        self,
    ):
        assert self.embeds is not None, "Must init embeddings before training data"

        in_features = self.embeds.shape[1]
        out_labels = self.train_label_matrix.shape[1]
        self.model = create_mlp(
            n_layers=self.num_layers,
            in_features=in_features,
            out_features=out_labels,
            hidden_features=self.hidden_dim,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

    @torch.no_grad()
    def _check_val_loss(
        self,
        embeds: torch.Tensor,
        labels: torch.Tensor,
    ):
        dataset = TensorDataset(embeds, labels)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        pos_weights = torch.full([self.train_label_matrix.shape[1]], fill_value=self.pos_weight).to(self.device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)

        all_preds = []
        total_loss = 0
        for batch_embeds, batch_labels in loader:
            batch_embeds = batch_embeds.to(self.device)
            batch_labels = batch_labels.to(self.device)
            preds = self.model(batch_embeds)
            loss = loss_fn(preds, batch_labels)

            total_loss += loss.item()
            all_preds.append(preds.detach().cpu())

        all_preds = torch.cat(all_preds, dim=0)
        pos_preds = all_preds[labels == 1]
        neg_preds = all_preds[labels == 0]
        check_preds = torch.cat((pos_preds, neg_preds))
        check_labels = torch.cat((
            torch.full_like(pos_preds, fill_value=1),
            torch.full_like(neg_preds, fill_value=0),
        ))

        return total_loss / len(loader), roc_auc_score(check_labels, check_preds)

    def run_preds(
        self,
        target_ids: List[int],
    ):
        embeds = self.embeds[target_ids]
        dataset = TensorDataset(embeds)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        all_preds = []
        with torch.no_grad():
            for (batch_embeds,) in loader:
                preds = self.model(batch_embeds.to(self.device))

                all_preds.append(preds.detach().cpu())

        return torch.cat(all_preds, dim=0)

    def _train(
        self,
    ):
        self.model.train()
        train_dataset = TensorDataset(self.train_embeds, self.train_label_matrix)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        # Strongly upweight positive examples, due to severe label imbalance.
        # Technically we may have the case that text X relates to protein Y, but that
        # relation is in a val or eval split, so that relation will be treated as a negative
        # example during training, but the positive labels are very sparse for
        # each protein, so this should wash out.
        pos_weights = torch.full([self.train_label_matrix.shape[1]], fill_value=self.pos_weight).to(self.device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        num_steps = 0
        metrics = []
        best_val_auc = None
        best_state = None
        best_step = None
        done = False
        while not done:
            for embeds, labels in train_loader:
                embeds = embeds.to(self.device)
                labels = labels.to(self.device)

                preds = self.model(embeds)
                loss = loss_fn(preds, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if num_steps % self.validation_steps == 0:
                    train_loss = loss.item()

                    iter_metrics = {
                        "step_num": num_steps,
                        "train_loss": train_loss,
                    }
                    msg = f"train loss: {train_loss:>7f}"
                    if self.val_embeds is not None:
                        self.model.eval()
                        val_loss, val_auc = self._check_val_loss(self.val_embeds, self.val_label_matrix)
                        self.model.train()
                        iter_metrics["val_loss"]= val_loss
                        iter_metrics["val_auc"]= val_auc

                        msg += f" val loss/AUC: {val_loss:>7f} {val_auc:>4f}"
                        if best_val_auc is None or val_auc > best_val_auc:
                            best_val_auc = val_auc
                            best_state = self.model.state_dict()
                            best_step = num_steps

                    metrics.append(iter_metrics)
                    print(f"{msg} [{num_steps}/{self.num_steps}]")

                num_steps += 1
                if num_steps == self.num_steps:
                    done = True
                    break
        if self.val_embeds is not None:
            print(f"best val AUC @ step {best_step}:  {best_val_auc:>4f}")
            self.model.load_state_dict(best_state)
        return pd.DataFrame(metrics)

    def load_data(
        self,
        dataset: Union[AASeqTextUnifiedDataset, AASeqDataset],
    ):
        self._init_embeds(dataset)
        self._init_data(dataset)
        self._init_model()

        checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.embed_type}.{dataset.name()}.mlp.pth")
        if os.path.exists(checkpoint_path):
            print(f"loading saved model from {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path))
            self.model.eval()
        else:
            self._train()
            print(f"saving trained model to {checkpoint_path}")
            torch.save(self.model.state_dict(), checkpoint_path)
        self.loaded = True

class MLPQAEval(BaseMLPModel):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        self.yes_token = 1
        self.no_token = 0

        super(MLPQAEval, self).__init__(model_config, eval_args, model_args, device)

    @torch.no_grad()
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

        sigmoid = torch.nn.Sigmoid()
        preds = sigmoid(self.run_preds(all_aaseqs))

        qa_probs = []
        ground_truth_filt = []
        for i, (_, text_id) in enumerate(aaseq_text_pairs):
            if text_id in self.text_id_to_idx:
                text_idx = self.text_id_to_idx[text_id]
                qa_probs.append(preds[i, text_idx])

                ground_truth_filt.append(ground_truth[i])
            else:
                if not self.filter_zero_shot:
                    raise ValueError(f"MLPQAEval: test set contained text ID not observed in train set: {text_id}")
                # else do nothing, i.e. skip this pair
        print(
            f"MLPQAEval: filtered {len(ground_truth) - len(ground_truth_filt)} / {len(ground_truth)} "
            "due to absence from train set."
        )

        qa_probs = np.array(qa_probs)
        ground_truth = np.array(ground_truth_filt)

        best_thresh, best_acc = optimal_qa_thresh_acc(qa_probs, ground_truth)
        print(f"MLPQAEval: best thresh of {best_thresh:0.3f} gives acc of {best_acc:0.3f}")

        results_dict["pred"] = torch.tensor(np.where(qa_probs >= best_thresh, self.yes_token, self.no_token))
        results_dict["y"] = torch.tensor(np.where(ground_truth == "yes", self.yes_token, self.no_token))
        return results_dict

    def get_predictions(
        self,
        data_loader: DataLoader,
        aaseq_type: str = 'protein',
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(data_loader.dataset, AASeqTextUnifiedDataset):
           raise ValueError(f"unexpected dataset type: {type(data_loader.dataset)}")

        self.load_data(data_loader.dataset)
        return self.calc_results(data_loader)

class MLPRetrievalEval(BaseMLPModel):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        super(MLPRetrievalEval, self).__init__(model_config, eval_args, model_args, device)

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

        preds = self.run_preds(target_ids)

        aaseq_id_to_idx = {aaseq_idx: i for i, aaseq_idx in enumerate(target_ids)}
        aaseq_order = [aaseq_id_to_idx[idx] for idx in target_order]

        text_order = []
        nan_rows = []
        for i, query_id in enumerate(query_order):
            if query_id in self.text_id_to_idx:
                text_order.append(self.text_id_to_idx[query_id])
            else:
                if not self.filter_zero_shot:
                    raise ValueError(f"MLPRetrievalEval: test set contained query ID not observed in train set: {query_id}")
                # Else we need to put in a placeholder row, and mark this row for filling with NaNs
                text_order.append(0)
                nan_rows.append(i)
        print(f"MLPRetrievalEval: filtered {len(nan_rows)} / {len(query_order)} queries due to absence from train set")

        # Predictions returned are protein X text -> transpose to text X protein
        ret = preds.T[text_order][:, aaseq_order]
        ret[nan_rows] = np.nan

        return ret

class ESMMLPQAEval(MLPQAEval):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        model_config["embed_type"] = "esm2"
        super(ESMMLPQAEval, self).__init__(model_config, eval_args, model_args, device)

class ESM3MLPQAEval(MLPQAEval):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        model_config["embed_type"] = "esm3"
        super(ESM3MLPQAEval, self).__init__(model_config, eval_args, model_args, device)

class GearNetMLPQAEval(MLPQAEval):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        model_config["embed_type"] = "gearnet"
        super(GearNetMLPQAEval, self).__init__(model_config, eval_args, model_args, device)

class ESMMLPRetrievalEval(MLPRetrievalEval):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        model_config["embed_type"] = "esm2"
        super(ESMMLPRetrievalEval, self).__init__(model_config, eval_args, model_args, device)

class ESM3MLPRetrievalEval(MLPRetrievalEval):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        model_config["embed_type"] = "esm3"
        super(ESM3MLPRetrievalEval, self).__init__(model_config, eval_args, model_args, device)

class GearNetMLPRetrievalEval(MLPRetrievalEval):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        model_config["embed_type"] = "gearnet"
        super(GearNetMLPRetrievalEval, self).__init__(model_config, eval_args, model_args, device)

class BlastMLPRetrievalEval(MLPRetrievalEval):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        model_config["embed_type"] = "blast"
        super(BlastMLPRetrievalEval, self).__init__(model_config, eval_args, model_args, device)