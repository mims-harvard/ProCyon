import os
import unittest

from random import sample
from typing import (
    Dict,
    List,
    Tuple,
)

import torch

import pandas as pd
import numpy as np

from torch.nn.functional import normalize
from sklearn.datasets import make_blobs

from procyon.data.dataset import (
    AASeqDataset,
    AASeqTextUnifiedDataset,
)
from procyon.data.data_utils import DATA_DIR
from procyon.evaluate.framework.args import EvalArgs
from procyon.evaluate.framework.knn import KnnQAEval
from procyon.evaluate.framework.metrics import precision_recall_topk
from procyon.evaluate.framework.retrieval import (
    calc_and_plot_auroc_auprc,
    get_retrieval_target_set,
    prep_for_retrieval_eval,
)

ALL_PROTEINS_FILE = os.path.join(
    DATA_DIR, "integrated_data/v1/protein/protein_info_filtered.pkl"
)

# Can run these tests from the command line as follows:
#   python -m unittest -v procyon/evaluate/framework/testing.py
# from the root of the repo


class TestPrecisionRecallTopK(unittest.TestCase):
    def test_correct(self):
        preds = np.array(
            [
                # Top-2 are correct.
                [0.5, 0.2, 0.1, 0.8],
                # None are correct.
                [0.2, 0.3, 0.0, 0.1],
                # All are correct.
                [0.2, 0.1, 0.7, 0.1],
                # Top and 3rd are correct.
                [0.2, 0.7, 0.4, 0.1],
            ]
        )

        labels = np.array(
            [
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 0, 0],
            ]
        )

        num_relevant = labels.sum(axis=1)
        expected_metrics_by_k = [
            # k = 1
            {
                "precision": (1 + 0 + 1 + 1) / 4,
                "recall": np.nanmean(np.nan_to_num([1, 0, 1, 1] / num_relevant)),
            },
            # k = 2
            {
                "precision": (2 + 0 + 2 + 1) / 8,
                "recall": np.nanmean(np.nan_to_num([2, 0, 2, 1] / num_relevant)),
            },
            # k = 3
            {
                "precision": (2 + 0 + 3 + 2) / 12,
                "recall": np.nanmean(np.nan_to_num([2, 0, 3, 2] / num_relevant)),
            },
            # k = 4
            {
                "precision": (2 + 0 + 4 + 2) / 16,
                "recall": np.nanmean(np.nan_to_num([2, 0, 4, 2] / num_relevant)),
            },
        ]
        for k in range(len(expected_metrics_by_k)):
            with self.subTest(i=k):
                got_precision, got_recall = precision_recall_topk(labels, preds, k + 1)
                expect_precision = expected_metrics_by_k[k]["precision"]
                expect_recall = expected_metrics_by_k[k]["recall"]

                self.assertAlmostEqual(
                    got_precision,
                    expect_precision,
                    msg=f"precision: k={k}, want {expect_precision: 0.3f} , got {got_precision: 0.3f}",
                )
                self.assertAlmostEqual(
                    got_recall,
                    expect_recall,
                    msg=f"recall: k={k}, want {expect_recall: 0.3f} , got {got_recall: 0.3f}",
                )

    def test_bad_labels(self):
        preds = np.array(
            [
                [0.2, 0.1],
                [0.2, 0.7],
            ]
        )

        labels = np.array(
            [
                [1, 0],
                [1, 2],
            ]
        )
        self.assertRaises(ValueError, precision_recall_topk, labels, preds, 1)


class TestAurocAuprc(unittest.TestCase):
    def test_correct(self):
        preds = torch.Tensor(
            [
                # Completely separable -> AUC = 1, AUPRC = 1
                [0.5, 0.2, 0.1, 0.8],
                # Completely inseparable -> AUC = 0.25, AUPRC = 0.5
                [0.2, 0.3, 0.0, 0.1],
                # Completely reversed-> AUC = 0, AUPRC = 0.5
                [0.2, 0.1, 0.7, 0.1],
                # Partially separable -> AUC = 0.75, AUPRC = 5/6
                [0.2, 0.7, 0.4, 0.1],
            ]
        )

        labels = torch.Tensor(
            [
                [1, 0, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 1, 0, 0],
            ]
        )
        expected_metrics_per_query = {
            "auroc": [1.0, 0.25, 0.0, 0.75],
            "auprc": [1.0, 0.5, 0.5, 5 / 6],
        }
        with self.subTest("averaged over queries"):
            got_auroc, got_auprc, _, _ = calc_and_plot_auroc_auprc(
                preds, labels, None, True
            )
            expect_auroc = np.mean(expected_metrics_per_query["auroc"])
            expect_auprc = np.mean(expected_metrics_per_query["auprc"])

            self.assertAlmostEqual(
                got_auroc,
                expect_auroc,
                msg=f"precision: want {expect_auroc: 0.3f} , got {got_auroc: 0.3f}",
            )
            self.assertAlmostEqual(
                got_auprc,
                expect_auprc,
                msg=f"recall: want {expect_auprc: 0.3f} , got {got_auprc: 0.3f}",
            )

        with self.subTest("combined over queries"):
            got_auroc, got_auprc, _, _ = calc_and_plot_auroc_auprc(
                preds, labels, None, False
            )
            expect_auroc = 0.5234375
            expect_auprc = 0.60625

            self.assertAlmostEqual(
                got_auroc,
                expect_auroc,
                f"precision: want {expect_auroc: 0.3f} , got {got_auroc: 0.3f}",
            )
            self.assertAlmostEqual(
                got_auprc,
                expect_auprc,
                f"recall: want {expect_auprc: 0.3f} , got {got_auprc: 0.3f}",
            )

        with self.subTest("with NaN entries"):
            nans = torch.full_like(preds, fill_value=float("nan"))
            nrows, ncols = preds.shape
            mod_preds = torch.stack((preds, nans), dim=2).view(nrows, ncols * 2)
            mod_labels = torch.stack((labels, nans), dim=2).view(nrows, ncols * 2)

            got_auroc, got_auprc, _, _ = calc_and_plot_auroc_auprc(
                mod_preds, mod_labels, None, True
            )
            expect_auroc = np.mean(expected_metrics_per_query["auroc"])
            expect_auprc = np.mean(expected_metrics_per_query["auprc"])

            self.assertAlmostEqual(
                got_auroc,
                expect_auroc,
                msg=f"precision: want {expect_auroc: 0.3f} , got {got_auroc: 0.3f}",
            )
            self.assertAlmostEqual(
                got_auprc,
                expect_auprc,
                msg=f"recall: want {expect_auprc: 0.3f} , got {got_auprc: 0.3f}",
            )


class DummyCollator:
    def __init__(self):
        pass

    def _get_input_contexts(self, a, b):
        return None


class DummyLoader(list):
    def __init__(self):
        self.collate_fn = DummyCollator()


class DummyAASeqTextDataset(AASeqTextUnifiedDataset):
    def __init__(
        self,
        aaseq_text_relations: List[Tuple[int, int]],
        eval_args: Dict = {},
    ):
        aaseq_ids = [x[0] for x in aaseq_text_relations]
        self.all_aaseqs = aaseq_ids
        self.unique_aaseq = np.unique(aaseq_ids)

        text_ids = [x[1] for x in aaseq_text_relations]
        self.all_texts = text_ids
        self.unique_text = np.unique(text_ids)

        # Add a dummy relation ID
        self.aaseq_text_relations = [(x[0], 0, x[1]) for x in aaseq_text_relations]
        self.eval_args = eval_args
        self.true_relations = self.aaseq_text_relations

    def name(self):
        return "dummy_aaseq_text"


class DummyAASeqDataset(AASeqDataset):
    def __init__(
        self,
        aaseq_relations: List[Tuple[int, int]],
        eval_args: Dict = {},
    ):
        aaseq_ids = [x[0] for x in aaseq_relations] + [x[1] for x in aaseq_relations]
        self.all_aaseqs = aaseq_ids
        self.unique_aaseq = np.unique(aaseq_ids)

        # Add a dummy relation ID
        self.aaseq_relations = [(x[0], 0, x[1]) for x in aaseq_relations] + [
            (x[1], 0, x[0]) for x in aaseq_relations if x[0] != x[1]
        ]
        self.eval_args = eval_args

    def name(self):
        return "dummy_aaseq"


class TestGetTargetSet(unittest.TestCase):
    def setUp(self):
        self.num_proteins_all = len(pd.read_pickle(ALL_PROTEINS_FILE))
        # Relations are either [aaseq, aaseq] or [aaseq, text]
        self.example_relations = [
            [0, 0],
            [0, 1],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 3],
        ]
        self.test_path = "./tmp.pkl"

    def tearDown(self) -> None:
        if os.path.exists(self.test_path):
            os.remove(self.test_path)
        return super().tearDown()

    def test_all_proteins(self):
        dataset = DummyAASeqTextDataset(self.example_relations)
        eval_args = EvalArgs(retrieval_eval_all_aaseqs=True)
        got_ids = get_retrieval_target_set(dataset, {}, eval_args)
        want_ids = pd.Series(np.arange(self.num_proteins_all))

        pd.testing.assert_series_equal(got_ids, want_ids, check_index=False)

    def test_query_subset(self):
        dataset = DummyAASeqTextDataset(self.example_relations)
        eval_args = EvalArgs(retrieval_eval_all_aaseqs=False)
        got_ids = get_retrieval_target_set(dataset, {}, eval_args)
        want_ids = pd.Series(dataset.unique_aaseq)

        pd.testing.assert_series_equal(got_ids, want_ids, check_index=False)

    def test_specified_subset(self):
        dataset = DummyAASeqTextDataset(self.example_relations)
        eval_args = EvalArgs(retrieval_eval_all_aaseqs=True)
        subset = np.random.choice(
            np.arange(self.num_proteins_all),
            size=5,
            replace=False,
        )
        pd.DataFrame(index=subset).to_pickle(self.test_path)

        got_ids = get_retrieval_target_set(
            dataset, {"target_subset": self.test_path}, eval_args
        )
        want_ids = pd.Series(subset)

        pd.testing.assert_series_equal(got_ids, want_ids, check_index=False)

    def test_specified_subset_of_query(self):
        dataset = DummyAASeqTextDataset(self.example_relations)
        eval_args = EvalArgs(retrieval_eval_all_aaseqs=False)
        subset = np.random.choice(
            np.arange(4),
            size=2,
            replace=False,
        )
        pd.DataFrame(index=subset).to_pickle(self.test_path)

        got_ids = get_retrieval_target_set(
            dataset, {"target_subset": self.test_path}, eval_args
        )
        want_ids = pd.Series(subset)

        pd.testing.assert_series_equal(got_ids, want_ids, check_index=False)

    def test_specified_subset_mismatch_query(self):
        dataset = DummyAASeqTextDataset(self.example_relations)
        eval_args = EvalArgs(retrieval_eval_all_aaseqs=False)
        subset = np.random.choice(
            np.arange(self.num_proteins_all),
            size=5,
            replace=False,
        )
        pd.DataFrame(index=subset).to_pickle(self.test_path)

        self.assertRaises(
            ValueError,
            get_retrieval_target_set,
            dataset,
            {"target_subset": self.test_path},
            eval_args,
        )


class TestPrepForRetrievalEval(unittest.TestCase):
    def setUp(self):
        self.num_proteins_all = len(pd.read_pickle(ALL_PROTEINS_FILE))
        # Relations are either [aaseq, aaseq] or [aaseq, text]
        self.example_relations = [
            [0, 0],
            [0, 1],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 3],
        ]
        self.test_path = "./tmp.pkl"

    def tearDown(self) -> None:
        if os.path.exists(self.test_path):
            os.remove(self.test_path)
        return super().tearDown()

    def test_aaseq_text_query_set(self):
        """Test with aaseq <-> text relations where possible targets are only from query dataset."""
        # Rows are queries (aaseq), cols are targets (texts)
        expect_labels = np.array(
            [
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1],
            ]
        )

        relations = sample(self.example_relations, k=len(self.example_relations))

        dataset = DummyAASeqTextDataset(relations)
        eval_args = EvalArgs(retrieval_eval_all_aaseqs=False)
        target_ids = get_retrieval_target_set(dataset, dataset.eval_args, eval_args)

        got_labels, unique_queries, unique_targets = prep_for_retrieval_eval(
            dataset,
            target_ids,
            filter_training=False,
        )

        np.testing.assert_array_equal(
            got_labels,
            expect_labels,
            f"labels do not match, got:\n{got_labels}\nexpect:\n{expect_labels}",
        )
        np.testing.assert_array_equal(
            unique_queries,
            dataset.unique_text,
            f"queries: got:\n{unique_queries}\nexpect:\n{dataset.unique_text}",
        )
        np.testing.assert_array_equal(
            unique_targets,
            dataset.unique_aaseq,
            f"targets: got:\n{unique_targets}\nexpect:\n{dataset.unique_aaseq}",
        )

    def test_aaseq_text_all_proteins(self):
        """Test with aaseq <-> text relations where possible targets are all proteins."""
        # Rows are queries (text), cols are targets (aaseq)
        expect_labels = np.zeros((4, self.num_proteins_all))
        for seq_id, text_id in self.example_relations:
            expect_labels[text_id, seq_id] = 1

        relations = sample(self.example_relations, k=len(self.example_relations))

        dataset = DummyAASeqTextDataset(relations)
        eval_args = EvalArgs(retrieval_eval_all_aaseqs=True)
        target_ids = get_retrieval_target_set(dataset, dataset.eval_args, eval_args)

        got_labels, unique_queries, unique_targets = prep_for_retrieval_eval(
            dataset,
            target_ids,
            filter_training=False,
        )

        np.testing.assert_array_equal(
            got_labels,
            expect_labels,
            f"labels do not match, got:\n{got_labels}\nexpect:\n{expect_labels}",
        )
        np.testing.assert_array_equal(
            unique_queries,
            dataset.unique_text,
            f"queries: got:\n{unique_queries}\nexpect:\n{dataset.unique_text}",
        )
        np.testing.assert_array_equal(
            unique_targets,
            np.arange(self.num_proteins_all),
            f"targets: got:\n{unique_targets}\nexpect:\nnp.arange({self.num_proteins_all})",
        )

    def test_aaseq_text_specified_subset(self):
        """Test with aaseq <-> text relations where possible targets are a specified subset of proteins."""
        # Rows are queries (text), cols are targets (aaseq)
        subset_size = 100
        # We have 5 targets in our dummy relations, test will use
        # 4 of them + 96 other random aaseq IDs.
        subset = np.concatenate(
            (
                np.arange(4),
                np.random.choice(
                    np.arange(6, self.num_proteins_all),
                    size=subset_size - 4,
                    replace=False,
                ),
            )
        )
        pd.DataFrame(index=subset).to_pickle(self.test_path)

        expect_labels = np.zeros((4, subset_size))
        for seq_id, text_id in self.example_relations:
            if seq_id == 4:
                continue
            expect_labels[text_id, seq_id] = 1

        relations = sample(self.example_relations, k=len(self.example_relations))

        dataset = DummyAASeqTextDataset(
            relations, eval_args={"target_subset": self.test_path}
        )
        eval_args = EvalArgs(retrieval_eval_all_aaseqs=True)
        target_ids = get_retrieval_target_set(dataset, dataset.eval_args, eval_args)

        got_labels, unique_queries, unique_targets = prep_for_retrieval_eval(
            dataset,
            target_ids,
            filter_training=False,
        )

        sorted_targets = sorted(subset)
        np.testing.assert_array_equal(
            got_labels,
            expect_labels,
            f"labels do not match, got:\n{got_labels}\nexpect:\n{expect_labels}",
        )
        np.testing.assert_array_equal(
            unique_queries,
            dataset.unique_text,
            f"queries: got:\n{unique_queries}\nexpect:\n{dataset.unique_text}",
        )
        np.testing.assert_array_equal(
            unique_targets,
            sorted_targets,
            f"targets: got:\n{unique_targets}\nexpect:\n{sorted_targets})",
        )

    def test_aaseq_aaseq_subset(self):
        """Test with aaseq <-> aaseq relations where possible targets are only from query set."""
        # Rows are queries (aaseq), cols are targets (aaseq)
        expect_labels = np.array(
            [
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 1, 0],
            ]
        )

        relations = sample(self.example_relations, k=len(self.example_relations))

        dataset = DummyAASeqDataset(relations)
        eval_args = EvalArgs(retrieval_eval_all_aaseqs=False)
        target_ids = get_retrieval_target_set(dataset, dataset.eval_args, eval_args)

        got_labels, unique_queries, unique_targets = prep_for_retrieval_eval(
            dataset,
            target_ids,
            filter_training=False,
        )

        np.testing.assert_array_equal(
            got_labels,
            expect_labels,
            f"labels do not match, got:\n{got_labels}\nexpect:\n{expect_labels}",
        )
        np.testing.assert_array_equal(
            unique_queries,
            dataset.unique_aaseq,
            f"queries: got:\n{unique_queries}\nexpect:\n{dataset.unique_aaseq}",
        )
        np.testing.assert_array_equal(
            unique_targets,
            dataset.unique_aaseq,
            f"targets: got:\n{unique_targets}\nexpect:\n{dataset.unique_aaseq}",
        )

    def test_aaseq_aaseq_all_proteins(self):
        """Test with aaseq <-> aaseq relations where possible targets are all proteins."""
        # Rows are queries (aaseq), cols are targets (aaseq)
        expect_labels = np.zeros((5, self.num_proteins_all))
        for seq_id_1, seq_id_2 in self.example_relations:
            expect_labels[seq_id_1, seq_id_2] = 1
            expect_labels[seq_id_2, seq_id_1] = 1

        relations = sample(self.example_relations, k=len(self.example_relations))

        dataset = DummyAASeqDataset(relations)
        eval_args = EvalArgs(retrieval_eval_all_aaseqs=True)
        target_ids = get_retrieval_target_set(dataset, dataset.eval_args, eval_args)

        got_labels, unique_queries, unique_targets = prep_for_retrieval_eval(
            dataset,
            target_ids,
            filter_training=False,
        )

        np.testing.assert_array_equal(
            got_labels,
            expect_labels,
            f"labels do not match, got:\n{got_labels}\nexpect:\n{expect_labels}",
        )
        np.testing.assert_array_equal(
            unique_queries,
            dataset.unique_aaseq,
            f"queries: got:\n{unique_queries}\nexpect:\n{dataset.unique_aaseq}",
        )
        np.testing.assert_array_equal(
            unique_targets,
            np.arange(self.num_proteins_all),
            f"targets: got:\n{unique_targets}\nexpect:\nnp.arange({self.num_proteins_all})",
        )

    def test_aaseq_aaseq_specified_subset(self):
        """Test with aaseq <-> aaseq relations where possible targets are a specified subset of proteins."""
        # Rows are queries (aaseq), cols are targets (aaseq)
        subset_size = 100
        # We have 5 targets in our dummy relations, test will use
        # 4 of them + 96 other random aaseq IDs. Note that since
        # relations are typically symmetrical with aaseq <-> aaseq,
        # this has the effect of breaking the symmetry (i.e. we'll have
        # the query ->  target relation 4 -> 3 but not 3 -> 4).
        subset = np.concatenate(
            (
                np.arange(4),
                np.random.choice(
                    np.arange(6, self.num_proteins_all),
                    size=subset_size - 4,
                    replace=False,
                ),
            )
        )
        pd.DataFrame(index=subset).to_pickle(self.test_path)

        expect_labels = np.zeros((5, subset_size))
        for seq_id_1, seq_id_2 in self.example_relations:
            if seq_id_2 != 4:
                expect_labels[seq_id_1, seq_id_2] = 1
            if seq_id_1 != 4:
                expect_labels[seq_id_2, seq_id_1] = 1

        relations = sample(self.example_relations, k=len(self.example_relations))

        dataset = DummyAASeqDataset(
            relations, eval_args={"target_subset": self.test_path}
        )
        eval_args = EvalArgs(retrieval_eval_all_aaseqs=True)
        target_ids = get_retrieval_target_set(dataset, dataset.eval_args, eval_args)

        got_labels, unique_queries, unique_targets = prep_for_retrieval_eval(
            dataset,
            target_ids,
            filter_training=False,
        )
        sorted_targets = sorted(subset)

        np.testing.assert_array_equal(
            got_labels,
            expect_labels,
            f"labels do not match, got:\n{got_labels}\nexpect:\n{expect_labels}",
        )
        np.testing.assert_array_equal(
            unique_queries,
            dataset.unique_aaseq,
            f"queries: got:\n{unique_queries}\nexpect:\n{dataset.unique_aaseq}",
        )
        np.testing.assert_array_equal(
            unique_targets,
            sorted_targets,
            f"targets: got:\n{unique_targets}\nexpect:\n{sorted_targets})",
        )


class TestKnnQAEval(unittest.TestCase):
    def test_positive_control(self):
        # First construct synthetic data consisting of
        # three different clusters in 2D, same labels/text
        # assocations for all samples in a given cluster.
        blob_centers = [(-10, 0), (10, 0), (0, 10)]

        # Positive assoications for each cluster.
        blob_labels = [
            [4],  # [0, 0, 0, 0, 1]
            [0, 1],  # [1, 1, 0, 0, 0]
            [2, 3],  # [0, 0, 1, 1, 0]
        ]

        # Store example negative labels per cluster.
        neg_blob_labels = [
            [0],  # [0, 0, 0, 0, 1]
            [2, 4],  # [1, 1, 0, 0, 0]
            [0, 1],  # [0, 0, 1, 1, 0]
        ]

        n_train_samples = 100
        n_test_samples = 20
        tot = (n_train_samples + n_test_samples) * len(blob_centers)

        n_samples = [n_train_samples + n_test_samples for _ in blob_centers]

        all_x, all_y = make_blobs(
            n_samples=n_samples,
            centers=blob_centers,
            random_state=42,
        )

        # Pre-normalize for cosine dist.
        all_x = normalize(torch.tensor(all_x))

        # Shuffle data and split into train/test
        shuffled_idxs = sample(range(tot), k=tot)
        train_idxs = shuffled_idxs[: n_train_samples * len(blob_centers)]
        test_idxs = shuffled_idxs[n_train_samples * len(blob_centers) :]

        train_x = all_x[train_idxs]
        train_y = all_y[train_idxs]

        test_x = all_x[test_idxs]
        test_y = all_y[test_idxs]

        # Create relationships for dummy training set and mock test data loader
        train_synth_relations = []
        for idx, label in zip(train_idxs, train_y):
            train_synth_relations.extend(
                [(idx, text_id) for text_id in blob_labels[label]]
            )

        test_pos_synth_relations = []
        test_neg_synth_relations = []
        for idx, label in zip(test_idxs, test_y):
            test_pos_synth_relations.extend(
                [(idx, text_id) for text_id in blob_labels[label]]
            )
            test_neg_synth_relations.extend(
                [(idx, text_id) for text_id in neg_blob_labels[label]]
            )

        # Make dataset and labels
        dataset = DummyAASeqTextDataset(train_synth_relations)
        eval_args = EvalArgs(retrieval_eval_all_aaseqs=False)
        target_ids = get_retrieval_target_set(dataset, dataset.eval_args, eval_args)

        got_labels, unique_queries, unique_targets = prep_for_retrieval_eval(
            dataset,
            target_ids,
            filter_training=False,
        )

        # Manually populate model to circumvent data loading that expects real datasets
        model = KnnQAEval({"embed_type": "esm2"}, None, None, None)

        model.label_matrix = got_labels.T
        model.aaseq_id_order = unique_targets
        model.text_id_to_idx = {text_id: i for i, text_id in enumerate(unique_queries)}
        model.embeds = all_x
        model.train_embeds = model.embeds[model.aaseq_id_order]
        model.loaded = True
        model.remove_self = False

        # Construct mock data loader for test set
        batch_size = 16
        num_batches = (len(test_pos_synth_relations) + batch_size - 1) // batch_size

        mock_loader = DummyLoader()
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(test_pos_synth_relations))
            pos_seq_ids, pos_text_ids = zip(*test_pos_synth_relations[start:end])
            neg_seq_ids, neg_text_ids = zip(*test_neg_synth_relations[start:end])

            all_seq_ids = pos_seq_ids + neg_seq_ids
            all_text_ids = pos_text_ids + neg_text_ids

            mock_loader.append(
                {
                    "reference_indices": {
                        "input": {
                            "seq": [(x,) for x in all_seq_ids],
                            "text": [(x,) for x in all_text_ids],
                        },
                    },
                    "target": {
                        "text": ["yes"] * len(pos_seq_ids) + ["no"] * len(neg_seq_ids),
                    },
                }
            )

        got = model.calc_results(mock_loader)
        got_acc = (got["pred"] == got["y"]).float().mean().item()

        self.assertAlmostEqual(got_acc, 1.0, msg=f"acc: want 1.0 , got {got_acc: 0.3f}")
