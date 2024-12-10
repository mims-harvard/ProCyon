import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import (
    Callable,
    List,
    Optional,
    Tuple,
    Union,
)

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

from procyon.data.sampling import negative_sampling_random_tail
from procyon.data.data_utils import (
    process_go_sims,
    process_pfam_sims,
    process_protein_sims,
    process_domain_sims,
    process_aaseq_sims,
    process_text_sims,
    DATA_DIR,
    first_unique_value_in_pandas_df,
)

# TODO: check this
def get_X_go_relations(aaseq_type, split, go_split_method):
    if split in ['train', 'val']:
        split = f'CL_{split}'
    if 'v1' in go_split_method or split == 'eval':
        if split == 'CL_train':
            return pd.read_csv(DATA_DIR+f'integrated_data/{aaseq_type}_go/{go_split_method}/{aaseq_type}_go_relations_{split}_indexed.csv')[['seq_id', 'relation', 'text_id']]
        else:
            return pd.concat([
                pd.read_csv(DATA_DIR+f'integrated_data/{aaseq_type}_go/{go_split_method}/{aaseq_type}_go_relations_{split}_zero_shot_indexed.csv'),
                pd.read_csv(DATA_DIR+f'integrated_data/{aaseq_type}_go/{go_split_method}/{aaseq_type}_go_relations_{split}_five_shot_indexed.csv'),
                pd.read_csv(DATA_DIR+f'integrated_data/{aaseq_type}_go/{go_split_method}/{aaseq_type}_go_relations_{split}_pt_ft_indexed.csv')])[['seq_id', 'relation', 'text_id']]
    else:
        return pd.read_csv(DATA_DIR+f'integrated_data/{aaseq_type}_go/{go_split_method}/{aaseq_type}_go_relations_{split}_indexed.csv')[['seq_id', 'relation', 'text_id']]



def get_and_check_relation_id(
    data_dir: str,
    want_relation: str,
) -> int:
    """Load the integer ID for a given relation, erroring if does not exist.

    Loads the integer ID for a relation defined in
    '{data_dir}/integrated_data/v1/relation2id.csv'.
    Args:
        data_dir: root path for data directory
        want_relation: string name for desired relation
    """
    relation2id = (
        pd.read_csv(os.path.join(data_dir, "integrated_data", "v1", "relation2id.csv"))
        .set_index("relation")
        .to_dict()["index"]
    )
    # Note that this mapping is slightly different than for other datasets,
    # since the relation in relation2id is described as e.g. "protein_protein"
    # whereas self.relation_type is something like "homology".
    if want_relation not in relation2id:
        raise ValueError(f"Unexpected relation: {want_relation}")
    return relation2id[want_relation]

def get_negative_sampling_mask_and_sims(
    data_dir: str,
    data_type: str,
    sims_type: str,
    expected_num: int,
    mask_filename: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve AA seq masks and similarity matrices for negative sampling.

    Loads pre-computed masks and similarity matrices from
    '{data_dir}/generated_data/negative_sampling[masks|probs]'.

    Args:
        data_dir: root path for data directory
        aaseq_type: type of amino acid sequence, e.g. "protein"
        aaseq_sims_type: method for calculating AA seq similarity, must match to precomputed file
        expected_num: expected number of rows in mask and similarity matrix
    """

    mask_path = os.path.join(data_dir,
                            "generated_data",
                            "negative_sampling_masks",
                            mask_filename)
    if os.path.exists(mask_path):
        # In the mask, 1 indicates possible, 0 indicates impossible.
        mask = np.load(mask_path, mmap_mode='r')
        assert len(mask) == expected_num
    else:
        print(f"Negative sampling mask not found, using None. For {data_type}_{sims_type}: {mask_path}")
        mask = None

    if sims_type is not None:
        sims = np.load(os.path.join(data_dir,
                                    "generated_data",
                                    "negative_sampling_probs",
                                    f"{data_type}_sims_{sims_type}.npy"),
                        mmap_mode='r')
        assert len(sims) == expected_num
    else:
        sims = None
    return mask, sims

def validate_specified_splits(data: pd.DataFrame,
                              dataset_name: str,
                              want_splits: List[str],
    ) -> None:
    any_missing = False
    for split in want_splits:
        n = (data.split == split).sum()
        if n == 0:
            any_missing = True
            print(f"dataset {dataset_name} has zero samples with split={split}, typo?")
    if any_missing:
        raise ValueError("Some specified splits have zero samples, see above messages.")

class ProteinDataset(Dataset):
    """
    Protein dataset for MLM.
    """
    def __init__(
        self,
        data_dir = DATA_DIR,
        training=True,
        all_data = False,
    ):
        self.data_dir = data_dir
        self.split = 'train' if training else 'val'
        self.all_data = all_data
        self._load_data()

    # TODO: add information about domains for contiguous masking.
    def _load_data(self):
        if self.all_data:
            protein_ids = pd.read_pickle(self.data_dir+f'integrated_data/v1/protein/protein_info_filtered.pkl')
        else:
            protein_ids = pd.read_pickle(self.data_dir+f'integrated_data/v1/protein/protein_info_filtered_CL_{self.split}.pkl')
        self.protein_ids = protein_ids['index'].values.tolist()

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, index):
        return self.protein_ids[index]

class ProteinEvalDataset(Dataset):
    """
    Protein dataset for retrieval evaluation, loads directly from an evaluation dataframe.
    """
    def __init__(
        self,
        protein_df
    ):
        self.proteins = torch.from_numpy(protein_df.values.flatten()).long()

    def __len__(self):
        return self.proteins.shape[0]

    def __getitem__(self, index):
        return self.proteins[index]

class TextCLDataset(Dataset):
    """
    Dataset for text CL - only supports GO CL for now
    """
    def __init__(
        self,
        data_dir,
        go_split_method,
        training=True,
    ):
        self.data_dir = data_dir
        self.split = 'train' if training else 'val'
        self.go_split_method = go_split_method
        self._load_data()

    def _load_data(self):
        # TODO: handle this in a better way (save sets of GOs in each split)
        #protein_go_relations = pd.read_csv(self.data_dir+f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_CL_{self.split}_indexed.csv')[['seq_id', 'relation', 'text_id']]
        protein_go_relations = get_X_go_relations('protein', self.split, self.go_split_method)
        self.go_ids = protein_go_relations['text_id'].unique().tolist()

    def __len__(self):
        return len(self.go_ids)

    def __getitem__(self, index):
        return self.go_ids[index]

class ProteinGODataset_OLD(Dataset):
    """
    Protein-GO relations dataset for CL. Currently implemented for edge sampling only. TODO: Make it compatible with neighbor sampling based on proteins (instead of sampling relations randomly).

    It is not possible to do in-batch negative sampling for GOs, since we require that negative GOs must stay within the same namespace of the positive GO. Thus, the GO negative sampling is done for each sample separately, while the protein negative sampling, if required, is done for each batch as a whole.
    """
    def __init__(
        self,
        data_dir: str,
        go_split_method: str,
        negative_sampling_strategy: str,  # choose from ['go_only', 'protein_go_both', 'protein_only']
        protein_sims_type: str,  # choose from ['jaccard', 'k-hop', None]
        go_sims_type: str,  # choose from ['jaccard', 'k-hop', None]
        num_neg_samples_protein_go_per_protein: int,
        num_neg_samples_protein_go_per_go: int,
        use_only_goa_proteins: bool,
        use_only_goa_gos: bool,
        training: bool = True,
        eval_split: str = None, # Don't set if you're using "training" or "val" splits (i.e. pretext splits)
    ):
        assert negative_sampling_strategy in ['go_only', 'protein_go_both', 'protein_only', 'in_batch']
        self.data_dir = data_dir
        self.go_split_method = go_split_method
        self.negative_sampling_strategy = negative_sampling_strategy
        self.protein_sims_type = protein_sims_type
        self.go_sims_type = go_sims_type
        self.num_neg_samples_protein_go_per_protein = num_neg_samples_protein_go_per_protein
        self.num_neg_samples_protein_go_per_go = num_neg_samples_protein_go_per_go
        self.use_only_goa_proteins = use_only_goa_proteins
        self.use_only_goa_gos = use_only_goa_gos
        if (not training) and (eval_split is not None):
            self.split = eval_split
        else:
            self.split = 'train' if training else 'val'

        self._load_data()

    def _load_data(self):
        go2id = pd.read_csv(self.data_dir+"integrated_data/go2id.csv", header=None, index_col=None).rename(columns={0:'index', 1:'go'})
        # relation2id = pd.read_csv(data_dir+"integrated_data/relation2id.csv", header=None, index_col=None).rename(columns={0:'index', 1:'relation'})
        protein2id = pd.read_csv(self.data_dir+"integrated_data/protein2id.csv", header=None, index_col=None).rename(columns={0:'index', 1:'protein'})

        # load cl train pretrain relations
        protein_go_relations = pd.read_csv(self.data_dir+f'integrated_data/protein_go/{self.go_split_method}/protein_go_relations_CL_{self.split}_indexed.csv')[['seq_id', 'relation', 'text_id']]
        protein_go_relations_cl = pd.concat([pd.read_csv(self.data_dir+f'integrated_data/protein_go/{self.go_split_method}/protein_go_relations_CL_train_indexed.csv'), pd.read_csv(self.data_dir+f'integrated_data/protein_go/{self.go_split_method}/protein_go_relations_CL_val_indexed.csv')])[['seq_id', 'relation', 'text_id']]

        ########## per-sample GO negative sampling preparation ##########
        # get all GOs available
        if self.use_only_goa_gos:
            self.all_gos = np.unique(protein_go_relations_cl.text_id.values)
        else:
            self.all_gos = go2id.index.values
        self.num_gos = len(self.all_gos)

        ## if negative sampling GOs, load GO negative sampling masks and similarity matrix
        if self.negative_sampling_strategy in ['protein_go_both', 'go_only']:
            self.go_masks = np.load(self.data_dir+'generated_data/negative_sampling_masks/go_generic_masks.npy', mmap_mode='r') # In the mask, 1 indicates possible, 0 indicates impossible.
            assert self.go_masks.shape[0] == go2id.shape[0]

            if self.go_sims_type is not None:
                self.go_sims = np.load(self.data_dir+f'generated_data/negative_sampling_probs/go_sims_{self.go_sims_type}.npy', mmap_mode='r')  # only keep the sims for the GOs that are available (on axis 1 because axis 0 is used for indexing)
                assert self.go_sims.shape[0] == go2id.shape[0]
            else:
                self.go_sims = [None] * len(go2id)

            ## load ground truth GOs for each protein
            self.true_gos = dict()
            for _, prot_idx, rel_idx, go_idx in protein_go_relations_cl.itertuples():
                self.true_gos.setdefault((prot_idx, rel_idx), []).append(go_idx)  # using List instead of Set for indexing mask

        ########## per-sample protein negative sampling preparation ##########
        # get all proteins available
        if self.use_only_goa_proteins:
            self.all_proteins = np.unique(protein_go_relations_cl.seq_id.values)
        else:
            self.all_proteins = protein2id[0].values
        self.num_proteins = len(self.all_proteins)

        # if negative sampling proteins, load protein negative sampling masks and similarity matrix
        if self.negative_sampling_strategy in ['protein_go_both', 'protein_only']:
            # dummy 2d mask from disk
            self.protein_masks = np.load(self.data_dir+'generated_data/negative_sampling_masks/protein_dummy_masks.npy', mmap_mode='r')
            assert self.protein_masks.shape[0] == len(protein2id)

            if self.protein_sims_type is not None:
                self.protein_sims = np.load(self.data_dir+f'generated_data/negative_sampling_probs/protein_sims_{self.protein_sims_type}.npy', mmap_mode='r')  # get the submatrix of the proteins
                assert self.protein_sims.shape[0] == len(protein2id)
            else:
                self.protein_sims = np.array([None] * len(protein2id))

            ## load ground proteins for each GO
            self.true_proteins = dict()
            for _, prot_idx, rel_idx, go_idx in protein_go_relations_cl.itertuples():
                self.true_proteins.setdefault((go_idx, rel_idx), []).append(prot_idx)

        # load protein-GO relations
        self.protein_go_relations = list(protein_go_relations.itertuples())

    def __getitem__(self, index):
        # NOTE: Only supports random edge sampling for now.
        _, prot_idx, rel_idx, go_idx = self.protein_go_relations[index]
        negative_protein_indices, negative_go_indices = [], []

        if self.negative_sampling_strategy in {'go_only', 'protein_go_both'}:
            go_mask = np.array(self.go_masks[go_idx, :])
            go_mask[self.true_gos[(prot_idx, rel_idx)]] = 0
            go_mask = go_mask[self.all_gos]
            go_sim = np.array(self.go_sims[go_idx, self.all_gos])

            go_prob = process_go_sims(go_sim, self.negative_sampling_strategy)
            negative_go_indices = negative_sampling_random_tail((prot_idx, rel_idx), self.num_neg_samples_protein_go_per_go, self.num_gos, go_mask, go_prob)

            # in case the GOs in GOA do not cover all GOs in the KG
            negative_go_indices = self.all_gos[negative_go_indices].tolist()

        if self.negative_sampling_strategy in {'protein_go_both', 'protein_only'}:
            protein_mask = np.array(self.protein_masks[prot_idx, :])
            protein_mask[self.true_proteins[(go_idx, rel_idx)]] = 0
            protein_mask = protein_mask[self.all_proteins]
            protein_sim = np.array(self.protein_sims[prot_idx, self.all_proteins])

            protein_prob = process_protein_sims(protein_sim, self.negative_sampling_strategy)
            negative_protein_indices = negative_sampling_random_tail((go_idx, rel_idx), self.num_neg_samples_protein_go_per_protein, self.num_proteins, protein_mask, protein_prob)

            # in case the proteins in GOA do not cover all proteins in the KG
            negative_protein_indices = self.all_proteins[negative_protein_indices].tolist()

        if self.negative_sampling_strategy == 'in_batch':
            negative_protein_indices, negative_go_indices = None, None

        return (prot_idx, rel_idx, go_idx), negative_protein_indices, negative_go_indices

    def __len__(self):
        return len(self.protein_go_relations)

class ProteinGODataset(Dataset):
    """
    Protein-GO relations dataset for CL. Currently implemented for edge sampling only. TODO: Make it compatible with neighbor sampling based on proteins (instead of sampling relations randomly).

    It is not possible to do in-batch negative sampling for GOs, since we require that negative GOs must stay within the same namespace of the positive GO. Thus, the GO negative sampling is done for each sample separately, while the protein negative sampling, if required, is done for each batch as a whole.
    """
    def __init__(
        self,
        data_dir: str,
        go_split_method: str,
        negative_sampling_strategy: str,  # choose from ['go_only', 'protein_go_both', 'protein_only']
        protein_sims_type: str,  # choose from ['jaccard', 'k-hop', None]
        go_sims_type: str,  # choose from ['jaccard', 'k-hop', None]
        num_neg_samples_protein_go_per_protein: int,
        num_neg_samples_protein_go_per_go: int,
        use_only_goa_proteins: bool,
        use_only_goa_gos: bool,
        split: bool = 'training',
        val_split_type = None,
        eval_split: str = None, # Don't set if you're using "training" or "val" splits (i.e. pretext splits)
        testing_kwargs: dict = None,
    ):
        assert negative_sampling_strategy in ['go_only', 'protein_go_both', 'protein_only', 'in_batch']
        self.data_dir = data_dir
        self.go_split_method = go_split_method
        self.negative_sampling_strategy = negative_sampling_strategy
        self.protein_sims_type = protein_sims_type
        self.go_sims_type = go_sims_type
        self.num_neg_samples_protein_go_per_protein = num_neg_samples_protein_go_per_protein
        self.num_neg_samples_protein_go_per_go = num_neg_samples_protein_go_per_go
        self.use_only_goa_proteins = use_only_goa_proteins
        self.use_only_goa_gos = use_only_goa_gos
        self.val_split_type = val_split_type
        self.testing_kwargs = testing_kwargs
        self.split = split

        assert self.split in {'train', 'val', 'test'}, "Split must be in {'train', 'val', 'test'}"
        print('Split is {}'.format(self.split))

        # # Decide on split
        # if testing_kwargs is not None: # Presence of this taskes precedent
        #     self.split = 'test'
        # elif (not training) and (eval_split is not None):
        #     self.split = eval_split
        # else:
        #     self.split = 'train' if training else 'val'

        if (self.split == 'test') or (self.split == 'val'):
            self._load_testing_data()
        else:
            self._load_data()

    def _load_data(self):
        go_info = pd.read_pickle(self.data_dir+"integrated_data/v1/go/go_info_filtered.pkl")
        # relation2id = pd.read_csv(data_dir+"integrated_data/v1/relation2id.csv", header=None, index_col=None).rename(columns={0:'index', 1:'relation'})
        protein_info = pd.read_pickle(self.data_dir+"integrated_data/v1/protein/protein_info_filtered.pkl")

        # protein_id, relation_type, go_id

        # load cl train pretrain relations
        if self.split == 'val':
            protein_go_relations = pd.read_csv(self.data_dir+f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_CL_val_{self.val_split_type}_indexed.csv')[['seq_id', 'relation', 'text_id']]
        else:
            protein_go_relations = pd.read_csv(self.data_dir+f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_CL_{self.split}_indexed.csv')[['seq_id', 'relation', 'text_id']]
        protein_go_relations_cl = pd.concat([
            pd.read_csv(self.data_dir+f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_CL_train_indexed.csv'),
            pd.read_csv(self.data_dir+f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_CL_val_pt_ft_indexed.csv'),
            pd.read_csv(self.data_dir+f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_CL_val_five_shot_indexed.csv'),
            pd.read_csv(self.data_dir+f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_CL_val_zero_shot_indexed.csv'),
        ])[['seq_id', 'relation', 'text_id']]

        # if self.split == 'val':
        #     protein_go_relations_cl = pd.read_csv(self.data_dir+f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_CL_val_{self.val_split_type}_indexed.csv')[['seq_id', 'relation', 'text_id']]
        # else:

        # elif self.val_split_type is not None:

        # else:
        #     protein_go_relations_cl = pd.read_csv(self.data_dir+f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_CL_eval_{self.split}_indexed.csv')[['seq_id', 'relation', 'text_id']]

        #protein_go_relations_cl = pd.concat([pd.read_csv(self.data_dir+f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_CL_train_indexed.csv'), pd.read_csv(self.data_dir+f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_CL_val_pt_ft_indexed.csv')])[['seq_id', 'relation', 'text_id']]

        # NOTE: WARNING: WE COMBINE VALIDATION WITH TRAINING ABOVE, DON'T DO THIS WITH ON-THE-FLY EVALUATION

        # TODO:
        # protein_go_relations = get_X_go_relations('protein', self.split, self.go_split_method)
        # protein_go_relations_cl = pd.concat([get_X_go_relations('protein', 'train', self.go_split_method), get_X_go_relations('protein', 'val', self.go_split_method)])

        ########## per-sample GO negative sampling preparation ##########
        # get all GOs available
        if self.use_only_goa_gos:
            # TODO: need a better way to access all_go's than loading all the dataframes above
            self.all_gos = np.unique(protein_go_relations_cl['text_id'].values)
        else:
            self.all_gos = go_info['index'].values
        self.num_gos = len(self.all_gos)

        ## if negative sampling GOs, load GO negative sampling masks and similarity matrix
        if self.negative_sampling_strategy in ['protein_go_both', 'go_only']:
            self.go_masks = np.load(self.data_dir+'generated_data/negative_sampling_masks/go_generic_masks.npy', mmap_mode='r') # In the mask, 1 indicates possible, 0 indicates impossible.
            assert self.go_masks.shape[0] == go_info.shape[0]
            self.go_sims_type = None
            if self.go_sims_type is not None:
                self.go_sims = np.load(self.data_dir+f'generated_data/negative_sampling_probs/go_sims_{self.go_sims_type}.npy', mmap_mode='r')  # only keep the sims for the GOs that are available (on axis 1 because axis 0 is used for indexing)
                assert self.go_sims.shape[0] == go_info.shape[0]
            else:
                self.go_sims = [None] * len(go_info)

            ## load ground truth GOs for each protein
            self.true_gos = dict()
            for _, prot_idx, rel_idx, go_idx in protein_go_relations_cl.itertuples():
                self.true_gos.setdefault((prot_idx, rel_idx), []).append(go_idx)  # using List instead of Set for indexing mask

        ########## per-sample protein negative sampling preparation ##########
        # get all proteins available
        if self.use_only_goa_proteins:
            self.all_proteins = np.unique(protein_go_relations_cl['seq_id'].values)
        else:
            self.all_proteins = protein_info['index'].values
        self.num_proteins = len(self.all_proteins)

        # if negative sampling proteins, load protein negative sampling masks and similarity matrix
        if self.negative_sampling_strategy in ['protein_go_both', 'protein_only']:
            # dummy 2d mask from disk
            self.protein_masks = np.load(self.data_dir+'generated_data/negative_sampling_masks/protein_dummy_masks.npy', mmap_mode='r')
            assert self.protein_masks.shape[0] == len(protein_info)
            self.protein_sims_type = None
            if self.protein_sims_type is not None:
                self.protein_sims = np.load(self.data_dir+f'generated_data/negative_sampling_probs/protein_sims_{self.protein_sims_type}.npy', mmap_mode='r')  # get the submatrix of the proteins
                assert self.protein_sims.shape[0] == len(protein_info)
            else:
                self.protein_sims = None

            ## load ground proteins for each GO
            self.true_proteins = dict()
            for _, prot_idx, rel_idx, go_idx in protein_go_relations_cl.itertuples():
                self.true_proteins.setdefault((go_idx, rel_idx), []).append(prot_idx)

        # load protein-GO relations
        self.protein_go_relations = list(protein_go_relations.itertuples())

    def _load_testing_data(self):
        # TODO: Can merge this in to _load_data

        # Break down testing kwargs:
        shot_level = self.testing_kwargs["shot_level"]
        if "use_preset_negatives" in self.testing_kwargs.keys():
            self.testing_use_preset_negatives = self.testing_kwargs["use_preset_negatives"]
        else:
            self.testing_use_preset_negatives = True

        if "num_negatives" in self.testing_kwargs.keys():
            self.testing_num_negatives = self.testing_kwargs["num_negatives"]
        else:
            self.testing_num_negatives = None

        sub_split_name = 'eval' if self.split == 'test' else 'CL_val'

        if self.testing_use_preset_negatives:
            # Example file: protein_go_relations_eval_pt_ft_indexed_with_10_negatives.csv
            protein_go_relations_df = pd.read_csv(os.path.join(self.data_dir, f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_{sub_split_name}_{shot_level}_indexed_with_10_negatives.csv'))
            if self.testing_num_negatives is not None:
                # Static indexing for now - could sample later - for reproducibility and consistency
                neg_strs_expanded = ['neg_seq_id_{}'.format(i) for i in range(self.testing_num_negatives)]
            else:
                neg_strs_expanded = ['neg_seq_id_{}'.format(i) for i in range(10)]
            self.protein_go_relations = list(protein_go_relations_df[['seq_id', 'relation', 'text_id']].itertuples())
            self.protein_go_negative_relations = list(protein_go_relations_df[neg_strs_expanded].itertuples())
        else:
            #raise NotImplementedError
            #protein_go_relations = pd.read_csv(os.path.join(self.data_dir, f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_eval_{shot_level}_indexed.csv'))[['seq_id', 'relation', 'text_id']]
            protein_go_relations = pd.read_csv(os.path.join(self.data_dir, f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_{sub_split_name}_{shot_level}_indexed_with_10_negatives.csv'))[['seq_id', 'relation', 'text_id']]

            # Break into only unique GO's
            self.unique_relations_by_go = first_unique_value_in_pandas_df(protein_go_relations, col = 'text_id')
            self.unique_gos = self.unique_relations_by_go["text_id"]
            self.unique_proteins = first_unique_value_in_pandas_df(protein_go_relations, col = 'seq_id')["seq_id"]

            # load protein-GO relations
            self.protein_go_relations = list(self.unique_relations_by_go.itertuples())

    def __getitem__(self, index):
        # NOTE: Only supports random edge sampling for now.
        # First blank index ignores the index in itertuples

        if (self.split == 'test') or (self.split == 'val'):
            # Get testing positives and negatives
            negative_protein_indices = []
            negative_go_indices = []

            if self.testing_use_preset_negatives:
                _, prot_idx, rel_idx, go_idx = self.protein_go_relations[index]
                negative_protein_indices = list(self.protein_go_negative_relations[index][1:])
                negative_go_indices = [go_idx] * len(negative_protein_indices)
            else:
                _, prot_idx, rel_idx, go_idx = self.protein_go_relations[index]
                negative_protein_indices, negative_go_indices = None, None
        else:
            _, prot_idx, rel_idx, go_idx = self.protein_go_relations[index]
            negative_protein_indices, negative_go_indices = [], []
            if self.negative_sampling_strategy in {'go_only', 'protein_go_both'}:
                go_mask = np.array(self.go_masks[go_idx, :])
                go_mask[self.true_gos[(prot_idx, rel_idx)]] = 0
                go_mask = go_mask[self.all_gos]
                # print(go_idx, self.all_gos)
                go_sim = np.array(self.go_sims[go_idx, self.all_gos])

                go_prob = process_go_sims(go_sim, self.negative_sampling_strategy)
                negative_go_indices = negative_sampling_random_tail((prot_idx, rel_idx), self.num_neg_samples_protein_go_per_go, self.num_gos, go_mask, go_prob)

                # in case the GOs in GOA do not cover all GOs in the KG
                negative_go_indices = self.all_gos[negative_go_indices].tolist()

            if self.negative_sampling_strategy in {'protein_go_both', 'protein_only'}:
                protein_mask = np.array(self.protein_masks[prot_idx, :])
                protein_mask[self.true_proteins[(go_idx, rel_idx)]] = 0
                protein_mask = protein_mask[self.all_proteins]
                if self.protein_sims is None:
                    protein_sim = None
                else:
                    protein_sim = np.array(self.protein_sims[prot_idx, self.all_proteins])

                protein_prob = process_protein_sims(protein_sim, self.negative_sampling_strategy)
                negative_protein_indices = negative_sampling_random_tail((go_idx, rel_idx), self.num_neg_samples_protein_go_per_protein, self.num_proteins, protein_mask, protein_prob)

                # in case the proteins in GOA do not cover all proteins in the KG
                negative_protein_indices = self.all_proteins[negative_protein_indices].tolist()
            elif self.negative_sampling_strategy == 'in_batch':
                negative_protein_indices, negative_go_indices = None, None

        if self.negative_sampling_strategy == 'in_batch':
            negative_protein_indices, negative_go_indices = None, None

        return (prot_idx, rel_idx, go_idx), negative_protein_indices, negative_go_indices

    def __len__(self):
        return len(self.protein_go_relations)

class AASeqTextDataset(Dataset):
    """
    Generalization of the ProteinGODataset to accommodate inputs across databases. Negative sampling is performed sample-wise for all types of inputs.
    """
    def __init__(
        self,
        data_dir: str,
        aaseq_type: str = "protein",
        text_type: str = "go",
        relation_type: str = "all",
        text_split_method: str = 'sample_aware_ontology_go_centric',
        negative_sampling_strategy: str = 'aaseq_only',  # choose from ['text_only', 'aaseq_text_both', 'aaseq_only']
        aaseq_sims_type: str = None,  # choose from ['jaccard', 'k-hop', None]
        text_sims_type: str = None,  # choose from ['jaccard', 'k-hop', None]
        num_neg_samples_aaseq_text_per_aaseq: int = None,
        num_neg_samples_aaseq_text_per_text: int = 64,
        use_only_aaseq_text_aaseqs: bool = True,
        use_only_aaseq_text_texts: bool = True,
        #training: bool = True,
        split = 'train',
        val_split_type = None,
        eval_split: str = None, # Don't set if you're using "training" or "val" splits (i.e. pretext splits)
        testing_kwargs: dict = None,
        seed: int = 42,
    ):
        assert negative_sampling_strategy in ['text_only', 'aaseq_text_both', 'aaseq_only', 'in_batch']
        self.data_dir = data_dir
        self.aaseq_type = aaseq_type
        self.text_type = text_type
        self.relation_type = relation_type
        self.text_split_method = text_split_method
        self.negative_sampling_strategy = negative_sampling_strategy
        self.aaseq_sims_type = aaseq_sims_type
        self.text_sims_type = text_sims_type
        self.num_neg_samples_aaseq_text_per_aaseq = num_neg_samples_aaseq_text_per_aaseq
        self.num_neg_samples_aaseq_text_per_text = num_neg_samples_aaseq_text_per_text
        self.use_only_aaseq_text_aaseqs = use_only_aaseq_text_aaseqs
        self.use_only_aaseq_text_texts = use_only_aaseq_text_texts
        self.val_split_type = val_split_type
        #self.training = training
        self.testing_kwargs = testing_kwargs
        self.split = split
        self.rng = np.random.default_rng(seed)
        #self.rng = None

        assert self.split in {'train', 'val', 'test'}, f"Split must be in {'train', 'val', 'test'}, got: {self.split}"

        if (self.split == 'test') or (self.split == 'val'):
            self._load_testing_data()
        else:
            self._load_data()

    def _load_data(self):
        text_info = pd.read_pickle(self.data_dir+f"integrated_data/v1/{self.text_type}/{self.text_type}_info_filtered.pkl")
        aaseq_info = pd.read_pickle(self.data_dir+f"integrated_data/v1/{self.aaseq_type}/{self.aaseq_type}_info_filtered.pkl")

        if self.relation_type != "all":
            if self.text_type != 'go':
                valid_rel = get_and_check_relation_id(
                    self.data_dir, self.relation_type
                )

        # load cl train pretrain relations
        if self.split == 'val':
            aaseq_text_relations = pd.read_csv(self.data_dir+f'integrated_data/v1/{self.aaseq_type}_{self.text_type}/{self.text_split_method}/{self.aaseq_type}_{self.text_type}_relations_CL_val_{self.val_split_type}_indexed.csv')
        else:
            aaseq_text_relations = pd.read_csv(self.data_dir+f'integrated_data/v1/{self.aaseq_type}_{self.text_type}/{self.text_split_method}/{self.aaseq_type}_{self.text_type}_relations_CL_{self.split}_indexed.csv')
        shot_level = 'five' if self.text_type not in {'drugbank', 'omim', 'disgenet', 'reactome'} else 'two'
        aaseq_text_relations_cl = pd.concat([
            pd.read_csv(self.data_dir+f'integrated_data/v1/{self.aaseq_type}_{self.text_type}/{self.text_split_method}/{self.aaseq_type}_{self.text_type}_relations_CL_train_indexed.csv'),
            pd.read_csv(self.data_dir+f'integrated_data/v1/{self.aaseq_type}_{self.text_type}/{self.text_split_method}/{self.aaseq_type}_{self.text_type}_relations_CL_val_pt_ft_indexed.csv'),
            pd.read_csv(self.data_dir+f'integrated_data/v1/{self.aaseq_type}_{self.text_type}/{self.text_split_method}/{self.aaseq_type}_{self.text_type}_relations_CL_val_{shot_level}_shot_indexed.csv'),
            pd.read_csv(self.data_dir+f'integrated_data/v1/{self.aaseq_type}_{self.text_type}/{self.text_split_method}/{self.aaseq_type}_{self.text_type}_relations_CL_val_zero_shot_indexed.csv'),
        ])

        if self.relation_type != 'all': # Filtering based on relation
            if self.text_type != 'go':
                aaseq_text_relations = aaseq_text_relations.loc[aaseq_text_relations['relation'] == valid_rel,:]
                aaseq_text_relations_cl = aaseq_text_relations_cl.loc[aaseq_text_relations_cl['relation'] == valid_rel,:]
            else:
                aaseq_text_relations = aaseq_text_relations.loc[aaseq_text_relations['text_type'].str.lower() == self.relation_type.lower(),:]
                aaesq_text_relations_cl = aaseq_text_relations_cl.loc[aaseq_text_relations_cl['text_type'].str.lower() == self.relation_type.lower(),:]

        # Filter afterwards:
        aaseq_text_relations = aaseq_text_relations[['seq_id', 'relation', 'text_id']]
        aaseq_text_relations_cl = aaseq_text_relations_cl[['seq_id', 'relation', 'text_id']]

        ########## per-sample GO negative sampling preparation ##########
        # get all GOs available
        if self.use_only_aaseq_text_texts:
            # TODO: need a better way to access all_go's than loading all the dataframes above
            self.all_texts = np.unique(aaseq_text_relations_cl['text_id'].values)
        else:
            self.all_texts = text_info['index'].values
        self.num_texts = len(self.all_texts)

        ## if negative sampling GOs, load GO negative sampling masks and similarity matrix
        if self.negative_sampling_strategy in ['aaseq_text_both', 'text_only']:
            self.text_masks, self.text_sims = get_negative_sampling_mask_and_sims(
                self.data_dir,
                self.text_type,
                self.text_sims_type,
                len(text_info),
                f"{self.text_type}_generic_masks.npy",
            )
            self.len_text_mask = len(text_info)

            ## load ground truth GOs for each protein
            self.true_texts = dict()
            for _, aaseq_idx, rel_idx, text_idx in aaseq_text_relations_cl.itertuples():
                self.true_texts.setdefault((aaseq_idx, rel_idx), []).append(text_idx)  # using List instead of Set for indexing mask

        ########## per-sample protein negative sampling preparation ##########
        # get all proteins available
        if self.use_only_aaseq_text_aaseqs:
            self.all_aaseqs = np.unique(aaseq_text_relations_cl['seq_id'].values)
        else:
            self.all_aaseqs = aaseq_info['index'].values
        self.num_aaseqs = len(self.all_aaseqs)

        # if negative sampling proteins, load protein negative sampling masks and similarity matrix
        if self.negative_sampling_strategy in ['aaseq_text_both', 'aaseq_only']:
            self.aaseq_masks, self.aaseq_sims = get_negative_sampling_mask_and_sims(
                self.data_dir,
                self.aaseq_type,
                self.aaseq_sims_type,
                len(aaseq_info),
                f"{self.aaseq_type}_dummy_masks.npy",
            )

            ## load ground proteins for each GO
            self.true_aaseqs = dict()
            for _, aaseq_idx, rel_idx, text_idx in aaseq_text_relations_cl.itertuples():
                self.true_aaseqs.setdefault((text_idx, rel_idx), []).append(aaseq_idx)

        # load protein-GO relations
        self.aaseq_text_relations = list(aaseq_text_relations.itertuples())

    def _load_testing_data(self):
        # TODO: Can merge this in to _load_data
        if self.relation_type != "all":
            if self.text_type != 'go':
                valid_rel = get_and_check_relation_id(
                    self.data_dir, self.relation_type
                )

        # Break down testing kwargs:
        shot_level = self.testing_kwargs["shot_level"]
        if "use_preset_negatives" in self.testing_kwargs.keys():
            self.testing_use_preset_negatives = self.testing_kwargs["use_preset_negatives"]
        else:
            self.testing_use_preset_negatives = True
        if "num_negatives" in self.testing_kwargs.keys():
            self.testing_num_negatives = self.testing_kwargs["num_negatives"]
        else:
            self.testing_num_negatives = None

        # TODO: Perform filtering of relations based on relation type considered

        if self.testing_use_preset_negatives:
            # Example file: protein_go_relations_eval_pt_ft_indexed_with_10_negatives.csv
            aaseq_text_relations_df = pd.read_csv(os.path.join(self.data_dir, f'integrated_data/v1/{self.aaseq_type}_{self.text_type}/{self.text_split_method}/{self.aaseq_type}_{self.text_type}_relations_eval_{shot_level}_indexed_with_10_negatives.csv'))

            if self.relation_type != 'all': # Filtering based on relation
                if self.text_type != 'go':
                    aaseq_text_relations_df = aaseq_text_relations_df.loc[aaseq_text_relations_df['relation'] == valid_rel,:]
                else:
                    aaseq_text_relations_df = aaseq_text_relations_df.loc[aaseq_text_relations_df['text_type'].str.lower() == self.relation_type.lower(),:]

            if self.testing_num_negatives is not None:
                # Static indexing for now - could sample later - for reproducibility and consistency
                neg_strs_expanded = ['neg_seq_id_{}'.format(i) for i in range(self.testing_num_negatives)]
            else:
                neg_strs_expanded = ['neg_seq_id_{}'.format(i) for i in range(10)]
            self.aaseq_text_relations = list(aaseq_text_relations_df[['seq_id', 'relation', 'text_id']].itertuples())
            self.aaseq_text_negative_relations = list(aaseq_text_relations_df[neg_strs_expanded].itertuples())
        else:
            #raise NotImplementedError
            #protein_go_relations = pd.read_csv(os.path.join(self.data_dir, f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_eval_{shot_level}_indexed.csv'))[['seq_id', 'relation', 'text_id']]
            aaseq_text_relations = pd.read_csv(os.path.join(self.data_dir, f'integrated_data/v1/{self.aaseq_type}_{self.text_type}/{self.text_split_method}/{self.aaseq_type}_{self.text_type}_relations_eval_{shot_level}_indexed_with_10_negatives.csv'))

            if self.relation_type != 'all': # Filtering based on relation
                if self.text_type != 'go':
                    aaseq_text_relations = aaseq_text_relations.loc[aaseq_text_relations['relation'] == valid_rel,:]
                else:
                    aaseq_text_relations = aaseq_text_relations.loc[aaseq_text_relations['text_type'].str.lower() == self.relation_type.lower(),:]

            aaseq_text_relations = aaseq_text_relations[['seq_id', 'relation', 'text_id']]

            self.unique_relations_by_text = first_unique_value_in_pandas_df(aaseq_text_relations, col = 'text_id')
            self.unique_texts = self.unique_relations_by_text["text_id"]
            self.unique_aaseq = first_unique_value_in_pandas_df(aaseq_text_relations, col = 'seq_id')["seq_id"]
            # load protein-GO relations
            self.full_aaseq_text_relations_eval = aaseq_text_relations
            self.aaseq_text_relations = list(self.unique_relations_by_text.itertuples())

        # TODO: write negative sampling capabilities based on negative sampling provided by testing kwargs

    def __getitem__(self, index):
        # NOTE: Only supports random edge sampling for now.
        # First blank index ignores the index in itertuples

        import ipdb; ipdb.set_trace()

        if (self.split == 'test') or (self.split == 'val'):
            # Get testing positives and negatives
            negative_aaseq_indices = []
            negative_text_indices = []

            if self.testing_use_preset_negatives:
                _, aaseq_idx, rel_idx, text_idx = self.aaseq_text_relations[index]
                negative_aaseq_indices = list(self.aaseq_text_negative_relations[index][1:])
                negative_text_indices = [text_idx] * len(negative_aaseq_indices)
            else:
                _, aaseq_idx, rel_idx, text_idx = self.aaseq_text_relations[index]
                negative_aaseq_indices, negative_text_indices = None, None
        else:
            _, aaseq_idx, rel_idx, text_idx = self.aaseq_text_relations[index]
            negative_aaseq_indices, negative_text_indices = [], []
            if self.negative_sampling_strategy in {'text_only', 'aaseq_text_both'}:
                if self.text_masks is None:
                    text_mask = np.ones((self.len_text_mask,))
                else:
                    text_mask = np.copy(self.text_masks[text_idx, :])
                text_mask[self.true_texts[(aaseq_idx, rel_idx)]] = 0
                text_mask = text_mask[self.all_texts]
                if self.text_sims is None:
                    text_sim = None
                else:
                    text_sim = np.array(self.text_sims[text_idx, self.all_texts])

                text_prob = process_go_sims(text_sim, self.negative_sampling_strategy)
                negative_text_indices = negative_sampling_random_tail((aaseq_idx, rel_idx),
                                                                      self.num_neg_samples_aaseq_text_per_text,
                                                                      self.num_texts,
                                                                      text_mask,
                                                                      text_prob,
                                                                      self.rng)

                # in case the texts in textA do not cover all texts in the KG
                negative_text_indices = self.all_texts[negative_text_indices].tolist()

            if self.negative_sampling_strategy in {'aaseq_text_both', 'aaseq_only'}:
                aaseq_mask = np.array(self.aaseq_masks[aaseq_idx, :])
                aaseq_mask[self.true_aaseqs[(text_idx, rel_idx)]] = 0
                aaseq_mask = aaseq_mask[self.all_aaseqs]
                if self.aaseq_sims is None:
                    aaseq_sim = None
                else:
                    aaseq_sim = np.array(self.aaseq_sims[aaseq_idx, self.all_aaseqs])

                aaseq_prob = process_aaseq_sims(aaseq_sim, self.negative_sampling_strategy)
                negative_aaseq_indices = negative_sampling_random_tail((text_idx, rel_idx),
                                                                       self.num_neg_samples_aaseq_text_per_aaseq,
                                                                       self.num_aaseqs,
                                                                       aaseq_mask,
                                                                       aaseq_prob,
                                                                       self.rng)

                # in case the aaseqs in textA do not cover all aaseqs in the KG
                negative_aaseq_indices = self.all_aaseqs[negative_aaseq_indices].tolist()
            elif self.negative_sampling_strategy == 'in_batch':
                negative_aaseq_indices, negative_text_indices = None, None
            else:
                raise ValueError('Negative sampling strategy not recognized')

        return (aaseq_idx, rel_idx, text_idx), negative_aaseq_indices, negative_text_indices

    def __len__(self):
        return len(self.aaseq_text_relations)

class AbstractNegativeSampler(ABC):
    @abstractmethod
    def get_negative_samples(self,
    ) -> Union[List[int], None]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

class NullNegativeSampler(AbstractNegativeSampler):
    def __init__(self):
        pass

    def get_negative_samples(self, _: int) -> List[int]:
        return None

    def __len__(self) -> int:
        return 0

class RepeatNegativeSampler(AbstractNegativeSampler):
    def __init__(self,
                 values: List[int],
                 num_repeats: int,
    ):
        self.values = values
        self.num = num_repeats

    def get_negative_samples(self, idx: int) -> List[int]:
        return [self.values[idx]] * self.num

    def __len__(self) -> int:
        return 0

class PresetNegativeSampler(AbstractNegativeSampler):
    def __init__(self,
                 preset_negs: List[Tuple],
    ):
        self.preset_negatives = preset_negs

    def get_negative_samples(self, idx: int) -> List[int]:
        return list(self.preset_negatives[idx])

    def __len__(self) -> int:
        return len(self.preset_negatives)

class SimBasedNegativeSampler(AbstractNegativeSampler):
    def __init__(
        self,
        data_dir: str,
        data_type: str,
        sims_type: Union[str, None],
        num_negs: int,
        num_expected: int,
        expected_mask_name: str,
        all_relations: List[Tuple],
        used_data: List[int],
        used_relations: List[Tuple],
        sims_to_probs_fn: Callable,
        seed: int = 42,
    ):
        self.used_data = used_data
        self.used_relations = used_relations
        self.sims_to_probs = sims_to_probs_fn
        self.num_negs = num_negs
        self.num_expected = num_expected
        self.rng = np.random.default_rng(seed)

        self.mask, self.sims = get_negative_sampling_mask_and_sims(data_dir,
                                                                   data_type,
                                                                   sims_type,
                                                                   self.num_expected,
                                                                   expected_mask_name)

        ## load ground truth relations for each positive
        self.true_relations = defaultdict(list)
        for pos_idx, rel_idx, neg_idx in all_relations:
            self.true_relations[(pos_idx, rel_idx)].append(neg_idx)

    def get_negative_samples(
        self,
        idx: int,
    ) -> List[int]:
        """Get negative samples for a given relation.

        Args:
            pos_idx - Index of the datapoint we want to sample negatives
        """
        lhs_idx, rel_idx, rhs_idx = self.used_relations[idx]
        if self.mask is None:
            mask = np.ones((self.num_expected,))
        else:
            mask = np.copy(self.mask[rhs_idx, :])

        mask[self.true_relations[(lhs_idx, rel_idx)]] = 0
        mask = mask[self.used_data]
        if self.sims is None:
            sims = None
        else:
            sims = np.array(self.sims[rhs_idx, self.used_data])

        probs = self.sims_to_probs(sims)
        negative_indices = negative_sampling_random_tail(None,
                                                         self.num_negs,
                                                         len(self.used_data),
                                                         mask,
                                                         probs,
                                                         self.rng)

        # in case the texts in textA do not cover all texts in the KG
        return self.used_data[negative_indices].tolist()

    def __len__(self) -> int:
        return self.num_expected


def load_unified_aaseq_text_relations(
    data_dir: str,
    aaseq_type: str,
    text_type: str,
    text_split_method: str,
    relation_type: str,
) -> pd.DataFrame:
    """Helper method for loading and checking aaseq <-> text relations."""
    all_relations = pd.read_csv(os.path.join(data_dir,
                                             "integrated_data",
                                             "v1",
                                             f"{aaseq_type}_{text_type}",
                                             text_split_method,
                                             f"{aaseq_type}_{text_type}_relations_indexed.unified.csv"))
    if relation_type != "all":
        # Filter dataframe by relation:
        if text_type != 'go':
            valid_rel = get_and_check_relation_id(
                data_dir,
                relation_type,
            )
            all_relations = all_relations.loc[lambda x: x.relation == valid_rel]
        else:
            all_relations = all_relations.loc[lambda x: x.text_type.str.lower() == relation_type.lower()]
    return all_relations

class AASeqTextUnifiedDataset(Dataset):
    """
    Generalization of the ProteinGODataset to accommodate inputs across databases. Negative sampling is performed sample-wise for all types of inputs.

    This is a modification of the above AASeqTextDataset to allow for loading from a single dataframe
    containing all relations for a given dataset with run-time subsetting to the specified splits.
    """
    def __init__(
        self,
        data_dir: str,
        aaseq_type: str,
        text_type: str,
        # Relations to include, can be 'all' to include all relations (although note that this
        # may not be compatible with downstream collators if they have different instructions
        # for different relations).
        relation_type: str,
        # List of splits to include in this dataset, e.g. [cl_val_pt_ft, cl_val_five_shot]
        splits_to_use: List[str],

        text_split_method: str = "sample_aware_ontology_go_centric",
        negative_sampling_strategy: str = "aaseq_only",  # choose from ['text_only', 'aaseq_text_both', 'aaseq_only']

        aaseq_sims_type: Union[str, None] = None,  # choose from ['jaccard', 'k-hop', None]
        text_sims_type: Union[str, None] = None,  # choose from ['jaccard', 'k-hop', None]
        num_neg_samples_aaseq_text_per_aaseq: int = Union[int, None],
        num_neg_samples_aaseq_text_per_text: int = 64,
        use_only_aaseq_text_aaseqs: bool = True,
        use_only_aaseq_text_texts: bool = True,

        num_preset_negs: Union[int, None] = None,
        # Random seed for similarity-based negative sampling, not used
        # if negative_sampling_strategy is in ["in_batch", "preset", "none"]
        seed: int = 42,
        use_perplexity_filtered_set = False,

        # Subset to a single relation per text or seq. This is primarily to support evaluation, where we may
        # only want to process a given text once (e.g. for retrieval or captioning, where the relation
        # doesn't actually affect the result).
        deduplicate_dataset: Optional[str] = None,

        # Subset to a pre-specified set of amino acid sequences, e.g. to only evaluate on a
        # small subset of the dataset.
        aaseq_subset: Optional[List[int]] = None,
    ):
        assert negative_sampling_strategy in [
            "text_only",
            "aaseq_text_both",
            "aaseq_only",
            "in_batch",
            "preset",
            "none",
        ]

        if ':' in text_type:
            tt_split = text_type.split(":")
            self.text_type = tt_split[0]
        else:
            self.text_type = text_type

        self.data_dir = data_dir
        self.aaseq_type = aaseq_type
        #self.text_type = text_type
        self.relation_type = relation_type
        self.splits_to_use = splits_to_use

        self.text_split_method = text_split_method
        self.negative_sampling_strategy = negative_sampling_strategy
        self.aaseq_sims_type = aaseq_sims_type
        self.text_sims_type = text_sims_type
        self.num_neg_samples_aaseq_text_per_aaseq = num_neg_samples_aaseq_text_per_aaseq
        self.num_neg_samples_aaseq_text_per_text = num_neg_samples_aaseq_text_per_text
        self.use_only_aaseq_text_aaseqs = use_only_aaseq_text_aaseqs
        self.use_only_aaseq_text_texts = use_only_aaseq_text_texts

        self.seed = seed
        self.use_perplexity_filtered_set = use_perplexity_filtered_set

        assert deduplicate_dataset is None or deduplicate_dataset in ["aaseq", "text"]
        self.deduplicate_dataset = deduplicate_dataset

        self.aaseq_subset = aaseq_subset

        # Using preset negatives is sufficiently different that this seems
        # like the most straightforward way to handle it
        if negative_sampling_strategy == "preset":
            self._load_data_preset_negatives(num_preset_negs)
        else:
            self._load_data()

    def _load_data(self):
        text_info = pd.read_pickle(os.path.join(self.data_dir,
                                                "integrated_data",
                                                "v1",
                                                self.text_type,
                                                f"{self.text_type}_info_filtered.pkl"))
        aaseq_info = pd.read_pickle(os.path.join(self.data_dir,
                                                 "integrated_data",
                                                 "v1",
                                                 self.aaseq_type,
                                                 f"{self.aaseq_type}_info_filtered.pkl"))
        unified_tag = "unified_pplfilter" if self.use_perplexity_filtered_set else "unified"
        all_relations = pd.read_csv(os.path.join(self.data_dir,
                                                  "integrated_data",
                                                  "v1",
                                                  f"{self.aaseq_type}_{self.text_type}",
                                                  self.text_split_method,
                                                  f"{self.aaseq_type}_{self.text_type}_relations_indexed.unified.csv"))
        if self.relation_type != "all":
            # Filter dataframe by relation:
            if self.text_type != 'go':
                valid_rel = get_and_check_relation_id(
                    self.data_dir, self.relation_type
                )
                all_relations = all_relations.loc[lambda x: x.relation == valid_rel]
            else:
                all_relations = all_relations.loc[lambda x: x.text_type.str.lower() == self.relation_type.lower()]

        if self.splits_to_use[0] == 'all': # Gets all data if split specified as all
            use_relations = all_relations[["seq_id", "relation", "text_id"]].copy()
        else:
            validate_specified_splits(all_relations,
                                    f"{self.aaseq_type}<->{self.text_type}:{self.relation_type}",
                                    self.splits_to_use)
            use_relations = (all_relations
                            .loc[lambda x: x.split.isin(self.splits_to_use)]
                            [["seq_id", "relation", "text_id"]])
        if self.aaseq_subset is not None:
            contained = pd.Series(self.aaseq_subset).isin(use_relations.seq_id)
            if contained.sum() == 0:
                raise ValueError(f"AASeqTextUnifiedDataset {self.aaseq_type}<->{self.text_type}:{self.relation_type}"
                                 "specified AAseq subset not found in relations.")
            use_relations = (use_relations.loc[lambda x: x.seq_id.isin(self.aaseq_subset)])

        all_relations = (all_relations
                         [["seq_id", "relation", "text_id"]])

        if self.use_only_aaseq_text_texts:
            self.all_texts = np.unique(all_relations["text_id"].values)
        else:
            self.all_texts = text_info["index"].values
        self.num_texts = len(self.all_texts)

        if self.use_only_aaseq_text_aaseqs:
            self.all_aaseqs = np.unique(all_relations["seq_id"].values)
        else:
            self.all_aaseqs = aaseq_info["index"].values
        self.num_aaseqs = len(self.all_aaseqs)

        ## if negative sampling GOs, load GO negative sampling masks and similarity matrix
        if self.negative_sampling_strategy in ["aaseq_text_both", "text_only"]:
            all_aaseq_major_relations = list(all_relations
                                             [["seq_id", "relation", "text_id"]]
                                             .itertuples(index=False))
            used_aaseq_major_relations = list(use_relations
                                             [["seq_id", "relation", "text_id"]]
                                              .itertuples(index=False))
            self.text_negative_sampler = SimBasedNegativeSampler(
                data_dir=self.data_dir,
                data_type=self.text_type,
                sims_type=self.text_sims_type,
                num_negs=self.num_neg_samples_aaseq_text_per_text,
                num_expected=len(text_info),
                expected_mask_name=f"{self.text_type}_generic_masks.npy",
                all_relations=all_aaseq_major_relations,
                used_data=self.all_texts,
                used_relations=used_aaseq_major_relations,
                sims_to_probs_fn=partial(process_go_sims, negative_sampling_strategy=None),
                seed=self.seed,
            )
        else:
            self.text_negative_sampler = NullNegativeSampler()

        if self.negative_sampling_strategy in ["aaseq_text_both", "aaseq_only"]:
            all_text_major_relations = list(all_relations
                                             [["text_id", "relation", "seq_id"]]
                                             .itertuples(index=False))
            used_text_major_relations = list(use_relations
                                             [["text_id", "relation", "seq_id"]]
                                              .itertuples(index=False))
            self.aaseq_negative_sampler = SimBasedNegativeSampler(
                data_dir=self.data_dir,
                data_type=self.aaseq_type,
                sims_type=self.aaseq_sims_type,
                num_negs=self.num_neg_samples_aaseq_text_per_aaseq,
                num_expected=len(aaseq_info),
                expected_mask_name=f"{self.aaseq_type}_dummy_masks.npy",
                all_relations=all_text_major_relations,
                used_data=self.all_aaseqs,
                used_relations=used_text_major_relations,
                sims_to_probs_fn=partial(process_aaseq_sims, negative_sampling_strategy=None),
                seed=self.seed,
            )
        else:
            self.aaseq_negative_sampler = NullNegativeSampler()


        self.true_relations = list(use_relations.itertuples(index=False))
        if self.deduplicate_dataset == "text":
            self.aaseq_text_relations = list(use_relations.drop_duplicates("text_id").itertuples(index=False))
        elif self.deduplicate_dataset == "aaseq":
            self.aaseq_text_relations = list(use_relations.drop_duplicates("seq_id").itertuples(index=False))
        else:
            self.aaseq_text_relations = self.true_relations

        self.unique_aaseq = use_relations.seq_id.drop_duplicates()
        self.unique_text = use_relations.text_id.drop_duplicates()

    def _load_data_preset_negatives(self,
                                    num_preset_negs: Union[int, None]):
        # Since we have to read a separate file to get the preset negatives, require that
        # we just have a single split specified per dataset.
        if len(self.splits_to_use) > 1:
            raise ValueError("Negative sampling with preset negatives requires a single split with preset "
                             f"negatives to be specified, got: {self.splits_to_use}. If you want to use "
                             "a mix of data with and without preset negatives, enter in config as two separate "
                             "dataset entries.")
        split_name = self.splits_to_use[0]

        # Again, since we have to read a separate file, require a strict naming scheme that
        # matches what's currently been used for splits with preset negatives. Can also use
        # this to pull out the maximum allowed number of negatives.
        preset_regex = "with_(\d+)_negatives"
        match = re.search(preset_regex, split_name)
        if match is None:
            raise ValueError(f"Using preset negatives requires a split named like {preset_regex} to facilitate "
                             f"reading the correct file and pulling out the max number of negatives, got: {split_name}")
        max_negs = int(match.group(1))

        relations_df = pd.read_csv(os.path.join(self.data_dir,
                                                "integrated_data",
                                                "v1",
                                                f"{self.aaseq_type}_{self.text_type}",
                                                self.text_split_method,
                                                f"{self.aaseq_type}_{self.text_type}_relations_{split_name}.csv"))

        if self.relation_type != "all":
            if self.text_type != 'go':
                valid_rel = get_and_check_relation_id(
                    self.data_dir, self.relation_type
                )
                relations_df = relations_df.loc[lambda x: x.relation == valid_rel]
            else:
                relations_df = relations_df.loc[lambda x: x.text_type.str.lower() == self.relation_type.lower()]

        if num_preset_negs is not None:
            if num_preset_negs > max_negs:
                raise ValueError(f"Number of desired negatives greater than number prespecified {num_preset_negs} > {max_negs}")
            # Static indexing for now - could sample later - for reproducibility and consistency
            max_negs = num_preset_negs
        neg_strs_expanded = [f"neg_seq_id_{i}" for i in range(max_negs)]

        self.aaseq_negative_sampler = PresetNegativeSampler(list(relations_df[neg_strs_expanded].itertuples(index=False)))
        self.text_negative_sampler = RepeatNegativeSampler(relations_df.text_id.values, max_negs)

        self.aaseq_text_relations = list(relations_df[["seq_id", "relation", "text_id"]].itertuples(index=False))
        self.unique_aaseq = relations_df.seq_id.drop_duplicates()
        self.unique_text = relations_df.text_id.drop_duplicates()

    def __getitem__(self, index):
        # NOTE: Only supports random edge sampling for now
        aaseq_idx, rel_idx, text_idx = self.aaseq_text_relations[index]
        negative_aaseq_indices = self.aaseq_negative_sampler.get_negative_samples(index)
        negative_text_indices = self.text_negative_sampler.get_negative_samples(index)

        return (aaseq_idx, rel_idx, text_idx), negative_aaseq_indices, negative_text_indices

    def __len__(self):
        return len(self.aaseq_text_relations)

    def name(self):
        return "_".join((self.aaseq_type, self.text_type, self.relation_type))

def load_unified_aaseq_relations(
    data_dir: str,
    aaseq_type: str,
    unified_tag: str,
    relation_type: str,
):
    relation_info = pd.read_csv(
        os.path.join(
            data_dir,
            "integrated_data",
            "v1",
            f"{aaseq_type}_{aaseq_type}",
            f"{aaseq_type}_{aaseq_type}_relations_indexed.{unified_tag}.csv",
        )
    )
    if relation_type != "all":
        relation_id = get_and_check_relation_id(
            data_dir, f"{aaseq_type}_{relation_type}"
        )
        relation_info = relation_info.loc[lambda x: x.relation == relation_id]
        if len(relation_info) == 0:
            raise ValueError(
                f"No AASeq <-> AASeq relations: {aaseq_type} , {relation_type}"
            )
    return relation_info

class AASeqDataset(Dataset):
    """Dataset for relations between the same type of amino acid sequence.

        Represents datasets of relations between two amino acid sequences of the same type
        (e.g. protein-protein or domain-domain), as opposed to relations between two amino
        acid sequences of different types (e.g. protein-domain).
    """
    def __init__(
        self,
        data_dir: str,
        aaseq_type: str,
        relation_type: str,

        # List of splits to include in this dataset, e.g. [cl_val_pt_ft, cl_val_five_shot]
        splits_to_use: List[str],

        # choose from ['text_only', 'aaseq_text_both', 'aaseq_only', "in_batch"]
        # note that we only distinguish between neg sampling for aaseqs vs not.
        negative_sampling_strategy: str = "aaseq_only",
        aaseq_sims_type: str = None,  # choose from ['jaccard', 'k-hop', None]
        num_neg_samples_per_aaseq: int = None,
        use_only_aaseqs_w_relation: bool = True,
        # Probability of swapping AA seqs in a relation. Assuming that relations
        # are undirected and represented in the relation file a single time per
        # edge, we'd like to randomly choose which in the pair should be presented
        # first.
        swap_prob: float = 0.5,
        seed: int = 42,
        use_perplexity_filtered_set = False,
        # Store reverse of each edge as a separate relation. Typically relation
        # file is assumed to be undirected, so if set to True will store
        # (a1, a2) and (a2, a1) even if only (a1, a2) occurs in the file. This
        # can be useful in eval settings where we want to make sure each protein
        # in the pair shows up as a query.
        store_reverse_edges = False,
    ):
        # Map of valid negative sampling strategy choices to whether or not
        # we should do negative sampling within this dataset.
        negative_sampling_choices = {
            "aaseq_only": True,
            "aaseq_text_both": True,
            "text_only": False,
            "in_batch": False,
        }

        assert negative_sampling_strategy in negative_sampling_choices
        assert swap_prob >= 0 and swap_prob <= 1.0
        self.rng = np.random.default_rng(seed=seed)
        self.seed = seed
        self.swap_prob = swap_prob

        self.data_dir = data_dir
        self.aaseq_type = aaseq_type
        self.relation_type = relation_type
        self.negative_sampling = negative_sampling_choices[negative_sampling_strategy]
        self.aaseq_sims_type = aaseq_sims_type
        self.num_neg_samples_per_aaseq = num_neg_samples_per_aaseq
        self.use_only_aaseqs_w_relation = use_only_aaseqs_w_relation
        self.splits_to_use = splits_to_use
        self.use_perplexity_filtered_set = use_perplexity_filtered_set
        self.store_reverse_edges = store_reverse_edges

        self._load_data()

    def _load_data(self):
        aaseq_info = pd.read_pickle(
            os.path.join(
                self.data_dir,
                "integrated_data",
                "v1",
                self.aaseq_type,
                f"{self.aaseq_type}_info_filtered.pkl",
            )
        )
        unified_tag = "unified_pplfilter" if self.use_perplexity_filtered_set else "unified"
        relation_info = pd.read_csv(
            os.path.join(
                self.data_dir,
                "integrated_data",
                "v1",
                f"{self.aaseq_type}_{self.aaseq_type}",
                f"{self.aaseq_type}_{self.aaseq_type}_relations_indexed.{unified_tag}.csv",
            )
        )

        if self.relation_type != "all":
            relation_id = get_and_check_relation_id(
                self.data_dir, f"{self.aaseq_type}_{self.relation_type}"
            )
            relation_info = relation_info.loc[lambda x: x.relation == relation_id]
            if len(relation_info) == 0:
                raise ValueError(
                    f"No AASeq <-> AASeq relations: {self.aaseq_type} , {self.relation_type}"
                )
            if relation_info.duplicated(["seq_id_1", "relation", "seq_id_2"]).any() and self.swap_prob != 0:
                raise ValueError(f"Found duplicate edges in AA seq <-> AA seq relation {self.aaseq_type}_{self.relation_type}."
                                 "Currently we assume edges are undirected and unique in the relation file. If you have "
                                 "directed edges, you should set swap_prob to 0.")

        if self.splits_to_use[0] == 'all':
            used_relations = relation_info.copy()
        else:
            validate_specified_splits(relation_info,
                                    f"{self.aaseq_type}<->{self.aaseq_type}:{self.relation_type}",
                                    self.splits_to_use)
            used_relations = relation_info.loc[lambda x: x.split.isin(self.splits_to_use)]

        if self.store_reverse_edges:
            print(f"dataset {self.name()}: storing reverse edges and setting swap_prob to 0")
            self.swap_prob = 0
            self.aaseq_relations = list(
                used_relations
                [["seq_id_1", "relation", "seq_id_2"]]
                .itertuples(index=False)
            ) + list(
                used_relations
                [["seq_id_2", "relation", "seq_id_1"]]
                .itertuples(index=False)
            )
        else:
            self.aaseq_relations = list(
                used_relations
                [["seq_id_1", "relation", "seq_id_2"]]
                .itertuples(index=False)
            )

        ########## per-sample protein negative sampling preparation ##########
        # get all proteins available
        if self.use_only_aaseqs_w_relation:
            self.all_aaseqs = pd.concat(
                (
                    used_relations.seq_id_1,
                    used_relations.seq_id_2,
                )
            ).unique()
        else:
            self.all_aaseqs = aaseq_info["index"].values
        self.num_aaseqs = len(self.all_aaseqs)

        # if negative sampling proteins, load protein negative sampling masks and similarity matrix
        if self.negative_sampling:
            # Need to duplicate each relation such that each node appears as the first or "positive"
            # in the pair since either one could be used at sampling time.
            full_relations = pd.concat((
                (relation_info
                 [["seq_id_1", "relation", "seq_id_2", "split"]]),
                (relation_info
                 [["seq_id_2", "relation", "seq_id_1", "split"]]
                 .rename(columns={"seq_id_2": "seq_id_1", "seq_id_1": "seq_id_2"})),
            ))
            if self.splits_to_use[0] != 'all':
                used_full_relations = (full_relations
                                .loc[lambda x: x.split.isin(self.splits_to_use)]
                                .drop(columns="split"))
            else:
                used_full_relations = (full_relations.drop(columns="split"))

            full_relations = (full_relations
                              .drop(columns="split"))

            self.negative_sampler = SimBasedNegativeSampler(
                data_dir=self.data_dir,
                data_type=self.aaseq_type,
                sims_type=self.aaseq_sims_type,
                num_negs=self.num_neg_samples_per_aaseq,
                num_expected=len(aaseq_info),
                expected_mask_name=f"{self.aaseq_type}_dummy_masks.npy",
                all_relations=list(full_relations.itertuples(index=False)),
                used_data=self.all_aaseqs,
                used_relations=list(used_full_relations.itertuples(index=False)),
                sims_to_probs_fn=partial(process_aaseq_sims, negative_sampling_strategy=None),
                seed=self.seed,
            )
        else:
            self.negative_sampler = NullNegativeSampler()

    def _load_data_preset_negatives(self):
        # Note that this is currently unimplemented for aaseq <-> aaseq relations, but
        # this stripped down version of _load_testing_data() from AASeqTextDataset is
        # left here in case we want to add it in in the future.
        raise Exception("unimplemented")

        # Break down testing kwargs:
        shot_level = self.testing_kwargs["shot_level"]
        self.testing_use_preset_negatives = self.testing_kwargs.get(
            "use_preset_negatives", True
        )
        self.testing_num_negatives = self.testing_kwargs.get("num_negatives", None)

        # TODO: Perform filtering of relations based on relation type considered

        # Example file: protein_go_relations_eval_pt_ft_indexed_with_10_negatives.csv
        relation_info = pd.read_csv(
            os.path.join(
                self.data_dir,
                "integrated_data",
                "v1",
                f"{self.aaseq_type}_{self.aaseq_type}",
                f"{self.text_split_method}",
                f"{self.aaseq_type}_{self.aaseq_type}_relations_eval_{shot_level}_indexed_with_10_negatives.csv",
            )
        )
        if self.relation_type != "all":
            relation_id = get_and_check_relation_id(
                self.data_dir, f"{self.aaseq_type}_{self.relation_type}"
            )
            relation_info = relation_info.loc[lambda x: x.relation == relation_id]
            if len(relation_info) == 0:
                raise ValueError(
                    f"No aa_seq <-> aa_seq relations: {self.aaseq_type} , {self.relation_type}"
                )

        if self.testing_use_preset_negatives:
            if self.testing_num_negatives is not None:
                # Static indexing for now - could sample later - for reproducibility and consistency
                neg_strs_expanded = [
                    "neg_seq_id_{}".format(i) for i in range(self.testing_num_negatives)
                ]
            else:
                neg_strs_expanded = ["neg_seq_id_{}".format(i) for i in range(10)]
            self.aaseq_aaseq_relations = list(
                relation_info[["seq_id_1", "relation", "seq_id_2"]].itertuples(
                    index=False
                )
            )
            self.aaseq_aaseq_negative_relations = list(
                relation_info[neg_strs_expanded][
                    ["seq_id_1", "relation" "seq_id_2"]
                ].itertuples(index=False)
            )

        else:
            self.aaseq_aaseq_relations = list(
                relation_info[["seq_id_1", "relation" "seq_id_2"]].itertuples(
                    index=False
                )
            )

    def __getitem__(self, index: int):
        aaseq_idx_1, rel_idx, aaseq_idx_2 = self.aaseq_relations[index]
        neg_index = index
        if self.rng.random() <= self.swap_prob:
            aaseq_idx_1, aaseq_idx_2 = aaseq_idx_2, aaseq_idx_1
            # Index of the swapped relation in the full set of directed relations
            # is i' = i + N
            neg_index = index + len(self)
        negative_aaseq_indices = self.negative_sampler.get_negative_samples(neg_index)

        return (
            (aaseq_idx_1, rel_idx, aaseq_idx_2),
            negative_aaseq_indices,
        )

    def __len__(self):
        return len(self.aaseq_relations)

    def name(self):
        return "_".join((self.aaseq_type, self.aaseq_type, self.relation_type))

class ProteinGODataset(Dataset):
    """
    Protein-GO relations dataset for CL. Currently implemented for edge sampling only. TODO: Make it compatible with neighbor sampling based on proteins (instead of sampling relations randomly).

    It is not possible to do in-batch negative sampling for GOs, since we require that negative GOs must stay within the same namespace of the positive GO. Thus, the GO negative sampling is done for each sample separately, while the protein negative sampling, if required, is done for each batch as a whole.
    """
    def __init__(
        self,
        data_dir: str,
        go_split_method: str,
        negative_sampling_strategy: str,  # choose from ['go_only', 'protein_go_both', 'protein_only']
        protein_sims_type: str,  # choose from ['jaccard', 'k-hop', None]
        go_sims_type: str,  # choose from ['jaccard', 'k-hop', None]
        num_neg_samples_protein_go_per_protein: int,
        num_neg_samples_protein_go_per_go: int,
        use_only_goa_proteins: bool,
        use_only_goa_gos: bool,
        split = 'training',
        val_split_type = None,
        eval_split: str = None, # Don't set if you're using "training" or "val" splits (i.e. pretext splits)
        testing_kwargs: dict = None,
    ):
        assert negative_sampling_strategy in ['go_only', 'protein_go_both', 'protein_only', 'in_batch']
        self.data_dir = data_dir
        self.go_split_method = go_split_method
        self.negative_sampling_strategy = negative_sampling_strategy
        self.protein_sims_type = protein_sims_type
        self.go_sims_type = go_sims_type
        self.num_neg_samples_protein_go_per_protein = num_neg_samples_protein_go_per_protein
        self.num_neg_samples_protein_go_per_go = num_neg_samples_protein_go_per_go
        self.use_only_goa_proteins = use_only_goa_proteins
        self.use_only_goa_gos = use_only_goa_gos
        self.val_split_type = val_split_type
        self.testing_kwargs = testing_kwargs
        self.split = split

        assert self.split in {'train', 'val', 'test'}, "Split must be in {'train', 'val', 'test'}"
        print('Split is {}'.format(self.split))

        # # Decide on split
        # if testing_kwargs is not None: # Presence of this taskes precedent
        #     self.split = 'test'
        # elif (not training) and (eval_split is not None):
        #     self.split = eval_split
        # else:
        #     self.split = 'train' if training else 'val'

        if (self.split == 'test') or (self.split == 'val'):
            self._load_testing_data()
        else:
            self._load_data()

    def _load_data(self):
        go_info = pd.read_pickle(self.data_dir+"integrated_data/v1/go/go_info_filtered.pkl")
        # relation2id = pd.read_csv(data_dir+"integrated_data/v1/relation2id.csv", header=None, index_col=None).rename(columns={0:'index', 1:'relation'})
        protein_info = pd.read_pickle(self.data_dir+"integrated_data/v1/protein/protein_info_filtered.pkl")

        # protein_id, relation_type, go_id

        # load cl train pretrain relations
        if self.split == 'val':
            protein_go_relations = pd.read_csv(self.data_dir+f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_CL_val_{self.val_split_type}_indexed.csv')[['seq_id', 'relation', 'text_id']]
        else:
            protein_go_relations = pd.read_csv(self.data_dir+f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_CL_{self.split}_indexed.csv')[['seq_id', 'relation', 'text_id']]
        protein_go_relations_cl = pd.concat([
            pd.read_csv(self.data_dir+f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_CL_train_indexed.csv'),
            pd.read_csv(self.data_dir+f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_CL_val_pt_ft_indexed.csv'),
            pd.read_csv(self.data_dir+f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_CL_val_five_shot_indexed.csv'),
            pd.read_csv(self.data_dir+f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_CL_val_zero_shot_indexed.csv'),
        ])[['seq_id', 'relation', 'text_id']]

        # if self.split == 'val':
        #     protein_go_relations_cl = pd.read_csv(self.data_dir+f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_CL_val_{self.val_split_type}_indexed.csv')[['seq_id', 'relation', 'text_id']]
        # else:

        # elif self.val_split_type is not None:

        # else:
        #     protein_go_relations_cl = pd.read_csv(self.data_dir+f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_CL_eval_{self.split}_indexed.csv')[['seq_id', 'relation', 'text_id']]

        #protein_go_relations_cl = pd.concat([pd.read_csv(self.data_dir+f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_CL_train_indexed.csv'), pd.read_csv(self.data_dir+f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_CL_val_pt_ft_indexed.csv')])[['seq_id', 'relation', 'text_id']]

        # NOTE: WARNING: WE COMBINE VALIDATION WITH TRAINING ABOVE, DON'T DO THIS WITH ON-THE-FLY EVALUATION

        # TODO:
        # protein_go_relations = get_X_go_relations('protein', self.split, self.go_split_method)
        # protein_go_relations_cl = pd.concat([get_X_go_relations('protein', 'train', self.go_split_method), get_X_go_relations('protein', 'val', self.go_split_method)])

        ########## per-sample GO negative sampling preparation ##########
        # get all GOs available
        if self.use_only_goa_gos:
            # TODO: need a better way to access all_go's than loading all the dataframes above
            self.all_gos = np.unique(protein_go_relations_cl['text_id'].values)
        else:
            self.all_gos = go_info['index'].values
        self.num_gos = len(self.all_gos)

        ## if negative sampling GOs, load GO negative sampling masks and similarity matrix
        if self.negative_sampling_strategy in ['protein_go_both', 'go_only']:
            self.go_masks = np.load(self.data_dir+'generated_data/negative_sampling_masks/go_generic_masks.npy', mmap_mode='r') # In the mask, 1 indicates possible, 0 indicates impossible.
            assert self.go_masks.shape[0] == go_info.shape[0]
            self.go_sims_type = None
            if self.go_sims_type is not None:
                self.go_sims = np.load(self.data_dir+f'generated_data/negative_sampling_probs/go_sims_{self.go_sims_type}.npy', mmap_mode='r')  # only keep the sims for the GOs that are available (on axis 1 because axis 0 is used for indexing)
                assert self.go_sims.shape[0] == go_info.shape[0]
            else:
                self.go_sims = [None] * len(go_info)

            ## load ground truth GOs for each protein
            self.true_gos = dict()
            for _, prot_idx, rel_idx, go_idx in protein_go_relations_cl.itertuples():
                self.true_gos.setdefault((prot_idx, rel_idx), []).append(go_idx)  # using List instead of Set for indexing mask

        ########## per-sample protein negative sampling preparation ##########
        # get all proteins available
        if self.use_only_goa_proteins:
            self.all_proteins = np.unique(protein_go_relations_cl['seq_id'].values)
        else:
            self.all_proteins = protein_info['index'].values
        self.num_proteins = len(self.all_proteins)

        # if negative sampling proteins, load protein negative sampling masks and similarity matrix
        if self.negative_sampling_strategy in ['protein_go_both', 'protein_only']:
            # dummy 2d mask from disk
            self.protein_masks = np.load(self.data_dir+'generated_data/negative_sampling_masks/protein_dummy_masks.npy', mmap_mode='r')
            assert self.protein_masks.shape[0] == len(protein_info)
            self.protein_sims_type = None
            if self.protein_sims_type is not None:
                self.protein_sims = np.load(self.data_dir+f'generated_data/negative_sampling_probs/protein_sims_{self.protein_sims_type}.npy', mmap_mode='r')  # get the submatrix of the proteins
                assert self.protein_sims.shape[0] == len(protein_info)
            else:
                self.protein_sims = None

            ## load ground proteins for each GO
            self.true_proteins = dict()
            for _, prot_idx, rel_idx, go_idx in protein_go_relations_cl.itertuples():
                self.true_proteins.setdefault((go_idx, rel_idx), []).append(prot_idx)

        # load protein-GO relations
        self.protein_go_relations = list(protein_go_relations.itertuples())

    def _load_testing_data(self):
        # TODO: Can merge this in to _load_data

        # Break down testing kwargs:
        shot_level = self.testing_kwargs["shot_level"]
        if "use_preset_negatives" in self.testing_kwargs.keys():
            self.testing_use_preset_negatives = self.testing_kwargs["use_preset_negatives"]
        else:
            self.testing_use_preset_negatives = True

        if "num_negatives" in self.testing_kwargs.keys():
            self.testing_num_negatives = self.testing_kwargs["num_negatives"]
        else:
            self.testing_num_negatives = None

        sub_split_name = 'eval' if self.split == 'test' else 'CL_val'

        if self.testing_use_preset_negatives:
            # Example file: protein_go_relations_eval_pt_ft_indexed_with_10_negatives.csv
            protein_go_relations_df = pd.read_csv(os.path.join(self.data_dir, f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_{sub_split_name}_{shot_level}_indexed_with_10_negatives.csv'))
            if self.testing_num_negatives is not None:
                # Static indexing for now - could sample later - for reproducibility and consistency
                neg_strs_expanded = ['neg_seq_id_{}'.format(i) for i in range(self.testing_num_negatives)]
            else:
                neg_strs_expanded = ['neg_seq_id_{}'.format(i) for i in range(10)]
            self.protein_go_relations = list(protein_go_relations_df[['seq_id', 'relation', 'text_id']].itertuples())
            self.protein_go_negative_relations = list(protein_go_relations_df[neg_strs_expanded].itertuples())
        else:
            #raise NotImplementedError
            #protein_go_relations = pd.read_csv(os.path.join(self.data_dir, f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_eval_{shot_level}_indexed.csv'))[['seq_id', 'relation', 'text_id']]
            protein_go_relations = pd.read_csv(os.path.join(self.data_dir, f'integrated_data/v1/protein_go/{self.go_split_method}/protein_go_relations_{sub_split_name}_{shot_level}_indexed_with_10_negatives.csv'))[['seq_id', 'relation', 'text_id']]

            # Break into only unique GO's
            self.unique_relations_by_go = first_unique_value_in_pandas_df(protein_go_relations, col = 'text_id')
            self.unique_gos = self.unique_relations_by_go["text_id"]
            self.unique_proteins = first_unique_value_in_pandas_df(protein_go_relations, col = 'seq_id')["seq_id"]

            # load protein-GO relations
            self.protein_go_relations = list(self.unique_relations_by_go.itertuples())

    def __getitem__(self, index):
        # NOTE: Only supports random edge sampling for now.
        # First blank index ignores the index in itertuples

        if (self.split == 'test') or (self.split == 'val'):
            # Get testing positives and negatives
            negative_protein_indices = []
            negative_go_indices = []

            if self.testing_use_preset_negatives:
                _, prot_idx, rel_idx, go_idx = self.protein_go_relations[index]
                negative_protein_indices = list(self.protein_go_negative_relations[index][1:])
                negative_go_indices = [go_idx] * len(negative_protein_indices)
            else:
                _, prot_idx, rel_idx, go_idx = self.protein_go_relations[index]
                negative_protein_indices, negative_go_indices = None, None
        else:
            _, prot_idx, rel_idx, go_idx = self.protein_go_relations[index]
            negative_protein_indices, negative_go_indices = [], []
            if self.negative_sampling_strategy in {'go_only', 'protein_go_both'}:
                go_mask = np.array(self.go_masks[go_idx, :])
                go_mask[self.true_gos[(prot_idx, rel_idx)]] = 0
                go_mask = go_mask[self.all_gos]
                # print(go_idx, self.all_gos)
                go_sim = np.array(self.go_sims[go_idx, self.all_gos])

                go_prob = process_go_sims(go_sim, self.negative_sampling_strategy)
                negative_go_indices = negative_sampling_random_tail((prot_idx, rel_idx), self.num_neg_samples_protein_go_per_go, self.num_gos, go_mask, go_prob)

                # in case the GOs in GOA do not cover all GOs in the KG
                negative_go_indices = self.all_gos[negative_go_indices].tolist()

            if self.negative_sampling_strategy in {'protein_go_both', 'protein_only'}:
                protein_mask = np.array(self.protein_masks[prot_idx, :])
                protein_mask[self.true_proteins[(go_idx, rel_idx)]] = 0
                protein_mask = protein_mask[self.all_proteins]
                if self.protein_sims is None:
                    protein_sim = None
                else:
                    protein_sim = np.array(self.protein_sims[prot_idx, self.all_proteins])

                protein_prob = process_protein_sims(protein_sim, self.negative_sampling_strategy)
                negative_protein_indices = negative_sampling_random_tail((go_idx, rel_idx), self.num_neg_samples_protein_go_per_protein, self.num_proteins, protein_mask, protein_prob)

                # in case the proteins in GOA do not cover all proteins in the KG
                negative_protein_indices = self.all_proteins[negative_protein_indices].tolist()
            elif self.negative_sampling_strategy == 'in_batch':
                negative_protein_indices, negative_go_indices = None, None

        return (prot_idx, rel_idx, go_idx), negative_protein_indices, negative_go_indices

    def __len__(self):
        return len(self.protein_go_relations)


class DomainGODataset(Dataset):
    """
    Domain-GO relations dataset for CL. Currently implemented for edge sampling only.

    It is not possible to do in-batch negative sampling for GOs, since we require that negative GOs must stay within the same namespace of the positive GO. Thus, the GO negative sampling is done for each sample separately, while the domain negative sampling, if required, is done for each batch as a whole.
    """
    def __init__(
        self,
        data_dir: str,
        go_split_method: str,
        negative_sampling_strategy: str,  # choose from ['go_only', 'domain_go_both', 'domain_only']
        domain_sims_type: str,  # choose from ['esm2-650m_embeds_cosine', 'levenstein', None]
        go_sims_type: str,  # choose from ['jaccard', 'k-hop', None]
        num_neg_samples_domain_go_per_domain: int,
        num_neg_samples_domain_go_per_go: int,
        use_only_domain_go_domains: bool,
        use_only_domain_go_gos: bool,
        training: bool = True,
        eval_split: str = None, # Don't set if you're using "training" or "val" splits (i.e. pretext splits)
    ):
        assert negative_sampling_strategy in ['go_only', 'domain_go_both', 'domain_only']
        self.data_dir = data_dir
        self.go_split_method = go_split_method
        self.negative_sampling_strategy = negative_sampling_strategy
        self.go_sims_type = go_sims_type
        self.domain_sims_type = domain_sims_type
        self.num_neg_samples_domain_go_per_go = num_neg_samples_domain_go_per_go
        self.num_neg_samples_domain_go_per_domain = num_neg_samples_domain_go_per_domain
        self.use_only_domain_go_gos = use_only_domain_go_gos
        self.use_only_domain_go_domains = use_only_domain_go_domains

        if (not training) and (eval_split is not None):
            self.split = eval_split
        else:
            self.split = 'train' if training else 'val'

        self._load_data()

    def _load_data(self):
        go_info = pd.read_pickle(self.data_dir+f'integrated_data/v1/go/go_info_filtered.pkl')
        # relation2id = pd.read_csv(data_dir+"integrated_data/v1/relation2id.csv", header=None, index_col=None).rename(columns={0:'index', 1:'relation'})
        domain_info = pd.read_pickle(self.data_dir+f'integrated_data/v1/domain/domain_info_filtered.pkl')
        # load cl train pretrain relations
        domain_go_relations = pd.read_csv(self.data_dir+f'integrated_data/v1/domain_go/{self.go_split_method}/domain_go_relations_CL_{self.split}_indexed.csv')[['seq_id', 'relation', 'text_id']]
        domain_go_relations_cl = pd.concat([pd.read_csv(self.data_dir+f'integrated_data/v1/domain_go/{self.go_split_method}/domain_go_relations_CL_train_indexed.csv'), pd.read_csv(self.data_dir+f'integrated_data/v1/domain_go/{self.go_split_method}/domain_go_relations_CL_val_indexed.csv')])[['seq_id', 'relation', 'text_id']]
        # domain_go_relations = get_X_go_relations('domain', self.split, self.go_split_method)
        # domain_go_relations_cl = pd.concat([get_X_go_relations('domain', 'train', self.go_split_method), get_X_go_relations('domain', 'val', self.go_split_method)])
        ########## per-sample GO negative sampling preparation ##########
        ## get all GOs available
        if self.use_only_domain_go_gos:
            self.all_gos = np.unique(domain_go_relations_cl['text_id'].values)
        else:
            self.all_gos = go_info['index'].values
        self.num_gos = len(self.all_gos)

        ## if negative sampling GOs, load GO negative sampling masks and similarity matrix
        if self.negative_sampling_strategy in ['domain_go_both', 'go_only']:
            self.go_masks = np.load(self.data_dir+'generated_data/negative_sampling_masks/go_generic_masks.npy', mmap_mode='r') # In the mask, 1 indicates possible, 0 indicates impossible.

            assert self.go_masks.shape[0] == go_info.shape[0]

            if self.go_sims_type is not None:
                self.go_sims = np.load(self.data_dir+f'generated_data/negative_sampling_probs/go_sims_{self.go_sims_type}.npy', mmap_mode='r')  # only keep the sims for the GOs that are available (on axis 1 because axis 0 is used for indexing)
                assert self.go_sims.shape[0] == go_info.shape[0]
            else:
                self.go_sims = [None] * len(go_info)
            ## load ground truth GOs for each domain
            self.true_gos = dict()
            for _, domain_idx, rel_idx, go_idx in domain_go_relations_cl.itertuples():
                self.true_gos.setdefault((domain_idx, rel_idx), []).append(go_idx)  # using List instead of Set for indexing mask

        ########## per-sample domain negative sampling preparation ##########
        # get all proteins available
        if self.use_only_domain_go_domains:
            self.all_domains = np.unique(domain_go_relations_cl.seq_id.values)
        else:
            self.all_domains = domain_info['index'].values
        self.num_domains = len(self.all_domains)

        # if negative sampling domains, load domain negative sampling masks and similarity matrix
        if self.negative_sampling_strategy in ['domain_go_both', 'domain_only']:
            # dummy 2d mask from disk
            self.domain_masks = np.load(self.data_dir+'generated_data/negative_sampling_masks/domain_dummy_masks.npy', mmap_mode='r')
            assert self.domain_masks.shape[0] == len(domain_info)
            if self.domain_sims_type is not None:
                self.domain_sims = np.load(self.data_dir+f'generated_data/negative_sampling_probs/domain_sims_{self.domain_sims_type}.npy', mmap_mode='r')  # get the submatrix of the domains
                assert self.domain_sims.shape[0] == len(domain_info)
            else:
                self.domain_sims = None
            ## load ground domains for each GO
            self.true_domains = dict()
            for _, domain_idx, rel_idx, go_idx in domain_go_relations_cl.itertuples():
                self.true_domains.setdefault((go_idx, rel_idx), []).append(domain_idx)

        # load domain-GO relations
        self.domain_go_relations = list(domain_go_relations.itertuples())

    def __getitem__(self, index):
        # NOTE: Only supports random edge sampling for now.
        _, domain_idx, rel_idx, go_idx = self.domain_go_relations[index]
        negative_domain_indices, negative_go_indices = [], []

        if self.negative_sampling_strategy in {'domain_go_both', 'domain_only'}:
            domain_mask = np.array(self.domain_masks[domain_idx, :])
            domain_mask[self.true_domains[(go_idx, rel_idx)]] = 0
            domain_mask = domain_mask[self.all_domains]
            if self.domain_sims is not None:
                domain_sim = np.array(self.domain_sims[domain_idx, self.all_domains])
            else:
                domain_sim = None
            domain_prob = process_domain_sims(domain_sim, self.negative_sampling_strategy)
            negative_domain_indices = negative_sampling_random_tail((go_idx, rel_idx), self.num_neg_samples_domain_go_per_domain, self.num_domains, domain_mask, domain_prob)

            # in case the domains in domain-go relations do not cover all domains in the KG
            negative_domain_indices = self.all_domains[negative_domain_indices].tolist()

        if self.negative_sampling_strategy in {'go_only', 'domain_go_both'}:
            go_mask = np.array(self.go_masks[go_idx, :])
            go_mask[self.true_gos[(domain_idx, rel_idx)]] = 0
            go_mask = go_mask[self.all_gos]
            go_sim = np.array(self.go_sims[go_idx, self.all_gos])

            go_prob = process_go_sims(go_sim, self.negative_sampling_strategy)
            negative_go_indices = negative_sampling_random_tail((domain_idx, rel_idx), self.num_neg_samples_domain_go_per_go, self.num_gos, go_mask, go_prob)

            # in case the GOs in domain-go relations do not cover all GOs in the KG
            negative_go_indices = self.all_gos[negative_go_indices].tolist()

        return (domain_idx, rel_idx, go_idx), negative_domain_indices, negative_go_indices

    def __len__(self):
        return len(self.domain_go_relations)


class DomainPfamDataset(Dataset):
    """
    Domain-Pfam relations dataset for CL. Currently implemented for edge sampling only.

    The Pfam negative sampling is done for each sample separately, while the domain negative sampling, if required, is done for each batch as a whole.
    """
    def __init__(
        self,
        data_dir: str,
        pfam_split_method: str,
        negative_sampling_strategy: str,  # choose from ['pfam_only', 'domain_pfam_both', 'domain_only']
        domain_sims_type: str,
        pfam_sims_type: str,
        num_neg_samples_domain_pfam_per_domain: int,
        num_neg_samples_domain_pfam_per_pfam: int,
        use_only_domain_pfam_domains: bool,
        use_only_domain_pfam_pfams: bool,
        training: bool = True,
        eval_split: str = None, # Don't set if you're using "training" or "val" splits (i.e. pretext splits)
    ):
        assert negative_sampling_strategy in ['domain_only', 'domain_pfam_both', 'pfam_only']
        self.data_dir = data_dir
        self.pfam_split_method = pfam_split_method
        self.negative_sampling_strategy = negative_sampling_strategy
        self.domain_sims_type = domain_sims_type
        self.pfam_sims_type = pfam_sims_type
        self.num_neg_samples_domain_pfam_per_domain = num_neg_samples_domain_pfam_per_domain
        self.num_neg_samples_domain_pfam_per_pfam = num_neg_samples_domain_pfam_per_pfam
        self.use_only_domain_pfam_domains = use_only_domain_pfam_domains
        self.use_only_domain_pfam_pfams = use_only_domain_pfam_pfams
        if (not training) and (eval_split is not None):
            self.split = eval_split
        else:
            self.split = 'train' if training else 'val'

        self._load_data()

    def _load_data(self):
        pfam_info = pd.read_pickle(self.data_dir+f'integrated_data/v1/pfam/pfam_info_filtered.pkl')
        # relation2id = pd.read_csv(data_dir+"integrated_data/v1/relation2id.csv", header=None, index_col=None).rename(columns={0:'index', 1:'relation'})
        domain_info = pd.read_pickle(self.data_dir+f'integrated_data/v1/domain/domain_info_filtered.pkl')

        # load cl train pretrain relations
        domain_pfam_relations = pd.read_csv(self.data_dir+f'integrated_data/v1/domain_pfam/{self.pfam_split_method}/domain_pfam_relations_CL_{self.split}_indexed.csv')[['seq_id', 'relation', 'text_id']]
        domain_pfam_relations_cl = pd.concat([pd.read_csv(self.data_dir+f'integrated_data/v1/domain_pfam/{self.pfam_split_method}/domain_pfam_relations_CL_train_indexed.csv'), pd.read_csv(self.data_dir+f'integrated_data/v1/domain_pfam/{self.pfam_split_method}/domain_pfam_relations_CL_val_indexed.csv')])[['seq_id', 'relation', 'text_id']]

        ########## per-sample Pfam negative sampling preparation ##########
        ## get all Pfams available
        if self.use_only_domain_pfam_pfams:
            self.all_pfams = np.unique(domain_pfam_relations_cl['text_id'].values)
        else:
            self.all_pfams = pfam_info['index'].values
        self.num_pfams = len(self.all_pfams)

        ## if negative sampling Pfams, load Pfam negative sampling masks and similarity matrix
        self.pfam_masks = np.load(self.data_dir+'generated_data/negative_sampling_masks/pfam_dummy_masks.npy', mmap_mode='r') # In the mask, 1 indicates possible, 0 indicates impossible.
        assert self.pfam_masks.shape[0] == pfam_info.shape[0]

        if self.pfam_sims_type is not None:
            self.pfam_sims = np.load(self.data_dir+f'generated_data/negative_sampling_probs/pfam_sims_{self.pfam_sims_type}.npy', mmap_mode='r')  # only keep the sims for the Pfams that are available (on axis 1 because axis 0 is used for indexing)
            assert self.pfam_sims.shape[0] == pfam_info.shape[0]
        else:
            self.pfam_sims = [None] * len(pfam_info)
        ## load ground truth Pfams for each protein
        self.true_pfams = dict()
        for _, domain_idx, rel_idx, pfam_idx in domain_pfam_relations_cl.itertuples():
            self.true_pfams.setdefault((domain_idx, rel_idx), []).append(pfam_idx)  # using List instead of Set for indexing mask

        ########## per-sample domain negative sampling preparation ##########
        # get all proteins available
        if self.use_only_domain_pfam_pfams:
            self.all_domains = np.unique(domain_pfam_relations_cl.seq_id.values)
        else:
            self.all_domains = domain_info['index'].values
        self.num_domains = len(self.all_domains)

        # if negative sampling domains, load domain negative sampling masks and similarity matrix
        if self.negative_sampling_strategy in ['domain_pfam_both', 'domain_only']:
            # dummy 2d mask from disk
            self.domain_masks = np.load(self.data_dir+'generated_data/negative_sampling_masks/domain_dummy_masks.npy', mmap_mode='r')
            assert self.domain_masks.shape[0] == len(domain_info)
            if self.domain_sims_type is not None:
                self.domain_sims = np.load(self.data_dir+f'generated_data/negative_sampling_probs/domain_sims_{self.domain_sims_type}.npy', mmap_mode='r')  # get the submatrix of the domains
                assert self.domain_sims.shape[0] == len(domain_info)
            else:
                self.domain_sims = None
            ## load ground domains for each pfam
            self.true_domains = dict()
            for _, domain_idx, rel_idx, pfam_idx in domain_pfam_relations_cl.itertuples():
                self.true_domains.setdefault((pfam_idx, rel_idx), []).append(domain_idx)

        # load domain-pfam relations
        self.domain_pfam_relations = list(domain_pfam_relations.itertuples())

    def __getitem__(self, index):
        # NOTE: Only supports random edge sampling for now.
        _, domain_idx, rel_idx, pfam_idx = self.domain_pfam_relations[index]
        negative_domain_indices, negative_pfam_indices = [], []

        if self.negative_sampling_strategy in {'domain_pfam_both', 'domain_only'}:
            domain_mask = np.array(self.domain_masks[domain_idx, :])
            domain_mask[self.true_domains[(pfam_idx, rel_idx)]] = 0
            domain_mask = domain_mask[self.all_domains]
            if self.domain_sims is not None:
                domain_sim = np.array(self.domain_sims[domain_idx, self.all_domains])
            else:
                domain_sim = None
            domain_prob = process_domain_sims(domain_sim, self.negative_sampling_strategy)
            negative_domain_indices = negative_sampling_random_tail((pfam_idx, rel_idx), self.num_neg_samples_domain_pfam_per_domain, self.num_domains, domain_mask, domain_prob)

            # in case the domains in domain-pfam relations do not cover all domains in the KG
            negative_domain_indices = self.all_domains[negative_domain_indices].tolist()

        if self.negative_sampling_strategy in {'pfam_only', 'domain_pfam_both'}:
            pfam_mask = np.array(self.pfam_masks[pfam_idx, :])
            pfam_mask[self.true_pfams[(domain_idx, rel_idx)]] = 0
            pfam_mask = pfam_mask[self.all_pfams]
            pfam_sim = np.array(self.pfam_sims[pfam_idx, self.all_pfams])

            pfam_prob = process_pfam_sims(pfam_sim, self.negative_sampling_strategy)
            negative_pfam_indices = negative_sampling_random_tail((domain_idx, rel_idx), self.num_neg_samples_domain_pfam_per_pfam, self.num_pfams, pfam_mask, pfam_prob)

            # the sampled indices are indices for self.all_pfams
            negative_pfam_indices = self.all_pfams[negative_pfam_indices].tolist()


        return (domain_idx, rel_idx, pfam_idx), negative_domain_indices, negative_pfam_indices

    def __len__(self):
        return len(self.domain_pfam_relations)


class ProteinProteinDataset(Dataset):
    """
    Protein-Protein Interaction Dataset for CL. Currently implemented for edge sampling only. TODO: Make it compatible with neighbor sampling based on proteins (instead of sampling relations randomly).
    """
    def __init__(
        self,
        data_dir,
        training=True,
        use_only_ppi_proteins=True
    ):
        self.data_dir = data_dir
        self.split = 'train' if training else 'val'
        self.use_only_ppi_proteins = use_only_ppi_proteins

        self._load_data()

    def _load_data(self):
        # relation2id = pd.read_csv(self.data_dir+"integrated_data/v1/relation2id.csv", header=None, index_col=None)
        # protein2id = pd.read_csv(self.data_dir+"integrated_data/v1/protein2id.csv", header=None, index_col=None)
        # self.num_relations = len(relation2id)
        # assert protein2id[1][0] == 'P31946'

        # load protein-protein relations
        protein_protein_relations = pd.read_csv(self.data_dir+f'integrated_data/v1/protein_protein/protein_protein_relations_CL_{self.split}_indexed.csv')[['src', 'relation', 'dst']]
        self.protein_protein_relations = list(protein_protein_relations.itertuples())

        # # get all proteins available
        # if self.use_only_ppi_proteins:
        #     self.all_proteins = np.unique(protein_protein_relations[['src', 'dst']].values)
        # else:
        #     self.all_proteins = protein2id[0].values
        # self.num_proteins = len(self.all_proteins)

    def __getitem__(self, index):
        _, head_prot_idx, rel_idx, tail_prot_idx = self.protein_protein_relations[index]

        return head_prot_idx, rel_idx, tail_prot_idx

    # def _negative_sampling(self, head_rel: Tuple[int, int], tail_rel: Tuple[int, int], num_neg_sample: int, true_proteins: dict, num_proteins, protein_mask_head: np.ndarray, protein_mask_tail: np.ndarray, protein_sim_head: np.ndarray = None, protein_sim_tail: np.ndarray = None) -> List[Tuple[int, int, int]]:
    #     if self.negative_sampling_strategy == 'protein_both':
    #         protein_probs_head = process_protein_sims(protein_sim_head)
    #         protein_probs_tail = process_protein_sims(protein_sim_tail)

    #         negative_protein_for_head_indices = negative_sampling_random_tail(head_rel, num_neg_sample, true_proteins[head_rel], num_proteins, protein_mask_head, protein_probs_head)
    #         negative_protein_for_tail_indices = negative_sampling_random_tail(tail_rel, num_neg_sample, true_proteins[tail_rel], num_proteins, protein_mask_tail, protein_probs_tail)

    #         negative_samples_head = ([head_rel[0]] * num_neg_sample, [head_rel[1]] * num_neg_sample, negative_protein_for_head_indices)
    #         negative_samples_tail = (negative_protein_for_tail_indices, [tail_rel[1]] * num_neg_sample, [tail_rel[0]] * num_neg_sample)

    #     else:
    #         raise NotImplementedError

    #     return negative_samples_head, negative_samples_tail

    def __len__(self):
        return len(self.protein_protein_relations)


