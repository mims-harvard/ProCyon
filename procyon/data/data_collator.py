import pickle

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from Bio import SeqIO
from esm.data import BatchConverter
from torch_geometric.utils import is_undirected

from procyon.data.data_utils import (
    process_protein_sims,
    convert_batch_protein,
    convert_batch_text,
)
from procyon.data.sampling import negative_sampling_random_tail


class ProteinMLMCollator:
    """
    Protein sequences collator for MLM. Conducts:
    1. Protein tokenization
    2. Masking
    """

    def __init__(
        self,
        data_dir: str,
        is_protein_tokenized: bool,
        protein_tokenizer: object = None,
        max_protein_len: int = None,  # by default, no truncation
        mlm: bool = True,
        masking_strategy: str = "esm2",  # choose from ['random', 'esm2']
        mlm_probability: float = 0.15,
    ):
        self.data_dir = data_dir
        self.is_protein_tokenized = is_protein_tokenized
        self.protein_tokenizer = protein_tokenizer
        self.max_protein_len = max_protein_len
        self.mlm = mlm
        self.masking_strategy = masking_strategy
        self.mlm_probability = mlm_probability

        self._load_data()

    def _load_data(self):
        if not self.is_protein_tokenized:
            self.protein_sequences = [
                str(seq.seq)
                for seq in SeqIO.parse(
                    self.data_dir + "integrated_data/v1/protein/protein_sequences.fa",
                    "fasta",
                )
            ]
            self.protein_tokens = None
        else:
            raise NotImplementedError

        if not self.is_protein_tokenized:
            self.batch_converter = BatchConverter(
                self.protein_tokenizer, truncation_seq_length=self.max_protein_len
            )

        self.special_tokens_ids = torch.tensor(
            self.protein_tokenizer.encode(
                "".join(self.protein_tokenizer.all_special_tokens)
            )
        )
        self.standard_tokens_ids = torch.tensor(
            self.protein_tokenizer.encode("".join(self.protein_tokenizer.standard_toks))
        )

    def __call__(
        self,
        batch_input: List[int],
    ) -> Dict[str, torch.Tensor]:
        batch_toks = self._convert_batch(batch_input)

        batch_output = dict()
        batch_output["indices"] = (
            batch_input  # Add indices to batch input for reference
        )
        if self.mlm:
            batch_output["data"], batch_output["labels"] = self._mask_tokens(
                batch_toks,
                mask_token_id=self.protein_tokenizer.mask_idx,
                special_tokens_ids=self.special_tokens_ids,
                standard_tokens_ids=self.standard_tokens_ids,
                masking_strategy=self.masking_strategy,
                mlm_probability=self.mlm_probability,
            )
        else:
            batch_output["data"] = batch_toks

        # output has keys: input_ids, labels
        return batch_output

    def _convert_batch(self, batch: List[int]) -> torch.Tensor:
        batch_toks = convert_batch_protein(
            batch,
            self.is_protein_tokenized,
            self.batch_converter,
            self.protein_sequences,
            self.protein_tokens,
            self.protein_tokenizer,
            self.max_protein_len,
        )

        return batch_toks

    def _mask_tokens(
        self,
        inputs: torch.Tensor,
        mask_token_id: int,
        special_tokens_ids: torch.Tensor,
        standard_tokens_ids: torch.Tensor,
        masking_strategy: str = "esm2",
        mlm_probability: float = 0.15,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mask tokens for some percentage of the sequences in the batch.

        Args:
            inputs: torch.Tensor of shape (batch_size, max(seq_len))
            mask_token_id: int
            special_tokens_ids: torch.Tensor
            standard_tokens_ids: torch.Tensor
            masking_strategy: choose from ['esm2', 'random'] -- esm2 (essentially RoBERTa) --> 15% MASK, among which 80% replaced with <mask>, 10% random, 10% original; random --> 15% MASK, all of which are replaced with <mask>
            mlm_probability: probability of masking each token for the whole batch
        """
        labels = inputs.clone()
        probability_matrix = torch.full(labels.size(), fill_value=mlm_probability)

        special_tokens_mask = torch.isin(labels, special_tokens_ids)
        probability_matrix.masked_fill_(special_tokens_mask, value=0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        # only compute loss on masked tokens.
        labels[~masked_indices] = -100  # for CELoss param ignore_index = -100

        if masking_strategy == "random":
            inputs[masked_indices] = mask_token_id

        elif masking_strategy == "esm2":
            # 80% of time, replace masked input tokens with mask token
            indices_replaced_mask = (
                torch.bernoulli(torch.full(labels.shape, fill_value=0.8)).bool()
                & masked_indices
            )
            inputs[indices_replaced_mask] = mask_token_id

            # 10% of time, replace masked input tokens with random (standard) word
            indices_replaced_random = (
                torch.bernoulli(torch.full(labels.shape, fill_value=0.5)).bool()
                & masked_indices
                & ~indices_replaced_mask
            )
            # random_words = torch.randint(len(self.alphabet), labels.shape, dtype=torch.long)
            random_words = standard_tokens_ids[
                torch.randint(len(standard_tokens_ids), labels.shape, dtype=torch.long)
            ]
            inputs[indices_replaced_random] = random_words[indices_replaced_random]

        else:
            raise NotImplementedError

        return inputs, labels


from procyon.data.data_utils import convert_batch_text


class TextCLCollator:
    """
    Text collator for CL (only supports GO for now):
    """

    def __init__(
        self,
        data_dir: str,
        go_split_method: str,
        go_tokenizer: object = None,
        max_go_len: int = None,  # by default, no truncation
        unsupervised_only: bool = False,
    ):
        self.data_dir = data_dir
        self.go_split_method = go_split_method
        self.go_tokenizer = go_tokenizer
        self.max_go_len = max_go_len
        self.unsupervised_only = unsupervised_only

        self.GOLD_COL = "def"
        # If unsupervised, both inputs to model should be the same (i.e. gold)
        self.AUGMENTED_COLS = (
            [self.GOLD_COL]
            if unsupervised_only
            else ["desc_concise_summary", "desc_rephrase", "desc_extend_multiple"]
        )

        self._load_data()

    def _load_data(self):
        self.sequences = pd.read_pickle(
            self.data_dir + "generated_data/node_data/go/go_descriptions.pkl"
        )

    def __call__(
        self,
        indices: List[int],  # positive relation, negative GOs
    ) -> Dict[str, torch.Tensor]:

        bs = len(indices)

        gold_sequences = np.array(self.sequences[self.GOLD_COL][indices])
        augmented_sequences = np.array(
            [
                self.sequences[np.random.choice(self.AUGMENTED_COLS)][idx]
                for idx in indices
            ]
        )
        all_sequences = np.concatenate([gold_sequences, augmented_sequences])

        # Create new index since we are passing the exact sequences to use
        toks, mask = convert_batch_text(
            np.arange(2 * bs),
            is_text_tokenized=False,
            text_sequences=all_sequences,
            text_tokens=None,
            text_tokenizer=self.go_tokenizer,
            max_text_len=self.max_go_len,
        )

        # Reshape the stacked results to the desired shape (bs, 2, toks) (to match SimCSE format)
        def reshape(x):
            # Reshape x into (2, bs, toks)
            x_reshaped = x.contiguous().view(2, bs, x.shape[1])
            # Permute the axes to get the desired shape (bs, 2, toks)
            return x_reshaped.permute(1, 0, 2)

        toks = reshape(toks)
        mask = reshape(mask)

        return {
            # NOTE: Format differs to other collators
            # Both input_ids and attention_mask are tensors of shape (batch size, 2, max len in batch)
            #       where the 2 represents a positive pair
            # This aligns with the SimCSE format
            "input_ids": toks,
            "attn_masks": mask,
        }


class ProteinGOCLCollator:
    """
    Protein-GO relations collator for CL. Conducts:
    1. Negative sampling (for proteins only when required, negative GOs are sampled per-sample back in ProteinGODataset).
    2. Unique entity extraction and protein and GO tokenization or embeddings extraction.
    """

    def __init__(
        self,
        data_dir: str,
        go_split_method: str,
        negative_sampling_strategy: str,  # choose from ['go_only', 'protein_go_both', 'protein_only']
        protein_sims_type: str,  # choose from ['esm2-650m_embeds_cosine', 'levenstein', None]
        num_neg_samples_protein_go_per_protein: int,
        use_only_goa_proteins: bool,
        is_protein_tokenized: bool,
        is_go_tokenized: bool,
        use_go_embeddings: bool,
        use_protein_embeddings: bool,
        go_def_col: str,
        protein_tokenizer: object = None,
        go_tokenizer: object = None,
        max_protein_len: int = None,
        max_go_len: int = None,  # by default, no truncation
    ):
        self.data_dir = data_dir
        self.go_split_method = go_split_method
        self.negative_sampling_strategy = negative_sampling_strategy
        self.protein_sims_type = protein_sims_type
        self.num_neg_samples_protein_go_per_protein = (
            num_neg_samples_protein_go_per_protein
        )
        self.use_only_goa_proteins = use_only_goa_proteins

        self.is_protein_tokenized = is_protein_tokenized
        self.is_go_tokenized = is_go_tokenized
        self.use_go_embeddings = use_go_embeddings
        self.use_protein_embeddings = use_protein_embeddings

        self.go_def_col = go_def_col
        self.protein_tokenizer = protein_tokenizer
        self.go_tokenizer = go_tokenizer
        self.max_protein_len = max_protein_len
        self.max_go_len = max_go_len

        self._load_data()

    def _load_data(self):

        # protein and GO sequences/tokens/embeddings
        if not self.is_protein_tokenized:
            self.protein_sequences = [
                str(seq.seq)
                for seq in SeqIO.parse(
                    self.data_dir + "integrated_data/protein_sequences.fa", "fasta"
                )
            ]
            self.protein_tokens = None
        else:
            raise NotImplementedError

        if self.use_go_embeddings:
            self.go_sequences = None
            self.go_tokens = None
        elif not self.is_go_tokenized:
            self.go_sequences = pd.read_pickle(
                self.data_dir + "generated_data/node_data/go/go_descriptions.pkl"
            )[self.go_def_col].values
            self.go_tokens = None
        else:
            raise NotImplementedError

        if not self.is_protein_tokenized:
            self.batch_converter = BatchConverter(
                self.protein_tokenizer, truncation_seq_length=self.max_protein_len
            )

    def __call__(
        self,
        batch_input: List[
            Tuple[Tuple[int], List[int]]
        ],  # positive relation, negative GOs
    ) -> Dict[str, torch.Tensor]:
        # (sampling B*N_negative and excluding false negatives)
        positive_proteins = [sample[0][0] for sample in batch_input]
        positive_relations = [sample[0][1] for sample in batch_input]
        positive_gos = [sample[0][2] for sample in batch_input]
        negative_proteins = sum([sample[1] for sample in batch_input], start=[])
        negative_gos = sum(
            [sample[2] for sample in batch_input], start=[]
        )  # NOTE: Originally List of Lists of negative GO ids. We can reshape it first and don't need to reshape it back because the negative component in our KEPLER CL loss function is a plain sum over the batch

        # get unique positive and negative protein and GO indices, and map relations to new ids, also fetch corresponding sequences/tokens/embeddings
        # With saved protein embeddings, treat the same way as saved GO embeddings
        unique_protein_indices, new_id_protein_mapping = torch.unique(
            torch.LongTensor(positive_proteins + negative_proteins), return_inverse=True
        )
        positive_proteins_new, negative_proteins_new = torch.split(
            new_id_protein_mapping, [len(positive_proteins), len(negative_proteins)]
        )
        if not self.use_protein_embeddings:
            unique_protein_toks = self._convert_batch(
                "sequence", unique_protein_indices.tolist()
            )
        else:
            unique_protein_toks = None

        unique_go_indices, new_id_go_mapping = torch.unique(
            torch.LongTensor(positive_gos + negative_gos), return_inverse=True
        )
        positive_gos_new, negative_gos_new = torch.split(
            new_id_go_mapping, [len(positive_gos), len(negative_gos)]
        )
        if not self.use_go_embeddings:
            unique_go_toks, unique_go_attn_masks = self._convert_batch(
                "text", unique_go_indices.tolist()
            )  # toks
        else:
            unique_go_toks, unique_go_attn_masks = None, None

        positive_relations_new = torch.LongTensor(positive_relations)

        # collect all negative relations. first consider those that permutate GOs, then those that permutate proteins.
        negative_proteins_final = torch.cat(
            [
                torch.repeat_interleave(
                    positive_proteins_new,
                    len(negative_gos_new) // len(positive_gos_new),
                ),
                negative_proteins_new,
            ]
        )
        negative_relations_final = torch.cat(
            [
                torch.repeat_interleave(
                    positive_relations_new,
                    len(negative_gos_new) // len(positive_gos_new),
                ),
                torch.repeat_interleave(
                    positive_relations_new,
                    len(negative_proteins_new) // len(positive_proteins_new),
                ),
            ]
        )
        negative_gos_final = torch.cat(
            [
                negative_gos_new,
                torch.repeat_interleave(
                    positive_gos_new,
                    len(negative_proteins_new) // len(positive_proteins_new),
                ),
            ]
        )

        return {
            # entity tokens. 2d tensors of (num unique entities, max len in the batch)
            "toks": {"sequence": unique_protein_toks, "text": unique_go_toks},
            # entity indices. 1d tensor of (num unique entites, )
            "indices": {"sequence": unique_protein_indices, "text": unique_go_indices},
            # attention masks
            "attn_masks": {
                "sequence": None,  # because we are using Encoder only
                "text": unique_go_attn_masks,  # GPT
            },
            "relations": {
                # positive relations.  ids have been remapped to the indices as in 'node_toks' or 'node_embeddings', except for relations, which indexes decoder.  1d tensors of size B.
                "positive_relations": {
                    "sequence": positive_proteins_new,
                    "relation": positive_relations_new,
                    "text": positive_gos_new,
                },
                # negative relations. ids have been remapped to the indices as in 'node_toks' or 'node_embeddings', except for relations, which indexes decoder.  1d tensors of size B x Neg (batch_size * num_neg_per_sample).
                "negative_relations": {
                    "sequence": negative_proteins_final,
                    "relation": negative_relations_final,
                    "text": negative_gos_final,
                },
            },
        }

    def _convert_batch(self, entity_type: str, unique_indices: List[int]):
        if entity_type == "sequence":
            batch_toks = convert_batch_protein(
                unique_indices,
                self.is_protein_tokenized,
                self.batch_converter,
                self.protein_sequences,
                self.protein_tokens,
                self.protein_tokenizer,
                self.max_protein_len,
            )
            return batch_toks
        elif entity_type == "text":
            batch_toks, batch_attn_masks = convert_batch_text(
                unique_indices,
                self.is_go_tokenized,
                self.go_sequences,
                self.go_tokens,
                self.go_tokenizer,
                self.max_go_len,
            )
            return batch_toks, batch_attn_masks


class DomainGOCLCollator:
    """
    Domain-GO relations collator for CL. Conducts:
    1. Negative sampling (for domains only when required, negative GOs are sampled per-sample back in DomainGODataset).
    2. Unique entity extraction and domain and GO tokenization or embeddings extraction.
    """

    def __init__(
        self,
        data_dir: str,
        go_split_method: str,
        negative_sampling_strategy: str,  # choose from ['go_only', 'domain_go_both', 'domain_only']
        domain_sims_type: str,  # choose from ['esm2-650m_embeds_cosine', 'levenstein', None]
        num_neg_samples_domain_go_per_domain: int,
        use_only_domain_go_domains: bool,
        is_domain_tokenized: bool,
        is_go_tokenized: bool,
        use_go_embeddings: bool,
        use_domain_embeddings: bool,
        go_def_col: str,
        domain_tokenizer: object = None,
        go_tokenizer: object = None,
        max_domain_len: int = None,
        max_go_len: int = None,  # by default, no truncation
    ):
        self.data_dir = data_dir
        self.go_split_method = go_split_method
        self.negative_sampling_strategy = negative_sampling_strategy
        self.domain_sims_type = domain_sims_type
        self.num_neg_samples_domain_go_per_domain = num_neg_samples_domain_go_per_domain
        self.use_only_domain_go_domains = use_only_domain_go_domains

        self.is_domain_tokenized = is_domain_tokenized
        self.is_go_tokenized = is_go_tokenized
        self.use_go_embeddings = use_go_embeddings
        self.use_domain_embeddings = use_domain_embeddings
        self.go_def_col = go_def_col

        self.domain_tokenizer = domain_tokenizer
        self.go_tokenizer = go_tokenizer
        self.max_domain_len = max_domain_len
        self.max_go_len = max_go_len

        self._load_data()

    def _load_data(self):
        # domain and GO sequences/tokens/embeddings
        if not self.is_domain_tokenized:
            self.domain_sequences = [
                str(seq.seq)
                for seq in SeqIO.parse(
                    self.data_dir + "integrated_data/domain_sequences.fa", "fasta"
                )
            ]
            self.domain_tokens = None
        else:
            raise NotImplementedError

        if self.use_go_embeddings:
            self.go_sequences = None
            self.go_tokens = None
        elif not self.is_go_tokenized:
            self.go_sequences = pd.read_pickle(
                self.data_dir + "generated_data/node_data/go/go_descriptions.pkl"
            )[self.go_def_col].values
            self.go_tokens = None
        else:
            raise NotImplementedError

        if not self.is_domain_tokenized:
            self.batch_converter = BatchConverter(
                self.domain_tokenizer, truncation_seq_length=self.max_domain_len
            )

    def __call__(
        self,
        batch_input: List[
            Tuple[Tuple[int], List[int]]
        ],  # positive relation, negative GOs
    ) -> Dict[str, torch.Tensor]:
        # (sampling B*N_negative and excluding false negatives)
        positive_domains = [sample[0][0] for sample in batch_input]
        positive_relations = [sample[0][1] for sample in batch_input]
        positive_gos = [sample[0][2] for sample in batch_input]
        negative_domains = sum([sample[1] for sample in batch_input], start=[])
        negative_gos = sum(
            [sample[2] for sample in batch_input], start=[]
        )  # NOTE: Originally List of Lists of negative GO ids. We can reshape it first and don't need to reshape it back because the negative component in our KEPLER CL loss function is a plain sum over the batch

        # get unique positive and negative domain and GO indices, and map relations to new ids, also fetch corresponding sequences/tokens/embeddings
        # With saved domain embeddings, treat the same way as saved GO embeddings
        unique_domain_indices, new_id_domain_mapping = torch.unique(
            torch.LongTensor(positive_domains + negative_domains), return_inverse=True
        )
        positive_domains_new, negative_domains_new = torch.split(
            new_id_domain_mapping, [len(positive_domains), len(negative_domains)]
        )
        if not self.use_domain_embeddings:
            unique_domain_toks = self._convert_batch(
                "sequence", unique_domain_indices.tolist()
            )
        else:
            unique_domain_toks = None

        unique_go_indices, new_id_go_mapping = torch.unique(
            torch.LongTensor(positive_gos + negative_gos), return_inverse=True
        )
        positive_gos_new, negative_gos_new = torch.split(
            new_id_go_mapping, [len(positive_gos), len(negative_gos)]
        )
        if not self.use_go_embeddings:
            unique_go_toks, unique_go_attn_masks = self._convert_batch(
                "text", unique_go_indices.tolist()
            )
        else:
            unique_go_toks, unique_go_attn_masks = None, None

        positive_relations_new = torch.LongTensor(positive_relations)

        # collect all negative relations. first consider those that permutate GOs, then those that permutate proteins.
        negative_domains_final = torch.cat(
            [
                torch.repeat_interleave(
                    positive_domains_new, len(negative_gos_new) // len(positive_gos_new)
                ),
                negative_domains_new,
            ]
        )
        negative_relations_final = torch.cat(
            [
                torch.repeat_interleave(
                    positive_relations_new,
                    len(negative_gos_new) // len(positive_gos_new),
                ),
                torch.repeat_interleave(
                    positive_relations_new,
                    len(negative_domains_new) // len(positive_domains_new),
                ),
            ]
        )
        negative_gos_final = torch.cat(
            [
                negative_gos_new,
                torch.repeat_interleave(
                    positive_gos_new,
                    len(negative_domains_new) // len(positive_domains_new),
                ),
            ]
        )

        return {
            # entity tokens. 2d tensors of (num unique entities, max len in the batch)
            "toks": {"sequence": unique_domain_toks, "text": unique_go_toks},
            # entity indices. 1d tensor of (num unique entites, )
            "indices": {"sequence": unique_domain_indices, "text": unique_go_indices},
            # attention masks
            "attn_masks": {
                "sequence": None,  # because we are using Encoder only
                "text": unique_go_attn_masks,  # GPT
            },
            "relations": {
                # positive relations.  ids have been remapped to the indices as in 'node_toks' or 'node_embeddings', except for relations, which indexes decoder.  1d tensors of size B.
                "positive_relations": {
                    "sequence": positive_domains_new,
                    "relation": positive_relations_new,
                    "text": positive_gos_new,
                },
                # negative relations. ids have been remapped to the indices as in 'node_toks' or 'node_embeddings', except for relations, which indexes decoder.  1d tensors of size B x Neg (batch_size * num_neg_per_sample).
                "negative_relations": {
                    "sequence": negative_domains_final,
                    "relation": negative_relations_final,
                    "text": negative_gos_final,
                },
            },
        }

    def _convert_batch(self, entity_type: str, unique_indices: List[int]):
        if entity_type == "sequence":
            batch_toks = convert_batch_protein(
                unique_indices,
                self.is_domain_tokenized,
                self.batch_converter,
                self.domain_sequences,
                self.domain_tokens,
                self.domain_tokenizer,
                self.max_domain_len,
            )
            return batch_toks
        elif entity_type == "text":
            batch_toks, batch_attn_masks = convert_batch_text(
                unique_indices,
                self.is_go_tokenized,
                self.go_sequences,
                self.go_tokens,
                self.go_tokenizer,
                self.max_go_len,
            )
            return batch_toks, batch_attn_masks


class DomainPfamCLCollator:
    """
    Domain-Pfam relations collator for CL. Conducts:
    1. Negative sampling (for domains only when required, negative Pfams are sampled per-sample back in DomainPfamDataset).
    2. Unique entity extraction and domain and Pfam tokenization or embeddings extraction.
    """

    def __init__(
        self,
        data_dir: str,
        pfam_split_method: str,
        negative_sampling_strategy: str,  # choose from ['pfam_only', 'domain_pfam_both', 'domain_only']
        domain_sims_type: str,  # choose from ['esm2-650m_embeds_cosine', 'levenstein', None]
        num_neg_samples_domain_pfam_per_domain: int,
        use_only_domain_pfam_domains: bool,
        is_domain_tokenized: bool,
        is_pfam_tokenized: bool,
        use_pfam_embeddings: bool,
        use_domain_embeddings: bool,
        domain_tokenizer: object = None,
        pfam_tokenizer: object = None,
        max_domain_len: int = None,
        max_pfam_len: int = None,  # by default, no truncation
    ):
        self.data_dir = data_dir
        self.pfam_split_method = pfam_split_method
        self.negative_sampling_strategy = negative_sampling_strategy
        self.domain_sims_type = domain_sims_type
        self.num_neg_samples_domain_pfam_per_domain = (
            num_neg_samples_domain_pfam_per_domain
        )
        self.use_only_domain_pfam_domains = use_only_domain_pfam_domains

        self.is_domain_tokenized = is_domain_tokenized
        self.is_pfam_tokenized = is_pfam_tokenized
        self.use_pfam_embeddings = use_pfam_embeddings
        self.use_domain_embeddings = use_domain_embeddings

        self.domain_tokenizer = domain_tokenizer
        self.pfam_tokenizer = pfam_tokenizer
        self.max_domain_len = max_domain_len
        self.max_pfam_len = max_pfam_len

        self._load_data()

    def _load_data(self):
        pfam_info = pd.read_pickle(
            self.data_dir + "integrated_data/v1/pfam/pfam_info_filtered.pkl"
        )
        domain_info = pd.read_pickle(
            self.data_dir + "integrated_data/v1/domain/domain_info_filtered.pkl"
        )

        # domain and Pfam sequences/tokens/embeddings
        if not self.is_domain_tokenized:
            self.domain_sequences = [
                str(seq.seq)
                for seq in SeqIO.parse(
                    self.data_dir + "integrated_data/domain_sequences.fa", "fasta"
                )
            ]
            self.domain_tokens = None
        else:
            raise NotImplementedError

        if self.use_pfam_embeddings:
            self.pfam_sequences = None
            self.pfam_tokens = None
        elif not self.is_pfam_tokenized:
            self.pfam_sequences = pd.read_pickle(
                self.data_dir
                + "generated_data/node_embeddings/pfam/pfam_plus_interpro_description_embeddings_BioGPT-Large_idmap.pkl"
            )
            self.pfam_sequences = self.pfam_sequences["description_combined"].values
            self.pfam_tokens = None
        else:
            raise NotImplementedError

        if not self.is_domain_tokenized:
            self.batch_converter = BatchConverter(
                self.domain_tokenizer, truncation_seq_length=self.max_domain_len
            )

    def __call__(
        self,
        batch_input: List[
            Tuple[Tuple[int], List[int]]
        ],  # positive relation, negative Pfams
    ) -> Dict[str, torch.Tensor]:
        # (sampling B*N_negative and excluding false negatives)
        positive_domains = [sample[0][0] for sample in batch_input]
        positive_relations = [sample[0][1] for sample in batch_input]
        positive_pfams = [sample[0][2] for sample in batch_input]
        negative_domains = sum([sample[1] for sample in batch_input], start=[])
        negative_pfams = sum(
            [sample[2] for sample in batch_input], start=[]
        )  # NOTE: Originally List of Lists of negative Pfam ids. We can reshape it first and don't need to reshape it back because the negative component in our KEPLER CL loss function is a plain sum over the batch

        # get unique positive and negative domain and Pfam indices, and map relations to new ids, also fetch corresponding sequences/tokens/embeddings
        # With saved domain embeddings, treat the same way as saved Pfam embeddings
        unique_domain_indices, new_id_domain_mapping = torch.unique(
            torch.LongTensor(positive_domains + negative_domains), return_inverse=True
        )
        positive_domains_new, negative_domains_new = torch.split(
            new_id_domain_mapping, [len(positive_domains), len(negative_domains)]
        )
        if not self.use_domain_embeddings:
            unique_domain_toks = self._convert_batch(
                "sequence", unique_domain_indices.tolist()
            )
        else:
            unique_domain_toks = None

        unique_pfam_indices, new_id_pfam_mapping = torch.unique(
            torch.LongTensor(positive_pfams + negative_pfams), return_inverse=True
        )
        positive_pfams_new, negative_pfams_new = torch.split(
            new_id_pfam_mapping, [len(positive_pfams), len(negative_pfams)]
        )
        if not self.use_pfam_embeddings:
            unique_pfam_toks, unique_pfam_attn_masks = self._convert_batch(
                "text", unique_pfam_indices.tolist()
            )
        else:
            unique_pfam_toks, unique_pfam_attn_masks = None, None

        positive_relations_new = torch.LongTensor(positive_relations)

        # collect all negative relations. first consider those that permutate Pfams, then those that permutate proteins.
        negative_domains_final = torch.cat(
            [
                torch.repeat_interleave(
                    positive_domains_new,
                    len(negative_pfams_new) // len(positive_pfams_new),
                ),
                negative_domains_new,
            ]
        )
        negative_relations_final = torch.cat(
            [
                torch.repeat_interleave(
                    positive_relations_new,
                    len(negative_pfams_new) // len(positive_pfams_new),
                ),
                torch.repeat_interleave(
                    positive_relations_new,
                    len(negative_domains_new) // len(positive_domains_new),
                ),
            ]
        )
        negative_pfams_final = torch.cat(
            [
                negative_pfams_new,
                torch.repeat_interleave(
                    positive_pfams_new,
                    len(negative_domains_new) // len(positive_domains_new),
                ),
            ]
        )

        return {
            # entity tokens. 2d tensors of (num unique entities, max len in the batch)
            "toks": {"sequence": unique_domain_toks, "text": unique_pfam_toks},
            # entity indices (original). 1d tensor of (num unique entites, )
            "indices": {"sequence": unique_domain_indices, "text": unique_pfam_indices},
            # attention masks
            "attn_masks": {
                "sequence": None,  # because we are using Encoder only
                "text": unique_pfam_attn_masks,  # GPT
            },
            "relations": {
                # positive relations.  ids have been remapped to the indices as in 'node_toks' or 'node_embeddings', except for relations, which indexes decoder.  1d tensors of size B.
                "positive_relations": {
                    "sequence": positive_domains_new,
                    "relation": positive_relations_new,
                    "text": positive_pfams_new,
                },
                # negative relations. ids have been remapped to the indices as in 'node_toks' or 'node_embeddings', except for relations, which indexes decoder.  1d tensors of size B x Neg (batch_size * num_neg_per_sample).
                "negative_relations": {
                    "sequence": negative_domains_final,
                    "relation": negative_relations_final,
                    "text": negative_pfams_final,
                },
            },
        }

    def _convert_batch(self, entity_type: str, unique_indices: List[int]):
        if entity_type == "sequence":
            batch_toks = convert_batch_protein(
                unique_indices,
                self.is_domain_tokenized,
                self.batch_converter,
                self.domain_sequences,
                self.domain_tokens,
                self.domain_tokenizer,
                self.max_domain_len,
            )
        elif entity_type == "text":
            batch_toks = convert_batch_text(
                unique_indices,
                self.is_pfam_tokenized,
                self.pfam_sequences,
                self.pfam_tokens,
                self.pfam_tokenizer,
                self.max_pfam_len,
            )

        return batch_toks


class ProteinProteinCLCollator:
    """
    Protein-Protein relations collator for CL. Conducts:
    1. Negative sampling. Sample `num_neg_samples_per_relation_end` of proteins for each end of each relations, then duplicate those negative samples for all ends of all relations. Mask all proteins within batch.
    2. "Unique" protein extraction and tokenization.
    """

    def __init__(
        self,
        data_dir: str,
        negative_sampling_strategy="protein_both",  # choose from ['protein_both']
        protein_sims_type="esm2-650m_embeds_cosine",  # choose from ['esm2-650m_embeds_cosine', None]
        num_neg_samples_protein_protein_per_protein=4,
        is_protein_tokenized: bool = False,
        use_only_ppi_proteins=True,
        protein_tokenizer: object = None,
        use_protein_embeddings: bool = False,
        max_protein_len: int = None,
    ):
        self.data_dir = data_dir
        self.negative_sampling_strategy = negative_sampling_strategy
        self.protein_sims_type = protein_sims_type
        self.num_neg_samples_protein_protein_per_protein = (
            num_neg_samples_protein_protein_per_protein
        )

        self.is_protein_tokenized = is_protein_tokenized
        self.use_only_ppi_proteins = use_only_ppi_proteins

        self.protein_tokenizer = protein_tokenizer
        self.max_protein_len = max_protein_len
        self.use_protein_embeddings = use_protein_embeddings

        self._load_data()

    def _load_data(self):
        # protein and GO sequences/tokens/embeddings
        if not self.is_protein_tokenized:
            self.protein_sequences = [
                str(seq.seq)
                for seq in SeqIO.parse(
                    self.data_dir + "integrated_data/protein_sequences.fa", "fasta"
                )
            ]
            self.protein_tokens = None
        else:
            self.protein_tokens = pd.read_pickle(
                self.data_dir + "integrated_data/protein_tokens.pkl"
            )
            self.protein_sequences = None

        if not self.is_protein_tokenized:
            self.batch_converter = BatchConverter(
                self.protein_tokenizer, truncation_seq_length=self.max_protein_len
            )

        # in-batch protein negative sampling preparation
        protein_info = pd.read_pickle(
            self.data_dir + "integrated_data/v1/protein/protein_info_filtered.pkl"
        )

        # load ppi pretrain train relations
        protein_protein_relations_cl = pd.concat(
            [
                pd.read_csv(
                    self.data_dir
                    + "integrated_data/protein_protein/protein_protein_relations_CL_train_indexed.csv"
                ),
                pd.read_csv(
                    self.data_dir
                    + "integrated_data/protein_protein/protein_protein_relations_CL_val_indexed.csv"
                ),
            ]
        )[["src", "relation", "dst"]]
        assert is_undirected(
            torch.from_numpy(protein_protein_relations_cl[["src", "dst"]].values).T
        )

        # get all proteins available
        if self.use_only_ppi_proteins:
            self.all_proteins = np.unique(
                protein_protein_relations_cl[["src", "dst"]].values
            )
        else:
            self.all_proteins = protein_info["index"].values
        self.num_proteins = len(self.all_proteins)

        # load masks and probs
        self.protein_masks = np.load(
            self.data_dir
            + "generated_data/negative_sampling_masks/protein_protein-protein_masks.npy",
            mmap_mode="r",
        )
        assert self.protein_masks.shape[0] == len(protein_info)

        if self.protein_sims_type is not None:
            self.protein_sims = np.load(
                self.data_dir
                + f"generated_data/negative_sampling_probs/protein_sims_{self.protein_sims_type}.npy",
                mmap_mode="r",
            )
            assert self.protein_sims.shape[0] == len(protein_info)
        else:
            self.protein_sims = np.array([None] * len(protein_info))

        # load ground truth protein relations for each protein to query during negative sampling.  we've processed the data and generated a true mask matrix, but for the sake of consistency, still generating a dictionary here.
        self.true_proteins = dict()
        for (
            _,
            head_prot_idx,
            rel_idx,
            tail_prot_idx,
        ) in protein_protein_relations_cl.itertuples():
            self.true_proteins.setdefault((head_prot_idx, rel_idx), []).append(
                tail_prot_idx
            )  # NOTE: We've post-processed the relations are undirected, we only need to store one direction

    def __call__(self, batch_input: List[Tuple[int, int, int]]):
        head_positive_proteins = [triplet[0] for triplet in batch_input]
        positive_relations = [triplet[1] for triplet in batch_input]
        tail_positive_proteins = [triplet[2] for triplet in batch_input]

        batch_protein_mask = np.array(
            self.protein_masks[head_positive_proteins + tail_positive_proteins, :][
                :, self.all_proteins
            ]
        )
        batch_protein_mask = (
            batch_protein_mask.sum(axis=0) == batch_protein_mask.shape[0]
        ).astype(float)

        # note that the indices here are NOT the true protein idx
        head_negative_proteins_indices_raw, tail_negative_proteins_indices_raw = (
            self._negative_sampling(
                head_positive_proteins,
                positive_relations,
                tail_positive_proteins,
                self.num_neg_samples_protein_protein_per_protein,
                self.true_proteins,
                self.num_proteins,
                batch_protein_mask,
                self.protein_sims,
            )
        )

        head_negative_proteins_indices_raw, tail_negative_proteins_indices_raw = (
            self.all_proteins[head_negative_proteins_indices_raw].tolist(),
            self.all_proteins[tail_negative_proteins_indices_raw].tolist(),
        )

        # get unique positive and negative protein indices, and map relations to new ids, also fetch corresponding sequences/tokens/embeddings
        unique_protein_indices, new_id_protein_mapping = torch.unique(
            torch.LongTensor(
                head_positive_proteins
                + tail_positive_proteins
                + head_negative_proteins_indices_raw
                + tail_negative_proteins_indices_raw
            ),
            return_inverse=True,
        )
        (
            head_positive_proteins_new,
            tail_positive_proteins_new,
            head_negative_proteins_indices_new,
            tail_negative_proteins_indices_new,
        ) = torch.split(
            new_id_protein_mapping,
            [
                len(head_positive_proteins),
                len(tail_positive_proteins),
                len(head_negative_proteins_indices_raw),
                len(tail_negative_proteins_indices_raw),
            ],
        )
        if not self.use_protein_embeddings:
            unique_protein_toks = self._convert_batch(unique_protein_indices)
        else:
            unique_protein_toks = None

        positive_relations_new = torch.LongTensor(positive_relations)

        # collect all negative relations. first consider those that permutate GOs, then those that permutate proteins.
        head_negative_proteins_final = torch.cat(
            [
                torch.repeat_interleave(
                    head_positive_proteins_new,
                    len(tail_negative_proteins_indices_new)
                    // len(head_positive_proteins_new),
                ),
                head_negative_proteins_indices_new,
            ]
        )
        negative_relations_final = torch.cat(
            [
                torch.repeat_interleave(
                    positive_relations_new,
                    len(tail_negative_proteins_indices_new)
                    // len(positive_relations_new),
                ),
                torch.repeat_interleave(
                    positive_relations_new,
                    len(head_negative_proteins_indices_new)
                    // len(positive_relations_new),
                ),
            ]
        )
        tail_negative_proteins_final = torch.cat(
            [
                tail_negative_proteins_indices_new,
                torch.repeat_interleave(
                    tail_positive_proteins_new,
                    len(head_negative_proteins_indices_new)
                    // len(head_positive_proteins_new),
                ),
            ]
        )

        return {
            # entity tokens, 2d tensor of (num unique entities, max seq len in the batch)
            "node_toks": {"protein": unique_protein_toks, "go": None},
            # entity indices, 1d tensor of (num unique entities, )
            "node_indices": {
                "protein": unique_protein_indices,
                "go": None,
            },
            "protein_protein_relations": {
                # positive relations. ids have been remapped to the indices as in 'node_toks', except for relations, which indexes decoder. 1d tensor of shape B (batch_size)
                "positive_relations": {
                    "head": head_positive_proteins_new,
                    "relation": positive_relations_new,
                    "tail": tail_positive_proteins_new,
                },
                # negative relations. ids have been remapped to the indices as in 'node_toks', except for relations, which indexes decoder.  1d tensor of shape B x Neg x 2B (batch_size * num_neg_per_sample * (2 * batch_size), since negatives are shared)
                "negative_relations": {
                    "head": head_negative_proteins_final,
                    "relation": negative_relations_final,
                    "tail": tail_negative_proteins_final,
                },
            },
        }

    def _convert_batch(self, unique_protein_indices: List[int]) -> torch.LongTensor:
        batch_toks = convert_batch_protein(
            unique_protein_indices,
            self.is_protein_tokenized,
            self.batch_converter,
            self.protein_sequences,
            self.protein_tokens,
            self.protein_tokenizer,
            self.max_protein_len,
        )

        return batch_toks

    def _negative_sampling(
        self,
        head_positive_proteins: List[int],
        positive_relations: List[int],
        tail_positive_proteins: List[int],
        num_neg_samples_protein_protein_per_protein: int,
        true_proteins: dict,
        num_proteins: int,
        batch_protein_mask: np.ndarray,
        protein_sims: np.ndarray = None,
    ) -> Tuple[List[int], List[int]]:
        if self.negative_sampling_strategy == "protein_both":
            # NOTE: Here we are essentially allowing for sampling with replacement because negatives are shared. I don't think it matters honestly given our random edge sampling. Thoughts?
            head_negative_protein_indices = []  # permutating head for fixed tail
            tail_negative_protein_indices = []  # permutating tail for fixed head
            for head_prot, rel, tail_prot in zip(
                head_positive_proteins, positive_relations, tail_positive_proteins
            ):
                protein_mask_head = batch_protein_mask

                protein_sim_head = np.array(protein_sims[head_prot, self.all_proteins])
                protein_prob_head = process_protein_sims(
                    protein_sim_head, self.negative_sampling_strategy
                )

                head_negative_protein_indices.extend(
                    negative_sampling_random_tail(
                        (tail_prot, rel),
                        num_neg_samples_protein_protein_per_protein,
                        num_proteins,
                        protein_mask_head,
                        protein_prob_head,
                    )
                )
                protein_mask_tail = batch_protein_mask

                protein_sim_tail = np.array(protein_sims[tail_prot, self.all_proteins])

                protein_prob_tail = process_protein_sims(
                    protein_sim_tail, self.negative_sampling_strategy
                )
                tail_negative_protein_indices.extend(
                    negative_sampling_random_tail(
                        (head_prot, rel),
                        num_neg_samples_protein_protein_per_protein,
                        num_proteins,
                        protein_mask_tail,
                        protein_prob_tail,
                    )
                )

        else:
            raise NotImplementedError

        # NOTE: For each negative proteins, we also duplicate the negative for all proteins within the batch
        return (head_negative_protein_indices + tail_negative_protein_indices) * len(
            head_positive_proteins
        ), (tail_negative_protein_indices + head_negative_protein_indices) * len(
            tail_positive_proteins
        )
