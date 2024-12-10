import pickle, os, json, copy
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from Bio import SeqIO

from procyon.data.data_utils import (
    convert_batch_protein,
    get_text_sequences,
    get_text_sequences_compositions,
)

import torch
from esm.data import BatchConverter

from procyon.data.constants import *

from procyon.data.instruct_tune.instruct_constructor import (
    get_prompt,
    get_prompt_open_def,
    sample_demonstrations_for_prompts,
)


def count_words_with_punctuation(input_string):
    # Replace common punctuation marks with spaces to treat them as part of words
    punctuation_marks = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    for char in punctuation_marks:
        input_string = input_string.replace(char, " a ")

    # Split the string into words using whitespace as the delimiter
    words = input_string.split()

    # Return the number of words
    return len(words)


class BaseITCollator:

    def __init__(
        self,
        data_dir: str,
        aaseq_type: str,
        text_type: str,
        relation_type: str,
        text_variant_type: str,
        aaseq_sims_type: str,  # choose from ['esm2-650m_embeds_cosine', 'levenstein', None] # TODO: Not used - Do we need it?
        is_aaseq_tokenized: bool,
        is_text_tokenized: bool,
        use_text_embeddings: bool,
        use_aaseq_embeddings: bool,
        aaseq_tokenizer: object = None,
        text_tokenizer: object = None,
        max_aaseq_len: int = None,
        max_text_len: int = None,  # by default, no truncation
        num_examples: int = None,  # If None, uses default number of examples
        sample_num_examples: bool = False,
        use_entity_compositions: bool = False,
        sample_entity_compositions: str = "uniform",
        evaluation=False,
        use_instructions=True,
        use_drug_embeddings=False,
        column_subset=None,
        insert_disease_function_context=False,
        disease_function_context_dropout=None,
        text_split_method: str = "sample_aware_ontology_go_centric",  # Adopted from Dataset class
        insert_go_ontology_context: bool = False,
        go_ontology_rag_level_upper_limit: int = 5,
        go_ontology_rag_num_context: int = 3,
        go_ontology_rag_sample_num_context: bool = False,
        insert_go_ontology_level: bool = False,
        use_go_ontology_level_groups: bool = True,
        insert_reactome_ontology_context: bool = False,
        reactome_ontology_rag_level_upper_limit: int = 5,
        reactome_ontology_rag_num_context: int = 3,
        reactome_ontology_rag_sample_num_context: bool = False,
        insert_reactome_ontology_level: bool = False,
        use_reactome_ontology_level_groups: bool = True,
        use_drug_context_augmentation: bool = False,
        use_entity_rephrasings: bool = False,
        use_task_def_rephrasings: bool = False,
        rephrasing_sample_prob: float = 0.5,
        use_personality_prompts_rephrasing: bool = False,
        exclude_levels_in_ontology_captioning=False,
        fixed_rephrasing_expertise_level=None,
        fixed_rephrasing_entity_rephrase_level=None,
        fixed_rephrasing_task_def_rephrase_level=None,
        long_protein_strategy=None,
    ):

        if ":" in text_type:
            tt_split = text_type.split(":")
            self.text_type = tt_split[0]
            self.cols_to_sample = tt_split[1:]
        else:
            self.text_type = text_type
            self.cols_to_sample = None

        self.data_dir = data_dir
        self.aaseq_type = aaseq_type
        # NOTE: Usually this would be "all", but for DrugBank, this would be one of "drug_target", "drug_transporter", "drug_carrier", and "drug_enzyme"
        self.relation_type = relation_type
        self.text_variant_type = text_variant_type
        self.aaseq_sims_type = aaseq_sims_type
        self.is_ppi = aaseq_type == text_type  # Should trigger when protein==protein

        self.is_aaseq_tokenized = is_aaseq_tokenized
        self.is_text_tokenized = is_text_tokenized
        self.use_aaseq_embeddings = use_aaseq_embeddings
        self.use_text_embeddings = use_text_embeddings

        self.aaseq_tokenizer = aaseq_tokenizer
        self.text_tokenizer = text_tokenizer
        self.max_aaseq_len = max_aaseq_len
        self.max_text_len = max_text_len
        self.num_examples = num_examples
        self.sample_num_examples = sample_num_examples

        self.use_entity_compositions = use_entity_compositions
        self.sample_entity_compositions = sample_entity_compositions
        self.evaluation = evaluation
        self.use_instructions = use_instructions
        self.use_drug_embeddings = use_drug_embeddings

        self.column_subset = column_subset

        self.text_split_method = text_split_method

        self.insert_disease_function_context = insert_disease_function_context
        self.disease_function_context_dropout = (
            disease_function_context_dropout
            if disease_function_context_dropout is not None
            else 0.0
        )

        self.insert_ontology_context = None
        self.insert_ontology_level = None

        self.use_drug_context_augmentation = use_drug_context_augmentation

        self.use_entity_rephrasings = use_entity_rephrasings
        self.use_task_def_rephrasings = use_task_def_rephrasings
        self.rephrasing_sample_prob = rephrasing_sample_prob
        self.use_personality_prompts_rephrasing = use_personality_prompts_rephrasing

        self.exclude_levels_in_ontology_captioning = (
            exclude_levels_in_ontology_captioning
        )

        self.fixed_rephrasing_expertise_level = fixed_rephrasing_expertise_level
        self.fixed_rephrasing_entity_rephrase_level = (
            fixed_rephrasing_entity_rephrase_level
        )
        self.fixed_rephrasing_task_def_rephrase_level = (
            fixed_rephrasing_task_def_rephrase_level
        )

        self.long_protein_strategy = long_protein_strategy

        if self.text_type == "go":
            self.insert_ontology_context = insert_go_ontology_context
            self.ontology_rag_level_upper_limit = go_ontology_rag_level_upper_limit
            self.ontology_rag_num_context = go_ontology_rag_num_context
            self.ontology_rag_sample_num_context = go_ontology_rag_sample_num_context
            self.insert_ontology_level = insert_go_ontology_level
        elif self.text_type == "reactome":
            self.insert_ontology_context = insert_reactome_ontology_context
            self.ontology_rag_level_upper_limit = (
                reactome_ontology_rag_level_upper_limit
            )
            self.ontology_rag_num_context = reactome_ontology_rag_num_context
            self.ontology_rag_sample_num_context = (
                reactome_ontology_rag_sample_num_context
            )
            self.insert_ontology_level = insert_reactome_ontology_level

        assert not use_text_embeddings, "Cannot use text embeddings when using IT"
        assert not is_text_tokenized

        self._load_data()

    def _load_data(self):
        # protein and GO sequences/tokens/embeddings
        if not self.is_aaseq_tokenized:
            self.aaseq_sequences = [
                str(seq.seq)
                for seq in SeqIO.parse(
                    os.path.join(
                        self.data_dir,
                        "integrated_data",
                        "v1",
                        self.aaseq_type,
                        f"{self.aaseq_type}_sequences.fa",
                    ),
                    "fasta",
                )
            ]
            self.aaseq_tokens = None
        else:
            raise NotImplementedError

        if self.is_ppi:
            self.text_sequences = None
        else:
            if self.use_entity_compositions:
                text_df = pd.read_pickle(
                    os.path.join(
                        self.data_dir,
                        "integrated_data",
                        "v1",
                        self.text_type,
                        f"{self.text_type}_info_filtered_composed.pkl",
                    )
                )
                self.text_sequences = get_text_sequences_compositions(
                    text_type=self.text_type,
                    text_info=text_df,
                    column_subset=self.column_subset,  # Caption filtering done here, filters set upstream and accessed via constants in procyon.data.constants
                )
            else:
                text_df = pd.read_pickle(
                    os.path.join(
                        self.data_dir,
                        "integrated_data",
                        "v1",
                        self.text_type,
                        f"{self.text_type}_info_filtered.pkl",
                    )
                )
                self.text_sequences = get_text_sequences(
                    text_type=self.text_type,
                    text_info=text_df,
                    text_variant_type=self.text_variant_type,
                )

        if ("drugbank" in self.text_type) and self.use_drug_embeddings:
            self.drug_mask = torch.load(
                os.path.join(
                    self.data_dir,
                    f"integrated_data/v1/{self.text_type}/drugbank_mask.pt",
                )
            )
        else:
            self.drug_mask = None

        self.text_tokens = None

        if not self.is_aaseq_tokenized:
            if self.long_protein_strategy == "truncate":
                self.batch_converter = BatchConverter(
                    self.aaseq_tokenizer, truncation_seq_length=self.max_aaseq_len
                )
            else:
                self.batch_converter = BatchConverter(
                    self.aaseq_tokenizer,
                )

        # Set knowledge domain:
        self.knowledge_domain = None
        if self.text_type in ["go", "reactome"]:
            self.knowledge_domain = "ontology"

            # Get text field to include in context insertions:
            self.ontology_rag_text_field = ONTOLOGY_RAG_SUBSETS[self.text_type]

            all_relations = pd.read_csv(
                os.path.join(
                    self.data_dir,
                    "integrated_data",
                    "v1",
                    f"{self.aaseq_type}_{self.text_type}",
                    self.text_split_method,
                    f"{self.aaseq_type}_{self.text_type}_relations_indexed.unified.csv",
                )
            )

            # Get specific text:
            if self.text_type == "go":
                go_df = pd.read_pickle(
                    os.path.join(
                        self.data_dir, "integrated_data/v1/go/go_info_all_composed.pkl"
                    )
                )
                self.ontology_metadata = text_df[
                    [f"{self.text_type}_id", f"{self.text_type}_level"]
                ]
                # Filter based on included keys in the split:
                if not self.evaluation:
                    # Need to account for those that are filtered out in CL_train:
                    eval_gos = (
                        all_relations["text_id"]
                        .loc[(all_relations["split"] == "eval_zero_shot")]
                        .unique()
                    )  # i.e., not train
                    self.eval_texts_set = set(
                        go_df[f"{self.text_type}_id"].iloc[eval_gos.tolist()].tolist()
                    )
                    go_df = go_df.drop(eval_gos)

                self.rag_ontology_text = {
                    go_df["go_id"].iloc[i]: go_df[self.ontology_rag_text_field].iloc[i]
                    for i in range(go_df.shape[0])
                }
                self.ancestor_bounds = (1, self.ontology_rag_level_upper_limit)

            elif self.text_type == "reactome":
                self.ontology_metadata = pd.read_pickle(
                    os.path.join(
                        self.data_dir,
                        "integrated_data/v1/reactome/reactome_info_filtered_with_level.pkl",
                    )
                )[[f"{self.text_type}_id", f"{self.text_type}_level"]]
                if not self.evaluation:
                    # Need to account for those that are filtered out in CL_train:
                    eval_texts = all_relations["text_id"].loc[
                        (all_relations["split"] == "eval_zero_shot")
                    ]  # i.e., not train
                    self.eval_texts_set = set(
                        self.ontology_metadata[f"{self.text_type}_id"]
                        .iloc[eval_texts.tolist()]
                        .tolist()
                    )
                    text_df = text_df.drop(eval_texts)

                self.rag_ontology_text = {
                    text_df["reactome_id"]
                    .iloc[i]: text_df[self.ontology_rag_text_field]
                    .iloc[i]
                    for i in range(text_df.shape[0])
                }
                self.ancestor_bounds = (1, self.ontology_rag_level_upper_limit)

            # Load reactome and go-specific ancestors:
            self.ancestor_ontology = torch.load(
                os.path.join(
                    self.data_dir,
                    f"integrated_data/v1/{self.text_type}/{self.text_type}_ancestors_dist_bylevel.pkl",
                )
            )

            if not self.evaluation:
                # Need to filter ancestor ontology:
                new_ancestor_ontology = {}
                for k in self.ancestor_ontology.keys():
                    new_ancestor_ontology[k] = {}
                    for i in self.ancestor_ontology[k].keys():
                        filtered_list = [
                            a
                            for a in self.ancestor_ontology[k][i]
                            if not (a in self.eval_texts_set)
                        ]
                        if len(filtered_list) > 0:
                            new_ancestor_ontology[k][i] = filtered_list

                self.ancestor_ontology = new_ancestor_ontology

        elif self.text_type in ["pfam", "uniprot", "ec", "gtop"]:
            self.knowledge_domain = "function"
        elif self.text_type in ["omim", "disgenet"]:
            self.knowledge_domain = "disease"
        elif self.text_type in ["drugbank"]:
            self.knowledge_domain = "drug"

        if self.knowledge_domain in ["disease", "drug"]:
            # Load uniprot
            self.functional_descriptions = pd.read_pickle(
                os.path.join(
                    self.data_dir,
                    "integrated_data/v1/protein/uniprot_functional_descriptions.pkl",
                )
            ).sort_values("index", axis=0)[
                "function"
            ]  # Make sure it's sorted by protein index

        if self.knowledge_domain == "drug":
            self.context_col = None
            if self.cols_to_sample is not None:
                if len(self.cols_to_sample) == 1:
                    if self.cols_to_sample[0] == "indication":
                        self.context_col = "moa"
                    elif self.cols_to_sample[0] == "moa":
                        self.context_col = "indication"
                    else:
                        raise NotImplementedError(
                            "sampling column {} not recognized".format(
                                self.cols_to_sample[0]
                            )
                        )
                else:
                    raise NotImplementedError

        # First, task definition rephrasings:
        if self.use_task_def_rephrasings:
            df_td = pd.read_csv(
                os.path.join(self.data_dir, "generated_data/task_def_rephrasings.csv")
            )

            # Get my subset:
            text_alias = self.text_type
            if self.aaseq_type == "domain":
                text_alias = "{}_{}".format(
                    self.aaseq_type, self.text_type
                )  # Must specify domain based on nomenclature in rephrasings df
            elif "drugbank" in self.text_type:
                text_alias = "drugbank_drug"

            present_mask = (
                (df_td["Dataset"] == text_alias)
                & (df_td["Relation"] == self.relation_type)
                & (df_td["Task"] == self.TASK)
            )
            if present_mask.sum() == 0:
                self.task_def_rephrase = None
            else:
                df_paraphrase = df_td["Paraphrase"].loc[present_mask].item()
                self.task_def_rephrase = json.loads(
                    df_paraphrase.strip("```").strip("json")
                )

        # Get entity rephrasings if needed:
        self.rephrased_entities = None
        if self.use_entity_rephrasings:
            if self.cols_to_sample is not None:
                cols_to_get = self.cols_to_sample
            elif self.column_subset[self.text_type] is not None:
                cols_to_get = self.column_subset[self.text_type]
            else:
                # Get default columns
                cols_to_get = list(ENTITY_DESCRIPTION_NAMES[self.text_type].keys())

            self.rephrased_columns_mapping = ENTITY_REPHRASING_COLUMN_NAMES[
                self.text_type
            ]

            rephrase_files = ENTITY_REPHRASING_FILES[self.text_type]
            if rephrase_files is not None:  # Filter by column + load files
                rephrase_dfs = {}

                for c in cols_to_get:
                    df_c = pd.read_pickle(
                        os.path.join(
                            self.data_dir,
                            f"integrated_data/v1/{self.text_type}",
                            rephrase_files[c],
                        )
                    )
                    my_rephrase_subset = []
                    for e in EXPERTISE_LEVEL:
                        for r in REPHRASE_ENTITY_LEVEL:
                            my_rephrase_subset.append(
                                f"{self.rephrased_columns_mapping[c]}_{e}_{r}"
                            )
                    # Filter by column:
                    rephrase_dfs[c] = df_c[my_rephrase_subset]

                self.rephrased_entities = rephrase_dfs
            else:
                self.rephrased_entities = (
                    None  # We don't have rephrasings for this dataset
                )

    def _convert_batch(self, entity_type: str, unique_indices: List[int]):
        assert entity_type == "sequence"
        batch_toks = convert_batch_protein(
            unique_indices,
            self.is_aaseq_tokenized,
            self.batch_converter,
            self.aaseq_sequences,
            self.aaseq_tokens,
            self.aaseq_tokenizer,
            self.max_aaseq_len,
        )
        return batch_toks

    def _sample_batch_entity_descriptions(self, text_ids, rephrasing_guide=None):
        # Get rows for the given text_ids
        rows_df = self.text_sequences.iloc[text_ids, :]

        if self.cols_to_sample is not None:
            if len(self.cols_to_sample) == 1:
                return rows_df[self.cols_to_sample[0]].tolist()
            else:
                raise NotImplementedError

        elif self.sample_entity_compositions == "uniform":

            # Below assisted by ChatGPT:
            # Initialize an empty list to store the sampled values
            sampled_values = []

            # Iterate over each row - Not ideal to use for loop, but not sure if there's a more efficient way??
            for _, row in rows_df.iterrows():
                # Filter out NaN values
                non_nan_values = row.dropna()
                non_nan_values = non_nan_values[
                    non_nan_values.apply(lambda x: isinstance(x, str))
                ]  # Make sure it's a string

                # Randomly sample from the non-NaN values if there are any, else assign NaN
                if not non_nan_values.empty:
                    sampled_values.append(np.random.choice(non_nan_values))
                else:
                    raise ValueError()

            return sampled_values
        else:
            raise NotImplementedError(
                "{} sampling strategy not implemented for compositions".format(
                    self.sample_entity_compositions
                )
            )

    def _get_input_contexts(
        self, aaseq_ids, text_ids, aaseq_list=None, text_list=None, get_drugs=False
    ):
        # Inputs are [[], [], ...], [[], [], ...] - lists of lists
        # Also set arg to make this optional:
        if self.knowledge_domain == "function":
            return None
        elif self.knowledge_domain == "drug" and not self.use_drug_context_augmentation:
            return None
        elif self.knowledge_domain == "ontology":  # Based on text_ids
            # We know it's reactome or GO
            if not self.insert_ontology_context:
                return None

            contexts = []
            for t in text_ids:
                if isinstance(t, torch.Tensor):
                    t = t.item()

                # Step 1: For each text id, get it's GO/Reactome ID
                text_id_db = self.ontology_metadata[f"{self.text_type}_id"].iloc[
                    t
                ]  # Based on database-specific ID

                # Get ancestors for each sample
                ancestors = self.ancestor_ontology[text_id_db]

                t_level = self.ontology_metadata[f"{self.text_type}_level"].iloc[t]

                # Sample number of ancestors
                # Two steps to keep class-balanced - sample number from uniform and
                if self.ontology_rag_sample_num_context:
                    N_context = np.random.randint(
                        low=0, high=self.ontology_rag_num_context
                    )  # Don't retrieve below that level
                else:
                    N_context = self.ontology_rag_num_context

                upper_bound = min(self.ancestor_bounds[1], t_level)

                # Level-balanced sampling:
                # Get ancestors' level classifications
                levels_to_sample = [
                    k
                    for k in ancestors.keys()
                    if ((k < upper_bound) and (k >= self.ancestor_bounds[0]))
                ]
                sample_size = min(len(levels_to_sample), N_context)
                if sample_size == 0:
                    contexts.append("")  # It's blank
                    continue

                sample_levels = np.random.choice(
                    levels_to_sample, size=sample_size, replace=False
                )

                # Sample actual ancestors + Get ancestors' text
                sample_ancestor_texts = []
                for l in sample_levels:
                    options = ancestors[l]
                    choice = np.random.choice(options)
                    sample_ancestor_texts.append(self.rag_ontology_text[choice])

                # Create contexts for each ancestor
                context_string_i = "\nContext:\n"
                for li, ti in zip(sample_levels, sample_ancestor_texts):
                    if self.insert_ontology_level:
                        lname = self._ontology_level_transform(li)
                        context_string_i += "Level: {}\n{}\n".format(lname, ti)
                    else:
                        context_string_i += "{}\n".format(ti)

                context_string_i += "End Context\n"

                contexts.append(context_string_i)
            return contexts
        elif self.knowledge_domain == "disease" or (
            (self.knowledge_domain == "drug") and (self.TASK == "qa")
        ):
            if not self.insert_disease_function_context:
                return None
            assert (
                self.aaseq_type == "protein"
            ), "Not yet implemented for domain-disease relations"

            f_i = self.functional_descriptions.iloc[aaseq_ids].tolist()

            functions = [
                (("Context: " + f + "\n") if isinstance(f, str) else "") for f in f_i
            ]  # Append with "Context"

            return functions
        elif (self.knowledge_domain == "drug") and (self.TASK == "caption"):
            if self.context_col is None:
                return None
            f_i = self.functional_descriptions.iloc[aaseq_ids].tolist()
            functions = [(f if isinstance(f, str) else "") for f in f_i]

            opp_col = (
                self.text_sequences[self.context_col].iloc[text_ids].tolist()
            )  # We know none of them are missing

            # Compose information:
            drug_context = None
            if self.context_col == "moa":
                drug_context = "Mechanism of Action: "
            elif self.context_col == "indication":
                drug_context = "Drug Indication: "

            # Align text and functions:
            assert len(aaseq_list) == len(text_list)
            all_contexts = []
            all_contexts_flat = []
            all_drug_struct = []
            all_drug_indices = []
            at_least_one_drug_flag = False
            drug_counter = 0
            for ai, ti in zip(aaseq_list, text_list):
                expand_functions = [functions[i] for i in ai]
                expand_drug_info = [opp_col[i] for i in ti]

                # Compose entry:
                context_strs = []
                drug_sub_inds = []
                for i in range(len(ai)):
                    c = "Context: " + expand_functions[i] + "\n"
                    drug_context_i = drug_context
                    c += drug_context_i + expand_drug_info[i] + "\n"
                    if (self.drug_mask is not None) and get_drugs:
                        if self.drug_mask[text_ids[ti[i]]]:
                            c += "Drug: <|drug|>"
                            all_drug_struct.append(text_ids[ti[i]])
                            drug_sub_inds.append(drug_counter)
                            drug_counter += 1
                            at_least_one_drug_flag = True
                    context_strs.append(c)
                    all_contexts_flat.append(c)
                all_drug_indices.append(drug_sub_inds)

                all_contexts.append(context_strs)

            if get_drugs:
                if not at_least_one_drug_flag:
                    all_drug_struct = None
                    all_drug_indices = None

                return (
                    all_contexts,
                    all_contexts_flat,
                    all_drug_struct,
                    all_drug_indices,
                )

            return all_contexts, all_contexts_flat

    def _ontology_level_transform(self, level):
        if self.text_type == "reactome":
            boundaries, level_names = ONTOLOGY_RAG_LEVEL_GROUPS["reactome"]
            if level < boundaries[0]:
                lname = level_names[0]
            elif level < boundaries[1]:
                lname = level_names[1]
            else:
                lname = level_names[2]
        elif self.text_type == "go":
            boundaries, level_names = ONTOLOGY_RAG_LEVEL_GROUPS["go"]
            if level < boundaries[0]:
                lname = level_names[0]
            elif level < boundaries[1]:
                lname = level_names[1]
            else:
                lname = level_names[2]
        else:
            raise NotImplementedError

        return lname

    def _sample_batch_entities_with_rephrasings(
        self,
        descriptions: list,
        unique_text_ids: list,
        text_ids_expand: list,
        rephrasing_guide: list,
    ):
        """
        Performs sampling of entities while considering rephrasings
            - Because sampling is performed on a by-sample basis, means we may need to add descriptions to the
                unique_text_ids list
        Inputs:
            text_ids: non-unique (i.e. expanded by batch) text_ids
            rephrasing_guide: list of strings
                - For each sample in batch, must be equal in length to batch length
        """
        uti_list_onthefly = []
        tid_exp_frozen = copy.deepcopy(text_ids_expand)
        for i, trow in enumerate(tid_exp_frozen):
            # Use expertise level given by rephrasing guide
            rguide_i = rephrasing_guide[i]
            if rguide_i is None:
                # Do nothing, but add original descriptions to the list
                uti_list_onthefly.append([unique_text_ids[k] for k in trow])
                continue

            new_descs = []
            new_text_ids = []
            starting_i = len(descriptions)  # Constantly updates as we add to it

            # Below re-samples the field in which we're choosing the column
            if (
                self.cols_to_sample is not None
            ):  # Only set if we only use one column (improperly named, but not gonna mess with it now :) )
                # Don't need to control for NaNs
                c_choice = self.cols_to_sample[0]
            elif self.column_subset is not None:
                # Need to control for NaN's
                # This is a bit messy...
                expert_rephrase_level = "{}_{}".format(*rguide_i)
                # Idea: choose candidate columns (from column_subset for the text_type) that don't have missing values in the rephrased entities for the given expert rephrase level
                c_subset = [
                    c
                    for c in self.column_subset[self.text_type]
                    if not (
                        pd.isnull(
                            self.rephrased_entities[c][
                                f"{self.rephrased_columns_mapping[c]}_{expert_rephrase_level}"
                            ].iloc[unique_text_ids[trow].tolist()]
                        )
                    ).any()
                ]
                if len(c_subset) == 0:
                    # Then we have no rephrasings available for this sample, move on and don't modify description
                    uti_list_onthefly.append([unique_text_ids[k] for k in trow])
                    continue
                c_choice = np.random.choice(
                    c_subset
                )  # Randomly choose from those that are non-nan
            else:  # If there is no subset defined - should be defined (i.e. using above statement) for most instances after 05/07/2024
                c_choice = np.random.choice(list(self.rephrased_columns_mapping.keys()))

            # Name of column based on the column choice
            # Must convert via rephrased_columns_mapping because rephrased columns may have different name than column in original dataset file
            rephrased_column_name = self.rephrased_columns_mapping[c_choice]

            df_c = self.rephrased_entities[c_choice]  # based on chosen column

            uti_local = []
            for relative_text_id, t in enumerate(
                trow
            ):  # relative_text_id defined as relative to descriptions we're adding
                # relative_text_id is on enumerate - tells you the index within this sub-list of the text_idx row
                # t tells you the original description index for the given
                expert_rephrase_level = f"{rguide_i[0]}_{rguide_i[1]}"  # <expertise_level>_<entity_rephrasing_level>
                d = df_c[f"{rephrased_column_name}_{expert_rephrase_level}"].iloc[
                    unique_text_ids[t].item()
                ]  # Need to

                if pd.isnull(d):
                    d = descriptions[
                        t
                    ]  # If the rephrasing doesn't exist for this in-context information, replace it with the original text
                    # t is relative to the original descriptions list
                    # Ends up repeating the description below, but this is fine since it will get filtered

                new_descs.append(d)
                new_text_ids.append(starting_i + relative_text_id)
                uti_local.append(unique_text_ids[t].item())

            uti_list_onthefly.append(
                uti_local
            )  # uti_list_onthefly is nested list, need to append local counter from sub-list
            descriptions += new_descs
            # uti_unique_list += []
            # NOTE: Below line has to come before transformation or you'll get index errors
            assert len(new_text_ids) == len(text_ids_expand[i])
            text_ids_expand[i] = new_text_ids  # REPLACE THE IDS

        # Now take out any descriptions that don't appear in the new descriptions

        unique_ids_post = set(np.unique(np.array(text_ids_expand).flatten()))
        # This is based on 0 indexing, but may not have consecutive indices bc some get taken out

        remove_ids = [
            i for i in range(max(unique_ids_post)) if (i not in unique_ids_post)
        ]

        text_ids_expand_np = np.array(text_ids_expand)
        text_ids_expand_np_frozen = text_ids_expand_np.copy()

        for ri in sorted(
            remove_ids, reverse=True
        ):  # MUST REVERSE SORT TO AVOID CONFLICTING POP'S
            text_ids_expand_np[
                text_ids_expand_np_frozen >= ri
            ] -= 1  # Gets all above the remove index, adjusts indices to account for popping the description
            descriptions.pop(ri)  # Removes and updates the list

        # Update unique_text_indices:
        new_uti = []
        uti_matching_to_descriptions_np = np.array(uti_list_onthefly)

        for i in list(np.sort(np.unique(text_ids_expand_np))):
            match_i = np.unique(
                uti_matching_to_descriptions_np[text_ids_expand_np == i]
            )
            new_uti.append(match_i[0])

        # Checks if all indices are consecutive
        assert (
            np.sort(np.unique(text_ids_expand_np.flatten()))
            - np.arange(text_ids_expand_np.max() + 1)
        ).sum() < 1e-8

        assert not np.any([pd.isnull(d) for d in descriptions])

        return descriptions, torch.LongTensor(new_uti), text_ids_expand_np.tolist()

    def _sample_rephrasing_levels(self, batch_length):
        if not self.use_entity_rephrasings or (batch_length == 0):
            return None

        rephrase_guide = []
        for _ in range(batch_length):
            # First, decide if you need to rephrase
            if np.random.uniform() < self.rephrasing_sample_prob:

                if self.fixed_rephrasing_expertise_level is None:
                    expert_level = np.random.choice(
                        EXPERTISE_LEVEL
                    )  # Uniform - can change later
                else:
                    expert_level = self.fixed_rephrasing_expertise_level

                if self.fixed_rephrasing_entity_rephrase_level is None:
                    # Sample:
                    r_entity_level = np.random.choice(
                        REPHRASE_ENTITY_LEVEL
                    )  # Uniform - can change later
                else:
                    r_entity_level = self.fixed_rephrasing_entity_rephrase_level

                rephrase_guide.append((expert_level, r_entity_level))
            else:
                # Equivalent to no rephrasing
                rephrase_guide.append(None)
        return rephrase_guide

    def _sample_task_def_rephrasings(self, batch_length, true_desc):
        if (
            not self.use_task_def_rephrasings
            or (batch_length == 0)
            or (self.task_def_rephrase is None)
        ):
            return [true_desc for _ in range(batch_length)]

        rephrased_defs = []
        for j in range(batch_length):
            # First, decide if you need to rephrase
            if np.random.uniform() < self.rephrasing_sample_prob:
                i = np.random.choice(np.arange(1, 6))

                if self.fixed_rephrasing_task_def_rephrase_level is None:
                    rlevel = np.random.choice(REPHRASE_TASK_DEF_LEVEL)
                else:
                    rlevel = self.fixed_rephrasing_task_def_rephrase_level

                def_r = self.task_def_rephrase[f"response{i}"][rlevel]
                rephrased_defs.append(def_r)
            else:
                # Equivalent to no rephrasing
                rephrased_defs.append(true_desc)
        return rephrased_defs


def construct_task_id(aaseq_type, text_type, relation_type, task_type):
    if aaseq_type.lower() == "domain":
        # Must insert domain into name
        task_id = f"domain_{text_type}_{relation_type}_{task_type}"
    elif (aaseq_type.lower() == "protein") or (aaseq_type.lower() == "peptide"):
        task_id = f"{text_type}_{relation_type}_{task_type}"
    else:
        raise NotImplementedError(
            "No dataset found for aaseq_type = {}".format(aaseq_type)
        )

    return task_id


drugbank_field_specific_instructions = {
    "moa": {
        "Definition": "You will be shown a protein and its functions, along with a drug and its mechanism of action. Your job is to describe {Biological Summary}, of which this protein should be {Relationship Summary}. {Task-Specific Relationship}.",
        "Biological Summary": "the indication of the drug",
    },
    "indication": {
        "Definition": "You will be shown a protein and its functions, along with a drug and its indication. Your job is to describe {Biological Summary}, of which this protein should be {Relationship Summary}. {Task-Specific Relationship}.",
        "Biological Summary": "the mechanism of action of the drug",
    },
}


def replace_drugbank_instructions(task, drugbank_col):
    # By-reference call - updates automatically
    task.update(drugbank_field_specific_instructions[drugbank_col])


def idxMapperNestedArrays(l, idx_map):
    """
    Recursive mapping of indices in nested lists

    Helper function to map the within batch indices back to the full dataset indices after
    collating of in-context examples. Will likely be helpful when existing data loading
    infrastructure is used for baseline models which don't use the instruction format.
    """
    if isinstance(l, (list, np.ndarray)):
        return [idxMapperNestedArrays(x, idx_map) for x in l]
    elif isinstance(l, (int, np.integer)):
        return idx_map[l]
    elif l is None:
        return None
    elif isinstance(l, torch.Tensor):
        if l.dim() > 0:
            return [idxMapperNestedArrays(x, idx_map) for x in l]
        elif torch.is_floating_point(l):
            raise ValueError(f"float tensor: {l}")
        else:
            return idx_map[l.item()]
    else:
        raise ValueError(f"unexpected type: {type(l)}")


class QACollator(BaseITCollator):
    TASK = "qa"

    def __init__(self, *args, **kwargs):  # Init code contained in BaseIT collator
        super(QACollator, self).__init__(*args, **kwargs)
        task_id = construct_task_id(
            aaseq_type=self.aaseq_type,
            text_type=self.text_type,
            relation_type=self.relation_type,
            task_type="qa",
        )

        json_file_path = (
            os.path.dirname(os.path.abspath(__file__))
            + f"/instruct_tune/tasks/{task_id}.json"
        )
        if self.use_instructions:
            # Update task:
            with open(json_file_path, "r") as j:
                task = json.loads(j.read())
            if self.use_task_def_rephrasings:
                (
                    self.template,
                    self.description_true,
                    self.positive_examples_strs,
                    self.negative_example_strs,
                    self.example_text_ids,
                    self.example_aaseq_ids,
                ) = get_prompt_open_def(
                    task,
                    num_examples=self.num_examples,
                    sample_examples=self.sample_num_examples,
                    is_ppi=self.is_ppi,
                    aaseq_type=self.aaseq_type,
                )
            else:
                (
                    self.template,
                    self.positive_examples_strs,
                    self.negative_example_strs,
                    self.example_text_ids,
                    self.example_aaseq_ids,
                ) = get_prompt(
                    task,
                    num_examples=self.num_examples,
                    sample_examples=self.sample_num_examples,
                    is_ppi=self.is_ppi,
                    aaseq_type=self.aaseq_type,
                )
                self.description_true = None
        else:
            self.template = "[EXT] <|protein|> [ANSWER] {answer}"

    def __call__(
        self,
        batch_input: List[Tuple[Tuple[int], List[int]]],
    ) -> Dict[str, torch.Tensor]:
        """
        NOTE: As of 06/17, trying to make this general, but only testing on Protein-GO
        NOTE: Use negative size of 1 with this - it isn't contrastive learning so don't need large number of negatives

        Batch input structure (for GO):
            (prot_idx, rel_idx, text_idx), negative_protein_indices, negative_text_indices

        For now (06/17), only have uniform "description" as the input
        """

        # Conduct sampling of descriptions:
        if not self.use_instructions:  # Use null template
            raise NotImplementedError("Must use instructions for QA")
        elif self.sample_num_examples:
            template, example_text_ids, example_aaseq_ids = (
                sample_demonstrations_for_prompts(
                    template=self.template,
                    positive_examples=self.positive_examples_strs,
                    negative_examples=self.negative_example_strs,
                    example_text_ids=self.example_text_ids,
                    example_aaseq_ids=self.example_aaseq_ids,
                    is_ppi=self.is_ppi,
                )
            )
        else:
            template = self.template
            example_text_ids = self.example_text_ids
            example_aaseq_ids = self.example_aaseq_ids

        num_context_descriptions = 0
        if self.is_ppi:
            positive_aaseqs = [sample[0][0] for sample in batch_input] + [
                sample[0][2] for sample in batch_input
            ]
            negative_aaseqs = sum([sample[1] for sample in batch_input], start=[])
            num_pos = len(batch_input)
            negs_per_sample = len(batch_input[0][1])
            num_neg = len(batch_input) * negs_per_sample

            # These lines do a few things:
            #   1. Unifies aaseq indices in unique_aaseq_indices
            #   2. Creates mapping of positive and negative aaseqs
            unique_aaseq_indices, new_id_aaseq_mapping = torch.unique(
                torch.LongTensor(positive_aaseqs + negative_aaseqs + example_aaseq_ids),
                return_inverse=True,
            )
            (
                positive_aaseqs_reidx_lhs,
                positive_aaseqs_reidx_rhs,
                negative_aaseqs_reidx,
                example_aaseq_ids_reidx,
            ) = torch.split(
                new_id_aaseq_mapping,
                [
                    len(batch_input),
                    len(batch_input),
                    len(negative_aaseqs),
                    len(example_aaseq_ids),
                ],
            )
            if not self.use_aaseq_embeddings:
                unique_aaseq_toks = self._convert_batch(
                    "sequence", unique_aaseq_indices.tolist()
                )
            else:
                unique_aaseq_toks = None

            all_pairs = torch.cat(
                (
                    torch.cat(
                        (
                            positive_aaseqs_reidx_lhs.unsqueeze(1),
                            positive_aaseqs_reidx_rhs.unsqueeze(1),
                        ),
                        dim=1,
                    ),
                    torch.cat(
                        (
                            positive_aaseqs_reidx_lhs.repeat_interleave(
                                negs_per_sample
                            ).unsqueeze(1),
                            negative_aaseqs_reidx.unsqueeze(1),
                        ),
                        dim=1,
                    ),
                )
            )

            if self.use_task_def_rephrasings:
                # Rephrase def's in each template here:
                if self.evaluation:
                    instructions_pos = [
                        template.format(answer="") for _ in range(num_pos)
                    ]
                    instructions_neg = [
                        template.format(answer="") for _ in range(num_neg)
                    ]
                    labels = (["yes"] * num_pos) + (["no"] * num_neg)
                else:
                    rephrased_defs_pos = self._sample_task_def_rephrasings(
                        batch_length=num_pos, true_desc=self.description_true
                    )
                    instructions_pos = [
                        template.format(definition=rephrased_defs_pos[i], answer="yes")
                        for i in range(num_pos)
                    ]
                    rephrased_defs_neg = self._sample_task_def_rephrasings(
                        batch_length=num_neg, true_desc=self.description_true
                    )
                    instructions_neg = [
                        template.format(definition=rephrased_defs_neg[i], answer="no")
                        for i in range(num_neg)
                    ]
                    labels = None
            else:
                if self.evaluation:
                    instructions_pos = [
                        template.format(answer="") for _ in range(num_pos)
                    ]
                    instructions_neg = [
                        template.format(answer="") for _ in range(num_neg)
                    ]
                    labels = (["yes"] * num_pos) + (["no"] * num_neg)
                else:
                    instructions_pos = [
                        template.format(answer="yes") for _ in range(num_pos)
                    ]
                    instructions_neg = [
                        template.format(answer="no") for _ in range(num_neg)
                    ]
                    labels = None

            instruct_list = instructions_pos + instructions_neg

            # Construct aaseq indices list:
            aaseq_list = [
                example_aaseq_ids_reidx.tolist() + lst for lst in all_pairs.tolist()
            ]

            # No texts to add in to instructions, so add empty lists for consistency for downstream
            # code. (i.e. enforce len(instructions) = len(aaseq_list) == len(text_list))
            text_list = [[] for i in range(len(instruct_list))]
            descriptions = []

            # Set drugs to none as it doesn't apply for ppi
            unique_drug_indices = None
            unique_text_indices = []
            input_drug = None
        else:
            positive_aaseqs = [sample[0][0] for sample in batch_input]
            positive_texts = [sample[0][2] for sample in batch_input]
            negative_aaseqs = sum(
                [sample[1] for sample in batch_input if sample[1] is not None], start=[]
            )
            negative_texts = sum(
                [sample[2] for sample in batch_input if sample[2] is not None], start=[]
            )

            # These lines do a few things:
            #   1. Unifies aaseq indices in unique_aaseq_indices
            #   2. Creates mapping of positive and negative aaseqs
            unique_aaseq_indices, new_id_aaseq_mapping = torch.unique(
                torch.LongTensor(positive_aaseqs + negative_aaseqs + example_aaseq_ids),
                return_inverse=True,
            )
            positive_aaseqs_new, negative_aaseqs_new, example_aaseq_ids_new = (
                torch.split(
                    new_id_aaseq_mapping,
                    [
                        len(positive_aaseqs),
                        len(negative_aaseqs),
                        len(example_aaseq_ids),
                    ],
                )
            )
            if not self.use_aaseq_embeddings:
                unique_aaseq_toks = self._convert_batch(
                    "sequence", unique_aaseq_indices.tolist()
                )
            else:
                unique_aaseq_toks = None

            # Do the same for text:
            unique_text_indices, new_id_text_mapping = torch.unique(
                torch.LongTensor(positive_texts + negative_texts + example_text_ids),
                return_inverse=True,
            )
            positive_texts_new, negative_texts_new, example_text_ids_new = torch.split(
                new_id_text_mapping,
                [len(positive_texts), len(negative_texts), len(example_text_ids)],
            )

            # collect all negative relations. first consider those that permutate GOs, then those that permutate aaseqs.
            negative_aaseqs_final = torch.cat(
                [
                    torch.repeat_interleave(
                        positive_aaseqs_new,
                        len(negative_texts_new) // len(positive_texts_new),
                    ),
                    negative_aaseqs_new,
                ]
            )
            negative_texts_final = torch.cat(
                [
                    negative_texts_new,
                    torch.repeat_interleave(
                        positive_texts_new,
                        len(negative_aaseqs_new) // len(positive_aaseqs_new),
                    ),
                ]
            )
            # Both the above are reindexed such that:
            #   (positive_aaseqs_new, negative_aaseqs_new) are rebased positive indices
            #   (negative_aaseqs_final, negative_texts_final) are rebased negative indices

            if self.use_entity_compositions:
                descriptions = self._sample_batch_entity_descriptions(
                    unique_text_indices.tolist()
                )  # Return is list
            else:
                descriptions = self.text_sequences[
                    unique_text_indices.tolist()
                ].tolist()

            # Construct instruction within collator (change later):
            # No sampling, just static template usage for now
            if self.use_task_def_rephrasings:
                if self.evaluation:
                    instructions_pos = [
                        template.format(answer="")
                        for _ in range(len(positive_texts_new))
                    ]
                    instructions_neg = [
                        template.format(answer="")
                        for _ in range(len(negative_texts_final))
                    ]
                    labels = (["yes"] * len(positive_texts_new)) + (
                        ["no"] * len(negative_texts_final)
                    )
                else:
                    rephrased_defs_pos = self._sample_task_def_rephrasings(
                        batch_length=len(positive_texts_new),
                        true_desc=self.description_true,
                    )
                    instructions_pos = [
                        template.format(definition=rephrased_defs_pos[i], answer="yes")
                        for i in range(len(positive_texts_new))
                    ]
                    rephrased_defs_neg = self._sample_task_def_rephrasings(
                        batch_length=len(negative_texts_final),
                        true_desc=self.description_true,
                    )
                    instructions_neg = [
                        template.format(definition=rephrased_defs_neg[i], answer="no")
                        for i in range(len(negative_texts_final))
                    ]
                    labels = None
            else:
                if self.evaluation:
                    instructions_pos = [
                        template.format(answer="")
                        for _ in range(len(positive_texts_new))
                    ]
                    instructions_neg = [
                        template.format(answer="")
                        for _ in range(len(negative_texts_final))
                    ]
                    labels = (["yes"] * len(positive_texts_new)) + (
                        ["no"] * len(negative_texts_final)
                    )
                else:
                    instructions_pos = [
                        template.format(answer="yes")
                        for _ in range(len(positive_texts_new))
                    ]
                    instructions_neg = [
                        template.format(answer="no")
                        for _ in range(len(negative_texts_final))
                    ]
                    labels = None

            instruct_list = instructions_pos + instructions_neg

            # Construct aaseq indices list:
            total_indices_aaseq = torch.cat(
                [positive_aaseqs_new, negative_aaseqs_final]
            )
            aaseq_list = [
                example_aaseq_ids_new.tolist() + lst
                for lst in total_indices_aaseq.unsqueeze(1).tolist()
            ]

            total_indices_text = torch.cat([positive_texts_new, negative_texts_final])
            text_list = [
                example_text_ids_new.tolist() + lst
                for lst in total_indices_text.unsqueeze(1).tolist()
            ]

            # Rephrasing:
            # Transform because of rephrasings:
            if self.use_entity_rephrasings and (self.rephrased_entities is not None):
                # This modifies a lot of variables
                rephrasing_guide = self._sample_rephrasing_levels(
                    batch_length=len(instruct_list)
                )
                descriptions, unique_text_indices, text_list = (
                    self._sample_batch_entities_with_rephrasings(
                        descriptions=descriptions,
                        unique_text_ids=unique_text_indices,
                        text_ids_expand=text_list,
                        rephrasing_guide=rephrasing_guide,
                    )
                )

                # Augment instructions with personality prompts
                if self.use_personality_prompts_rephrasing:
                    for ind in range(len(instruct_list)):
                        if rephrasing_guide[ind] is not None:
                            pp = PERSONALITY_PROMPTS[
                                rephrasing_guide[ind][0]
                            ]  # First index is personality/level
                            instruct_list[ind] = pp + "\n" + instruct_list[ind]

            if self.insert_ontology_level and (self.knowledge_domain == "ontology"):
                for i in range(len(descriptions)):
                    level = self.ontology_metadata[f"{self.text_type}_level"].iloc[
                        unique_text_indices[i].item()
                    ]
                    lname = self._ontology_level_transform(level)
                    descriptions[i] = "Level: {}\n".format(lname) + descriptions[i]

            # Process drugs if needed:
            if (
                self.drug_mask is not None
            ):  # If this passes, we know this is a drug-based dataset
                # Base everything off of text indices
                drug_sample_mask = self.drug_mask[unique_text_indices]
                unique_drug_indices = unique_text_indices[
                    drug_sample_mask
                ]  # Re-index by the boolean

                # Fill drug at beginning for appropriate descriptions:
                j_counter = 0
                map_to_mask = {}
                for i, m in enumerate(drug_sample_mask.tolist()):
                    if not m:
                        continue
                    else:
                        map_to_mask[i] = j_counter
                        j_counter += 1
                        descriptions[i] = descriptions[i] + "\nDrug: <|drug|>"

                # Build input drugs:
                input_drug = []
                for tlist_i in text_list:
                    # Idea: filter out texts in the input that don't have valid drugs
                    # NOTE: text_list is already re-based to text inputs in the batch, so valid to map it to drug_sample_mask
                    sub_drugs = [
                        map_to_mask[j] for j in tlist_i if drug_sample_mask[j].item()
                    ]
                    input_drug.append(sub_drugs)

            else:
                unique_drug_indices = None
                input_drug = None

            # Need to interleave descriptions with context:
            remove_contexts = True

            context_texts = self._get_input_contexts(
                unique_aaseq_indices, unique_text_indices, aaseq_list, text_list
            )
            # Need to keep track of how many new context descriptions are added so that we can maintain
            # the mapping of within batch text IDs to global dataset text IDs.
            orig_num_descriptions = len(descriptions)
            if context_texts is not None:
                remove_contexts = False
                if (self.knowledge_domain == "disease") or (
                    self.knowledge_domain == "drug"
                ):
                    # Top-level - iterate over sublists:
                    # Context_texts are mapped by unique_aaseq_indices
                    # context_texts are unique
                    # Insert context_texts into descriptions:
                    # We know that context_texts is ordered by unique_aaseq_indices
                    aaseq_to_text_context_id = {
                        ai.item(): (len(descriptions) + i)
                        for i, ai in enumerate(unique_aaseq_indices)
                    }
                    descriptions += context_texts

                    # Filtering for dropout
                    for i in range(
                        len(instruct_list)
                    ):  # Should be len of demonstrations
                        if (
                            np.random.binomial(1, self.disease_function_context_dropout)
                            == 1
                        ):  # I.e., dropping
                            # If dropout, set to "" (empty string)
                            instruct_list[i] = instruct_list[i].replace("[CONTEXT]", "")
                            # Nothing if no dropout
                            continue
                        else:
                            # Make sure to replace [CONTEXT] with [EXT]
                            instruct_list[i] = instruct_list[i].replace(
                                "[CONTEXT]", "[EXT]"
                            )

                            # Rule: [EXT, CONTEXT]...  - Specific for QA
                            # Interleave text list with EXTs
                            context_idx = [
                                aaseq_to_text_context_id[unique_aaseq_indices[j].item()]
                                for j in aaseq_list[i]
                            ]
                            text_list[i] = [
                                item
                                for pair in zip(text_list[i], context_idx)
                                for item in pair
                            ]
                elif self.knowledge_domain == "ontology":
                    # Don't need to handle dropout because it's already been done
                    text_to_text_context_id = {
                        ai.item(): (len(descriptions) + i)
                        for i, ai in enumerate(unique_text_indices)
                    }
                    descriptions += context_texts

                    # Filtering for dropout
                    for i in range(
                        len(instruct_list)
                    ):  # Should be len of demonstrations
                        # Make sure to replace [CONTEXT] with [EXT]
                        instruct_list[i] = instruct_list[i].replace(
                            "[CONTEXT]", "[EXT]"
                        )

                        # Rule: [EXT, CONTEXT]...  - Specific for QA
                        # Interleave text list with EXTs
                        context_idx = [
                            text_to_text_context_id[unique_text_indices[j].item()]
                            for j in text_list[i]
                        ]
                        text_list[i] = [
                            item
                            for pair in zip(text_list[i], context_idx)
                            for item in pair
                        ]
                num_context_descriptions = len(descriptions) - orig_num_descriptions

            else:
                remove_contexts = True

            if remove_contexts:
                for i in range(len(instruct_list)):
                    instruct_list[i] = instruct_list[i].replace(
                        "[CONTEXT]", ""
                    )  # Remove instances of context if we aren't inserting

        unique_aaseq_indices_list = unique_aaseq_indices.tolist()
        if not isinstance(unique_text_indices, list):
            unique_text_indices_list = unique_text_indices.tolist()
        else:
            unique_text_indices_list = unique_text_indices
        unique_text_indices_list = unique_text_indices_list + [
            "NA" for _ in range(num_context_descriptions)
        ]

        rdict = {
            "data": {
                "seq": (
                    unique_aaseq_indices
                    if self.use_aaseq_embeddings
                    else unique_aaseq_toks
                ),
                "seq_idx": unique_aaseq_indices,
                "text": descriptions,
                "drug": unique_drug_indices,
            },
            "input": {
                "seq": aaseq_list,  # List of lists, not tensor
                "text": text_list,  # List of lists, not tensor
                "drug": input_drug,
            },
            "target": {
                "seq": None,
                "text": labels,
                "drug": None,
            },
            "instructions": instruct_list,
            "reference_indices": {
                "seq": unique_aaseq_indices_list,
                "text": unique_text_indices_list,
                "input": {
                    "seq": idxMapperNestedArrays(aaseq_list, unique_aaseq_indices_list),
                    "text": idxMapperNestedArrays(text_list, unique_text_indices_list),
                    "drug": None,
                },
                # Targets for QA are just yes/no
                "target": None,
            },
        }
        return rdict


class RetrievalCollator(BaseITCollator):
    TASK = "retrieval"

    def __init__(
        self, train_retrieval_lm=False, *args, **kwargs
    ):  # Init code contained in BaseIT collator
        super(RetrievalCollator, self).__init__(*args, **kwargs)
        self.train_retrieval_lm = train_retrieval_lm
        task_id = construct_task_id(
            aaseq_type=self.aaseq_type,
            text_type=self.text_type,
            relation_type=self.relation_type,
            task_type="retrieval",
        )
        json_file_path = (
            os.path.dirname(os.path.abspath(__file__))
            + f"/instruct_tune/tasks/{task_id}.json"
        )
        if self.use_instructions:
            with open(json_file_path, "r") as j:
                task = json.loads(j.read())
            if self.use_task_def_rephrasings:
                (
                    self.template,
                    self.description_true,
                    self.positive_examples_strs,
                    self.negative_example_strs,
                    self.example_text_ids,
                    self.example_aaseq_ids,
                ) = get_prompt_open_def(
                    task,
                    num_examples=self.num_examples,
                    sample_examples=self.sample_num_examples,
                    is_ppi=self.is_ppi,
                    aaseq_type=self.aaseq_type,
                )
            else:
                (
                    self.template,
                    self.positive_examples_strs,
                    self.negative_example_strs,
                    self.example_text_ids,
                    self.example_aaseq_ids,
                ) = get_prompt(
                    task,
                    num_examples=self.num_examples,
                    sample_examples=self.sample_num_examples,
                    is_ppi=self.is_ppi,
                    aaseq_type=self.aaseq_type,
                )
                self.description_true = None
        else:
            self.template = "[EXT] Protein: [PROT]"

    def __call__(
        self,
        batch_input: List[Tuple[Tuple[int], List[int]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Batch input structure (for GO):
            (prot_idx, rel_idx, text_idx), negative_protein_indices, negative_text_indices
        """

        if self.is_ppi:
            positive_aaseqs = [sample[0][0] for sample in batch_input] + [
                sample[0][2] for sample in batch_input
            ]
            positive_texts = []
            ntext_list = []
        else:
            positive_aaseqs = [sample[0][0] for sample in batch_input]
            positive_texts = [sample[0][2] for sample in batch_input]
            ntext_list = [sample[2] for sample in batch_input if sample[2] is not None]

        naaseq_list = [sample[1] for sample in batch_input if sample[1] is not None]
        if len(naaseq_list) > 0:
            negative_aaseqs = sum(naaseq_list, start=[])
        else:
            negative_aaseqs = None
        if len(ntext_list) > 0:
            negative_texts = sum(ntext_list, start=[])
        else:
            negative_texts = None

        rephrasing_guide = self._sample_rephrasing_levels(
            batch_length=len(positive_texts)
        )

        # Conduct sampling:
        if not self.use_instructions:  # Use null template
            template = self.template
            example_text_ids = []
            example_aaseq_ids = []
        elif self.sample_num_examples:
            template, example_text_ids, example_aaseq_ids = (
                sample_demonstrations_for_prompts(
                    template=self.template,
                    positive_examples=self.positive_examples_strs,
                    negative_examples=self.negative_example_strs,
                    example_text_ids=self.example_text_ids,
                    example_aaseq_ids=self.example_aaseq_ids,
                    is_ppi=self.is_ppi,
                )
            )
        else:
            template = self.template
            if self.use_task_def_rephrasings:
                # Rephrase def's in each template here:
                rephrased_defs = self._sample_task_def_rephrasings(
                    batch_length=len(positive_aaseqs), true_desc=self.description_true
                )

            example_text_ids = self.example_text_ids
            example_aaseq_ids = self.example_aaseq_ids

        # positive_proteins, negative_proteins, positive_texts, negative_texts hold original indices
        #   - Use these to access GO DF

        # These lines do a few things:
        #   1. Unifies protein indices in unique_protein_indices
        #   2. Creates mapping of positive and negative proteins
        if negative_aaseqs is not None:  # Proxy for non-in-batch negative sampling
            whole_aaseq_reference = (
                positive_aaseqs + negative_aaseqs + example_aaseq_ids
            )
            unique_aaseq_indices, new_id_aaseq_mapping = torch.unique(
                torch.LongTensor(whole_aaseq_reference), return_inverse=True
            )
            if self.is_ppi:
                (
                    positive_aaseqs_lhs_new,
                    positive_aaseqs_rhs_new,
                    negative_aaseqs_new,
                    example_aaseq_ids_new,
                ) = torch.split(
                    new_id_aaseq_mapping,
                    [
                        len(batch_input),
                        len(batch_input),
                        len(negative_aaseqs),
                        len(example_aaseq_ids),
                    ],
                )
            else:
                positive_aaseqs_new, negative_aaseqs_new, example_aaseq_ids_new = (
                    torch.split(
                        new_id_aaseq_mapping,
                        [
                            len(positive_aaseqs),
                            len(negative_aaseqs),
                            len(example_aaseq_ids),
                        ],
                    )
                )
        else:  # In-batch negative sampling
            whole_aaseq_reference = positive_aaseqs + example_aaseq_ids
            unique_aaseq_indices, new_id_aaseq_mapping = torch.unique(
                torch.LongTensor(whole_aaseq_reference), return_inverse=True
            )
            positive_aaseqs_new, example_aaseq_ids_new = torch.split(
                new_id_aaseq_mapping, [len(positive_aaseqs), len(example_aaseq_ids)]
            )
            negative_aaseqs_new = None
            if self.is_ppi:
                (
                    positive_aaseqs_lhs_new,
                    positive_aaseqs_rhs_new,
                    example_aaseq_ids_new,
                ) = torch.split(
                    new_id_aaseq_mapping,
                    [len(batch_input), len(batch_input), len(example_aaseq_ids)],
                )
            else:
                positive_aaseqs_new, example_aaseq_ids_new = torch.split(
                    new_id_aaseq_mapping, [len(positive_aaseqs), len(example_aaseq_ids)]
                )

        if not self.use_aaseq_embeddings:
            unique_aaseq_toks = self._convert_batch(
                "sequence", unique_aaseq_indices.tolist()
            )
        else:
            unique_aaseq_toks = None

        if self.is_ppi:
            input_aaseqs = [
                example_aaseq_ids_new.tolist() + x
                for x in positive_aaseqs_lhs_new.unsqueeze(1).tolist()
            ]  # For as many instructions, input that many aaseq's of examples into the model
            output_aaseqs_pos = positive_aaseqs_rhs_new.tolist()

            if negative_aaseqs is not None:
                assert len(negative_aaseqs_new) == len(positive_aaseqs_lhs_new)
                output_aaseqs_neg = negative_aaseqs_new.tolist()
            else:
                output_aaseqs_neg = None

            if self.use_task_def_rephrasings:
                sampled_templates = [
                    template.format(definition=rephrased_defs[i])
                    for i in range(len(positive_aaseqs_lhs_new))
                ]
            else:
                sampled_templates = [
                    template for _ in range(len(positive_aaseqs_lhs_new))
                ]
            instruct_list = sampled_templates

            # No texts to add in to instructions, so add empty lists for consistency for downstream
            # code. (i.e. enforce len(instructions) = len(aaseq_list) == len(text_list))
            text_list = [[] for i in range(len(instruct_list))]
            descriptions = []
            unique_text_indices = []

            unique_drug_indices = None
            input_drug = None
        else:
            # Do the same for text:
            if negative_texts is not None:  # Only used if not in-batch
                whole_text_reference = (
                    positive_texts + negative_texts + example_text_ids
                )
                unique_text_indices, new_id_text_mapping = torch.unique(
                    torch.LongTensor(whole_text_reference), return_inverse=True
                )
                positive_texts_new, negative_texts_new, example_text_ids_new = (
                    torch.split(
                        new_id_text_mapping,
                        [
                            len(positive_texts),
                            len(negative_texts),
                            len(example_text_ids),
                        ],
                    )
                )
            else:  # In-batch negative sampling
                whole_text_reference = positive_texts + example_text_ids
                unique_text_indices, new_id_text_mapping = torch.unique(
                    torch.LongTensor(whole_text_reference), return_inverse=True
                )
                positive_texts_new, example_text_ids_new = torch.split(
                    new_id_text_mapping, [len(positive_texts), len(example_text_ids)]
                )
                negative_texts_new = None

            # collect all negative relations. first consider those that permutate GOs, then those that permutate aaseqs.

            if (
                negative_texts is not None
            ):  # Don't even perform operations if we're using in-batch negative sampling
                negative_texts_final = torch.cat(
                    [
                        negative_texts_new,
                        torch.repeat_interleave(
                            positive_texts_new,
                            len(negative_aaseqs_new) // len(positive_aaseqs_new),
                        ),
                    ]
                ).tolist()

                if len(negative_texts) > 0:
                    # Don't need to generate more texts if we aren't actually sampling negative texts
                    # Negative sampling/construction of instructions:
                    raise NotImplementedError
                else:
                    descriptions_neg = None
            else:
                negative_texts_final = None
                descriptions_neg = None

            # Get string descriptions:
            if self.use_entity_compositions:
                descriptions_pos = self._sample_batch_entity_descriptions(
                    unique_text_indices.tolist()
                )  # Return is list
            else:
                descriptions_pos = self.text_sequences[
                    unique_text_indices.tolist()
                ].tolist()

            if descriptions_neg is not None:
                descriptions = descriptions_pos + descriptions_neg
            else:
                descriptions = descriptions_pos

            # Construct instruction within collator (change later):
            # No sampling, just static template usage for now
            if self.use_task_def_rephrasings:
                sampled_templates = [
                    template.format(definition=rephrased_defs[i])
                    for i in range(len(positive_aaseqs))
                ]
            else:
                sampled_templates = [
                    template for _ in range(len(positive_aaseqs))
                ]  # TEMPORARY UNTIL SAMPLING
            instruct_list = sampled_templates

            # Only need to input positive examples as the text list (not like QA where negative proteins go in input)
            input_aaseqs = [
                example_aaseq_ids_new.tolist() for _ in instruct_list
            ]  # For as many instructions, input that many aaseq's of examples into the model
            text_list = [
                example_text_ids_new.tolist() + lst
                for lst in positive_texts_new.unsqueeze(1).tolist()
            ]
            output_aaseqs_pos = positive_aaseqs_new.tolist()
            output_aaseqs_neg = negative_texts_final

            # Transform because of rephrasings:
            if self.use_entity_rephrasings and (self.rephrased_entities is not None):
                # This modifies a lot of variables
                descriptions, unique_text_indices, text_list = (
                    self._sample_batch_entities_with_rephrasings(
                        descriptions=descriptions,
                        unique_text_ids=unique_text_indices,
                        text_ids_expand=text_list,
                        rephrasing_guide=rephrasing_guide,
                    )
                )

                # Augment instructions with personality prompts
                if self.use_personality_prompts_rephrasing:
                    for ind in range(len(instruct_list)):
                        if rephrasing_guide[ind] is not None:
                            pp = PERSONALITY_PROMPTS[rephrasing_guide[ind][0]]
                            instruct_list[ind] = pp + "\n" + instruct_list[ind]

            # Process drugs if needed:
            if (
                self.drug_mask is not None
            ):  # If this passes, we know this is a drug-based dataset
                # Base everything off of text indices
                drug_sample_mask = self.drug_mask[unique_text_indices]
                unique_drug_indices = unique_text_indices[
                    drug_sample_mask
                ]  # Re-index by the boolean

                # Fill drug at beginning for appropriate descriptions:
                j_counter = 0
                map_to_mask = {}
                for i, m in enumerate(drug_sample_mask.tolist()):
                    if not m:
                        continue
                    else:
                        map_to_mask[i] = j_counter
                        j_counter += 1

                        descriptions[i] = descriptions[i] + "\nDrug: <|drug|>"

                # Build input drugs:
                input_drug = []
                for tlist_i in text_list:
                    # Idea: filter out texts in the input that don't have valid drugs
                    # NOTE: text_list is already re-based to text inputs in the batch, so valid to map it to drug_sample_mask
                    sub_drugs = [
                        map_to_mask[j] for j in tlist_i if drug_sample_mask[j].item()
                    ]
                    input_drug.append(sub_drugs)

            else:
                unique_drug_indices = None
                input_drug = None

        if isinstance(input_aaseqs, list):
            if not any(input_aaseqs):
                input_aaseqs = None

        unique_aaseq_indices_list = unique_aaseq_indices.tolist()
        if not isinstance(unique_text_indices, list):
            unique_text_indices_list = unique_text_indices.tolist()
        else:
            unique_text_indices_list = unique_text_indices
        rdict = {
            "data": {
                "seq": (
                    unique_aaseq_indices
                    if self.use_aaseq_embeddings
                    else unique_aaseq_toks
                ),
                "text": descriptions,
                "seq_idx": unique_aaseq_indices,
                "text_idx": unique_text_indices,
                "drug": unique_drug_indices,
                "rephrase_indicator": rephrasing_guide,
            },
            "input": {
                "seq": input_aaseqs,  # None bc no input sequences atm
                "text": text_list,
                "drug": input_drug,
            },
            "target": {
                "seq": {
                    "positive": output_aaseqs_pos,
                    "negative": output_aaseqs_neg,
                },
                "text": None,
                "drug": None,
            },  # Targets - provide seq if retrieval, text if LM-like task (QA or Retrieval)
            "instructions": instruct_list,  # Sampled instructions
            "reference_indices": {
                "input": {
                    "seq": idxMapperNestedArrays(
                        input_aaseqs, unique_aaseq_indices_list
                    ),
                    "text": idxMapperNestedArrays(text_list, unique_text_indices_list),
                    "drug": None,
                },
                "target": {
                    "seq": {
                        "positive": idxMapperNestedArrays(
                            output_aaseqs_pos, unique_aaseq_indices_list
                        ),
                        "negative": idxMapperNestedArrays(
                            output_aaseqs_neg, unique_aaseq_indices_list
                        ),
                    },
                    "text": None,
                },
            },
        }

        return rdict


class CaptionCollator(BaseITCollator):
    TASK = "caption"

    def __init__(self, *args, **kwargs):  # Init code contained in BaseIT collator
        super(CaptionCollator, self).__init__(*args, **kwargs)
        task_id = construct_task_id(
            aaseq_type=self.aaseq_type,
            text_type=self.text_type,
            relation_type=self.relation_type,
            task_type="caption",
        )
        json_file_path = (
            os.path.dirname(os.path.abspath(__file__))
            + f"/instruct_tune/tasks/{task_id}.json"
        )
        if self.use_instructions:
            with open(json_file_path, "r") as j:
                task = json.loads(j.read())
            # Update task:
            if self.knowledge_domain == "drug":
                if self.context_col is not None:
                    replace_drugbank_instructions(task, drugbank_col=self.context_col)
                (
                    self.template,
                    self.positive_examples_strs,
                    self.negative_example_strs,
                    self.example_text_ids,
                    self.example_aaseq_ids,
                ) = get_prompt(
                    task,
                    num_examples=self.num_examples,
                    sample_examples=self.sample_num_examples,
                    aaseq_type=self.aaseq_type,
                )
            elif self.use_task_def_rephrasings:
                (
                    self.template,
                    self.description_true,
                    self.positive_examples_strs,
                    self.negative_example_strs,
                    self.example_text_ids,
                    self.example_aaseq_ids,
                ) = get_prompt_open_def(
                    task,
                    num_examples=self.num_examples,
                    sample_examples=self.sample_num_examples,
                    is_ppi=self.is_ppi,
                    aaseq_type=self.aaseq_type,
                )
            else:
                (
                    self.template,
                    self.positive_examples_strs,
                    self.negative_example_strs,
                    self.example_text_ids,
                    self.example_aaseq_ids,
                ) = get_prompt(
                    task,
                    num_examples=self.num_examples,
                    sample_examples=self.sample_num_examples,
                    aaseq_type=self.aaseq_type,
                )
        else:
            self.template = "<|protein|> [ANSWER] [EXT]"

    def __call__(
        self,
        batch_input: List[Tuple[Tuple[int], List[int]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Batch input structure (for GO):
            (prot_idx, rel_idx, text_idx), negative_protein_indices, negative_text_indices
        """

        positive_aaseqs = [sample[0][0] for sample in batch_input]
        positive_texts = [sample[0][2] for sample in batch_input]
        # ONLY use the positive aaseq and text pairs

        rephrasing_guide = self._sample_rephrasing_levels(
            batch_length=len(positive_texts)
        )

        # Conduct sampling:
        if not self.use_instructions:  # Use null template
            template = self.template
            example_text_ids = []
            example_aaseq_ids = []
        elif self.sample_num_examples:
            template, example_text_ids, example_aaseq_ids = (
                sample_demonstrations_for_prompts(
                    template=self.template,
                    positive_examples=self.positive_examples_strs,
                    negative_examples=self.negative_example_strs,
                    example_text_ids=self.example_text_ids,
                    example_aaseq_ids=self.example_aaseq_ids,
                    is_ppi=self.is_ppi,
                )
            )
        else:
            template = self.template
            if self.use_task_def_rephrasings and (self.knowledge_domain != "drug"):
                # Rephrase def's in each template here:
                rephrased_defs = self._sample_task_def_rephrasings(
                    batch_length=len(positive_aaseqs), true_desc=self.description_true
                )

            example_text_ids = self.example_text_ids
            example_aaseq_ids = self.example_aaseq_ids

        # These lines do a few things:
        #   1. Unifies aaseq indices in unique_aaseq_indices
        #   2. Creates mapping of positive and negative aaseqs
        unique_aaseq_indices, new_id_aaseq_mapping = torch.unique(
            torch.LongTensor(positive_aaseqs + example_aaseq_ids), return_inverse=True
        )
        positive_aaseqs_new, example_aaseq_ids_new = torch.split(
            new_id_aaseq_mapping, [len(positive_aaseqs), len(example_aaseq_ids)]
        )
        if not self.use_aaseq_embeddings:
            unique_aaseq_toks = self._convert_batch(
                "sequence", unique_aaseq_indices.tolist()
            )
        else:
            unique_aaseq_toks = None

        # Do the same for text:
        unique_text_indices, new_id_text_mapping = torch.unique(
            torch.LongTensor(positive_texts + example_text_ids), return_inverse=True
        )
        positive_texts_new, example_text_ids_new = torch.split(
            new_id_text_mapping, [len(positive_texts), len(example_text_ids)]
        )

        assert len(positive_aaseqs_new) == len(positive_texts_new)

        if self.use_entity_compositions:
            descriptions = self._sample_batch_entity_descriptions(
                unique_text_indices.tolist()
            )  # Return is list
        else:
            descriptions = self.text_sequences[unique_text_indices.tolist()].tolist()

        if self.use_task_def_rephrasings and (self.knowledge_domain != "drug"):
            instruct_list = [
                template.format(definition=rephrased_defs[i])
                for i in range(len(positive_texts_new))
            ]
        else:
            instruct_list = [template for _ in range(len(positive_texts_new))]

        # Construct aaseq, text indices lists:
        aaseq_list = [
            example_aaseq_ids_new.tolist() + lst
            for lst in positive_aaseqs_new.unsqueeze(1).tolist()
        ]
        text_list = [
            example_text_ids_new.tolist() + lst
            for lst in positive_texts_new.unsqueeze(1).tolist()
        ]

        # Transform because of rephrasings:
        if self.use_entity_rephrasings and (self.rephrased_entities is not None):
            # This modifies a lot of variables
            descriptions, unique_text_indices, text_list = (
                self._sample_batch_entities_with_rephrasings(
                    descriptions=descriptions,
                    unique_text_ids=unique_text_indices,
                    text_ids_expand=text_list,
                    rephrasing_guide=rephrasing_guide,
                )
            )

            # Augment instructions with personality prompts
            if self.use_personality_prompts_rephrasing:
                for ind in range(len(instruct_list)):
                    if rephrasing_guide[ind] is not None:
                        pp = PERSONALITY_PROMPTS[rephrasing_guide[ind][0]]
                        instruct_list[ind] = pp + "\n" + instruct_list[ind]

        if self.insert_ontology_level and (self.knowledge_domain == "ontology"):
            target_caption_ids = set(
                [text_list[i][-1] for i in range(len(text_list))]
            )  # Chance that we exclude level if there's overlap in contexts and descriptions, but chance is low
            for i in range(len(descriptions)):
                if (
                    i in target_caption_ids
                ) and self.exclude_levels_in_ontology_captioning:
                    continue

                level = self.ontology_metadata[f"{self.text_type}_level"].iloc[
                    unique_text_indices[i].item()
                ]
                lname = self._ontology_level_transform(level)
                descriptions[i] = "Level: {}\n".format(lname) + descriptions[i]

        unique_drug_indices = None
        input_drug = None

        # Need to interleave descriptions with context:
        remove_contexts = True
        context_texts = self._get_input_contexts(
            unique_aaseq_indices,
            unique_text_indices,
            aaseq_list,
            text_list,
            get_drugs=(("drugbank" in self.text_type) and (self.drug_mask is not None)),
        )
        # Need to keep track of how many new context descriptions are added so that we can maintain
        # the mapping of within batch text IDs to global dataset text IDs.
        orig_num_descriptions = len(descriptions)
        if context_texts is not None:
            # Top-level - iterate over sublists:
            remove_contexts = False
            # Context_texts are mapped by unique_aaseq_indices
            # context_texts are unique
            if self.knowledge_domain == "disease":
                # Insert context_texts into descriptions:
                # We know that context_texts is ordered by unique_aaseq_indices
                aaseq_to_text_context_id = {
                    ai.item(): (len(descriptions) + i)
                    for i, ai in enumerate(unique_aaseq_indices)
                }
                descriptions += context_texts

                # Filtering for dropout
                for i in range(len(instruct_list)):  # Should be len of demonstrations
                    if (
                        np.random.binomial(1, self.disease_function_context_dropout)
                        == 1
                    ):  # I.e., dropping
                        # If dropout, set to "" (empty string)
                        instruct_list[i] = instruct_list[i].replace("[CONTEXT]", "")
                        # Nothing if no dropout
                        continue
                    else:
                        # Make sure to replace [CONTEXT] with [EXT]
                        instruct_list[i] = instruct_list[i].replace(
                            "[CONTEXT]", "[EXT]"
                        )

                        # Rule: [CONTEXT, EXT]...  - Note this is different for caption, QA, and retrieval
                        # Interleave text list with EXTs
                        context_idx = [
                            aaseq_to_text_context_id[unique_aaseq_indices[j].item()]
                            for j in aaseq_list[i]
                        ]
                        text_list[i] = [
                            item
                            for pair in zip(context_idx, text_list[i])
                            for item in pair
                        ]
            elif self.knowledge_domain == "ontology":
                # Don't need to handle dropout because it's already been done
                text_to_text_context_id = {
                    ai.item(): (len(descriptions) + i)
                    for i, ai in enumerate(unique_text_indices)
                }
                descriptions += context_texts

                # Filtering for dropout
                for i in range(len(instruct_list)):  # Should be len of demonstrations
                    # Make sure to replace [CONTEXT] with [EXT]
                    instruct_list[i] = instruct_list[i].replace("[CONTEXT]", "[EXT]")

                    # Rule: [CONTEXT, EXT]...  - Note this is different for caption, QA, and retrieval
                    # Interleave text list with EXTs
                    context_idx = [
                        text_to_text_context_id[unique_text_indices[j].item()]
                        for j in text_list[i]
                    ]
                    text_list[i] = [
                        item for pair in zip(context_idx, text_list[i]) for item in pair
                    ]
            elif self.knowledge_domain == "drug":
                # Don't need to handle dropout because it's already been done
                if self.drug_mask is not None:
                    (
                        stacked_contexts,
                        flat_contexts,
                        all_drug_struct,
                        all_drug_indices,
                    ) = context_texts
                    # All drug struct inserted in context function
                    unique_drug_indices = torch.LongTensor(all_drug_struct)
                    input_drug = all_drug_indices
                else:
                    stacked_contexts, flat_contexts = context_texts

                # Filtering for dropout
                counting = 0
                for i in range(len(instruct_list)):  # Should be len of demonstrations
                    # Make sure to replace [CONTEXT] with [EXT]
                    instruct_list[i] = instruct_list[i].replace("[CONTEXT]", "[EXT]")
                    len_before = len(descriptions)
                    descriptions += stacked_contexts[i]
                    context_idx = [
                        len_before + j for j in range(len(stacked_contexts[i]))
                    ]  # We know that stacked contexts are ordered
                    text_list[i] = [
                        item for pair in zip(context_idx, text_list[i]) for item in pair
                    ]
            else:
                raise NotImplementedError
        else:
            remove_contexts = True
        if remove_contexts:
            for i in range(len(instruct_list)):
                instruct_list[i] = instruct_list[i].replace(
                    "[CONTEXT]", ""
                )  # Remove instances of context if we aren't inserting

        if self.evaluation:  # NEED TO USE THIS OPTION
            # Need to trim instructions to not include last [EXT] and leave out the last description

            target_text = [t[-1] for t in text_list]
            text_list = [t[:-1] for t in text_list]  # Trim up to last included

            # We know the last part of the string for each instruction will contain "[EXT]", thus we need to remove it
            for i in range(len(instruct_list)):
                instruct_list[i] = instruct_list[i][
                    :-6
                ]  # Excludes " [EXT]" so the final part of the string will end in "... [ANSWER]"
        else:
            target_text = None

        # If it's a drug dataset, insert drug structure into the context

        if isinstance(aaseq_list, list):
            if not any(aaseq_list):
                aaseq_list = None

        unique_aaseq_indices_list = unique_aaseq_indices.tolist()
        if not isinstance(unique_text_indices, list):
            unique_text_indices_list = unique_text_indices.tolist()
        else:
            unique_text_indices_list = unique_text_indices

        # Add NAs for the index mapping for all context descriptions.
        num_context_descriptions = len(descriptions) - orig_num_descriptions
        unique_text_indices_list = unique_text_indices_list + [
            "NA" for _ in range(num_context_descriptions)
        ]

        rdict = {
            "data": {
                "seq": (
                    unique_aaseq_indices
                    if self.use_aaseq_embeddings
                    else unique_aaseq_toks
                ),
                "seq_idx": unique_aaseq_indices,
                "text": descriptions,
                "text_idx": unique_text_indices_list,
                "drug": unique_drug_indices,
            },
            "input": {
                "seq": aaseq_list,
                "text": text_list,  # List not tensor
                "drug": input_drug,
            },
            "target": {
                "seq": None,
                "text": target_text,
                "drug": None,
            },
            "instructions": instruct_list,
            "reference_indices": {
                "input": {
                    "seq": idxMapperNestedArrays(aaseq_list, unique_aaseq_indices_list),
                    "text": idxMapperNestedArrays(text_list, unique_text_indices_list),
                    "drug": None,
                },
                "target": {
                    "seq": None,
                    "text": idxMapperNestedArrays(
                        target_text, unique_text_indices_list
                    ),
                },
            },
        }

        return rdict
