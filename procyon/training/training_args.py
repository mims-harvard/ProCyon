import os
from enum import Enum
from dataclasses import dataclass, field, fields
from transformers.training_args import TrainingArguments
from typing import Tuple
from dataclasses import asdict

from procyon.data.data_utils import DATA_DIR, MODEL_DIR


def fill_modelargs_from_dict(d):
    margs = ModelArgs()
    for k, v in d.items():
        setattr(margs, k, v)
    return margs


def replace_data_dir(data_obj):
    for field, value in asdict(data_obj).items():
        if isinstance(value, str):
            setattr(data_obj, field, value.replace('DATA_DIR', DATA_DIR).replace('MODEL_DIR', MODEL_DIR))


@dataclass
class ModelArgs:
    # TODO: @Owen, @Tom, revise the model arguments here. Remove any unused args

    ######################## Encoders ########################
    ############ Protein
    protein_encoder_num_params: str = field(
        default="650m",
        metadata={
            "help": "Version of ESM to use as protein encoder. (Associated weights should be stored in DATA_DIR/model_weights/).",
            "choices": ["3b", "650m", "35m", "8m"],
        }
    )
    # TODO: Replace all protein with aaseq
    aaseq_encoder_num_params: str = field(
        default="650m",
        metadata={
            "help": "Version of ESM to use as AA seq encoder. (Associated weights should be stored in DATA_DIR/model_weights/).",
            "choices": ["3b", "650m", "35m", "8m"],
        }
    )
    protein_tokenizer_name: str = field(
        default="ESM-1b",
        metadata={
            "help": "Protein tokenizer name"
        }
    )
    aaseq_tokenizer_name: str = field(
        default="ESM-1b",
        metadata={
            "help": "AA seq tokenizer name"
        }
    )
    max_protein_len: int = field(
        default = 1024,
        metadata = {
            "help": "Max number of residues for a protein seq in one forward pass"
        }
    )
    max_aaseq_len: int = field(
        default = 1024,
        metadata = {
            "help": "Max number of residues for an AA seq in one forward pass"
        }
    )
    long_protein_strategy: str = field(
        default = "split",
        metadata = {
            "help": "Chosen strategy for long protein sequences.",
            "choices": ["split", "truncate"]
        }
    )
    long_aaseq_strategy: str = field(
        default = "split",
        metadata = {
            "help": "Chosen strategy for long AA sequences.",
            "choices": ["split", "truncate"]
        }
    )
    is_protein_tokenized: bool = field(
        default=False,
        metadata={
            "help": "Whether or not the protein sequences are already tokenized."
        }
    )
    is_aaseq_tokenized: bool = field(
        default=False,
        metadata={
            "help": "Whether or not the protein sequences are already tokenized."
        }
    )
    
    protein_pooling_opt: str = field(
        default = "max",
        metadata = {
            "help": "Chosen option for protein pooling."
        }
    )
    aaseq_pooling_opt: str = field(
        default = "max",
        metadata = {
            "help": "Chosen option for AA seq pooling."
        }
    )
    protein_enc_batch_limit: int = field(
        default=None,
        metadata={
            "help": "Max number of protein chunks to encode in one forward pass for the protein encoder"
        }
    )
    aaseq_enc_batch_limit: int = field(
        default=None,
        metadata={
            "help": "Max number of AA seq chunks to encode in one forward pass for the AA seq encoder"
        }
    )

    ############ Text
    text_encoder_fname: str = field(
        default="BioGPT-Large",
        metadata={
            "help": "fname of text sequence pretrained model weights (stored in DATA_DIR/model_weights/).",
            "choices": ["BioGPT-Large", "biogpt", "pubmedgpt"]
        }
    )
    text_tokenizer_name: str = field(
        default="BioGPT-Large", 
        metadata={
            "help": "Text tokenizer name"
        }
    )
    max_text_len: int = field(
        default=1024,
        metadata={
            "help": "Max text sequence length"
        }
    )
    text_pooler_type: str = field(
        default = None,
        metadata = {
            "help": "Chosen option for text pooling (see implementation of Pooler class). Defaults to 'cls' for BERT based models, and 'final' for GPT based models.",
            "choices": ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last", "avg_all", "final_avg", "final"],
        }
    )
    is_go_tokenized: bool = field(
        default=False,  
        metadata={
            "help": "Whether or not the GO terms are already tokenized."
        }
    )
    is_pfam_tokenized: bool = field(
        default=False,  
        metadata={
            "help": "Whether or not the Pfam terms are already tokenized."
        }
    )
    
    

    ######################## Decoder ########################
    decoder_dim: int = field(
        default=512,
        metadata={
            "help": "Number of dimensions for decoder."
        }
    )
    decoder_nlayers: int = field(
        default=3,
        metadata={
            "help": "Number of layers for an MLP decoder"
        }
    )
    protein_text_combine_strategy: str = field(
        default = 'concat',
        metadata = {
            "help": "Strategy to combine sequence and text outputs",
            "choices": ["concat", "max"]
        }
    )
    # TODO: Replace all protein with aaseq
    aaseq_text_combine_strategy: str = field(
        default = 'concat',
        metadata = {
            "help": "Strategy to combine AA sequence and text outputs",
            "choices": ["concat", "max"]
        }
    )

    ######################## Shallow embeddings ########################
    use_text_embeddings: bool = field(
        default = True,
        metadata = {
            "help": "If true, uses GO embeddings by retrieving from path specified in model_args.",
        }
    )
    go_embeddings_path: str = field(
        default = os.path.join(DATA_DIR, "generated_data/node_embeddings/go/go_description_embeddings_BioGPT-Large_final_token.pt"),
        metadata = {
            "help": "[If model_args.use_text_embeddings is True] The path to retrieve GO embeddings from.",
        }
    )
    pfam_embeddings_path: str = field(
        default = os.path.join(DATA_DIR, "generated_data/node_embeddings/pfam/pfam_plus_interpro_description_embeddings_BioGPT-Large_final_token.pt"),
        metadata = {
            "help": "[If model_args.use_text_embeddings is True] The path to retrieve GO embeddings from.",
        }
    )
    drugbank_embeddings_path: str = field(
        default = os.path.join(DATA_DIR, "generated_data/node_embeddings/drug/drugbank_background_moa_embeddings_BioGPT-Large_final_token.pt"),
        metadata={
            "help": "[If model_args.use_text_embeddings is True] The path to retrieve drugbank embeddings from.",
        }
    )
    reactome_embeddings_path: str = field(
        default = os.path.join(DATA_DIR, "generated_data/node_embeddings/reactome/reactome_description_embeddings_BioGPT-Large_final_token.pt"),
        metadata={
            "help": "[If model_args.use_text_embeddings is True] The path to retrieve reactome embeddings from.",
        }
    )
    omim_embeddings_path: str = field(
        default = os.path.join(DATA_DIR, "generated_data/node_embeddings/reactome/omim_description_embeddings_BioGPT-Large_final_token.pt"),
        metadata={
            "help": "[If model_args.use_text_embeddings is True] The path to retrieve reactome embeddings from.",
        }
    )
    ec_embeddings_path: str = field(
        default = os.path.join(DATA_DIR, "generated_data/node_embeddings/reactome/ec_description_embeddings_BioGPT-Large_final_token.pt"),
        metadata={
            "help": "[If model_args.use_text_embeddings is True] The path to retrieve reactome embeddings from.",
        }
    )
    # TODO: Replace all protein with aaseq
    use_aaseq_embeddings: bool = field(
        default = False,
        metadata = {
            "help": "If true, uses protein and domains embeddings by retrieving from saved path",
        }
    )
    protein_seq_embeddings_path: str = field(
        default = os.path.join(DATA_DIR, "generated_data/node_embeddings/protein/protein_esm2-3b_mean.pt"),
        metadata = {
            "help": "[If model_args.use_protein_embeddings is True] The path to retrieve protein embeddings from.",
        }
    )
    protein_embeddings_idmap_path: str = field(
        default = os.path.join(DATA_DIR, '/generated_data/node_embeddings/protein/protein_esm2-3b_mean.pkl'),
        metadata = {
            "help": "[If model_args.use_protein_embeddings is True] The path to retrieve protein embedding idmap from.",
        }
    )
    domain_embeddings_path: str = field(
        default = os.path.join(DATA_DIR, "/generated_data/node_embeddings/domain/domain_esm2-3b_mean.pt"),
        metadata = {
            "help": "[If model_args.use_domain_embeddings is True] The path to retrieve domain embeddings from.",
        }
    )
    domain_embeddings_idmap_path: str = field(
        default = os.path.join(DATA_DIR, 'generated_data/node_embeddings/domain/domain_esm2-3b_mean.pkl'),
        metadata = {
            "help": "[If model_args.use_domain_embeddings is True] The path to retrieve domain embedding idmap from.",
        }
    )
    mouse_ortholog_embeddings_path: str = field(
        default = os.path.join(DATA_DIR, "/generated_data/node_embeddings/mouse_ortholog/mouse_ortholog_esm2-3b_max.pt"),
        metadata = {
            "help": "[If model_args.use_mouse_ortholog_embeddings is True] The path to retrieve mouse protein embeddings from.",
        }
    )
    mouse_ortholog_embeddings_idmap_path: str = field(
        default=os.path.join(DATA_DIR, '/generated_data/node_embeddings/mouse_ortholog/mouse_ortholog_esm2-3b_max.pkl'),
        metadata={
            "help": "[If model_args.use_mouse_ortholog_embeddings is True] The path to retrieve mouse protein embedding idmap from.",
        }
    )

    ######################## Model Training / Freezing ########################

    # TODO: make this more robust to different model sizes
    freeze_protein_encoder: str = field(
        default=None,
        metadata={
            "help":"Whether or not to freeze the protein encoder.",
            "choices":[None, "embed", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "all", "lora", "adapter", "prefix"]
        }
    )
    freeze_aaseq_encoder: str = field(
        default=None,
        metadata={
            "help":"Whether or not to freeze the AA seq encoder.",
            "choices":[None, "embed", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "all"]
        }
    )
    # TODO: make this more robust to different model sizes
    freeze_text_encoder: str = field(
        default=None,
        metadata={
            "help":"Whether or not to freeze the text encoder.",
            "choices":[None, "embed", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', "all", "lora", "adapter", "prefix"]
        }
    )
    freeze_text_embeddings: bool = field(
        default=True,
        metadata={
            "help":"Whether or not to freeze the GO and Pfam shallow embeddings."
        }
    )
    freeze_aaseq_embeddings: bool = field(
        default=False,
        metadata={
            "help":"Whether or not to freeze the potein and domain shallow embeddings."
        }
    )

    # NOTE: The below arguments have been moved to freeze_text_encoder and freeze_protein_encoder arguments
    # use_lora: bool = field(
    #     default=False,
    #     metadata={
    #         'help': 'use lora'
    #     }
    # )
    
    # use_adapter: bool = field(
    #     default=False,
    #     metadata={
    #         'help': 'use adapter'
    #     }
    # )
    
    # use_prefix: bool = field(
    #     default=False,
    #     metadata={
    #         'help': 'use prefix'
    #     }
    # )

    # Breakdown by protein vs. text encoder:
    aaseq_lora_alpha: float = field(
        default=8,
        metadata = {
            'help': 'scaling up lora weights'
        }
    )

    aaseq_lora_r: int = field(
        default=8,
        metadata = {
            'help': 'lora dimension'
        }
    )

    aaseq_adapter_rank: int = field(
        default=8,
        metadata = {
            'help': 'lora dimension'
        }
    )

    text_lora_alpha: float = field(
        default=8,
        metadata = {
            'help': 'scaling up lora weights'
        }
    )

    text_lora_r: int = field(
        default=8,
        metadata = {
            'help': 'lora dimension'
        }
    )

    text_adapter_rank: int = field(
        default=8,
        metadata = {
            'help': 'lora dimension'
        }
    )

    def __post_init__(self):
        replace_data_dir(self)


@dataclass
class DataArgs:
    """dataset and data collator instantiation args"""
    # ablations
    use_protein_mlm: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to use protein MLM."
        }
    )
    use_text_cl: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use text CL (supervised SimCSE)."
        }
    )
    use_text_cl_unsupervised_only: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use text CL (unsupervised SimCSE)."
        }
    )
    use_protein_go_cl: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to use protein-go CL."
        }
    )
    use_protein_protein_cl: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use protein-protein CL."
        }
    )
    # use_go_go_cl: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "Whether or not to use go-go CL."
    #     }
    # )
    # use_pfam_cl: bool = field(
    #     default=True,
    #     metadata={
    #         "help": "Whether or not to use pfam-based CL. If False, turns off pfam cl altogether."
    #     }
    # )
    # use_pfam_protein_cl: bool = field(
    #     default=True,
    #     metadata={
    #         "help":"Whether or not to use pfam-protein based CL for pfam CL. No effect if use_pfam_cl is False."
    #     }
    # )
    # use_pfam_go_cl: bool = field(
    #     default=True,
    #     metadata={
    #         "help":"Whether or not to use pfam-go based CL for pfam CL. No effect if use_pfam_cl is False."
    #     }
    # )
    # use_pfam_pfam_cl: bool = field(
    #     default=True,
    #     metadata={
    #         "help":"Whether or not to use pfam-pfam based CL for pfam CL. No effect if use_pfam_cl is False."
    #     }
    # )
    use_domain_go_cl: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to use domain-go CL."
        }
    )
    use_domain_pfam_cl: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to use domain-Pfam CL."
        }
    )
    
    # data:
    data_dir: str = field(
        default=DATA_DIR,
        metadata={
            "help": "Path to pretrain data."
        }
    )  # the fnames are hard-coded in the dataset classes to reduce redundancy
    go_split_method: str = field(
        default="sample_aware_ontology_go_centric",
        metadata={
            "help": "The method to split GO terms into CL train, CL val, eval pt-ft, eval few-shot, and eval zero-shot sets.",
            "choices": ["sample_random_random_go_centric", "sample_aware_random_go_centric", "sample_random_temporal_go_centric", "sample_aware_temporal_go_centric", "sample_random_ontology_go_centric", "sample_aware_ontology_go_centric"]
        }
    )
    pfam_split_method: str = field(
        default="random_pfam_centric",
        metadata={
            "help": "The method to split Pfam terms into CL train, CL val, eval pt-ft, eval few-shot, and eval zero-shot sets.",
            "choices": ["random_pfam_centric", "clan_aware_pfam_centric"]
        }
    )
    go_def_col: str = field(
        default="description_combined",
            metadata={
            "help": "The name of the column to use for GO descriptions in X-GO CL (for generated_data/node_data/go/go_descriptions.pkl).",
        }
    )
    pfam_def_col: str = field(
        default="description_combined",
            metadata={
            "help": "The name of the column to use for Pfam descriptions in X-Pfam CL (for generated_data/node_data/go/go_descriptions.pkl [TODO: this file does not exist yet]).",
        }
    )
    # negative sampling
    negative_sampling_strategy_protein_go: str = field(
        default="go_only",
        metadata={
            "choices":["go_only", "protein_go_both", "protein_only"], 
            "help":"Negative sampling strategy for protein-go CL."
        }
    )
    negative_sampling_strategy_protein_protein: str = field(
        default="protein_both",
        metadata={
            "choices":["protein_both"],
            "help":"Negative sampling strategy for protein-protein CL."
        }
    )
    # negative_sampling_strategy_pfam_go: str = field(
    #     default="go_only",
    #     metadata={
    #         "choices":["go_only", "pfam_go_both"],
    #         "help":"Negative sampling strategy for pfam-go CL."
    #     }
    # )
    # negative_sampling_strategy_pfam_protein: str = field(
    #     default="protein_only",
    #     metadata={
    #         "choices":["protein_only"],
    #         "help":"Negative sampling strategy for pfam-protein CL."
    #     }
    # )    
    # negative_sampling_strategy_pfam_pfam: str = field(
    #     default="both",
    #     metadata={
    #         "choices":["both", "tail_only", "head_only"],
    #         "help":"Negative sampling strategy for pfam-pfam CL."
    #     }
    # )    
    negative_sampling_strategy_domain_go: str = field(
        default="go_only",
        metadata={
            "choices":["go_only", "domain_go_both", "domain_only"], 
            "help":"Negative sampling strategy for domain-go CL."
        }
    )
    negative_sampling_strategy_domain_pfam: str = field(
        default="pfam_only",
        metadata={
            "choices":["pfam_only", "domain_pfam_both", "domain_only"], 
            "help":"Negative sampling strategy for domain-Pfam CL."
        }
    )
    
    use_only_goa_gos: bool = field(
        default=True,
        metadata={
            "help":"Whether or not to use only GOA GOs for protein-GO GO negative sampling.",
        }
    )
    # TODO: Rename as protein_go
    use_only_protein_go_gos: bool = field(
        default=True,
        metadata={
            "help":"Whether or not to use only GOA GOs for protein-GO GO negative sampling.",
        }
    )
    use_only_goa_proteins: bool = field(
        default=True,
        metadata={
            "help":"Whether or not to use only GOA proteins for protein-GO protein negative sampling.",
        }
    )
    use_only_protein_go_proteins: bool = field(
        default=True,
        metadata={
            "help":"Whether or not to use only GOA proteins for protein-GO protein negative sampling.",
        }
    )
    use_only_ppi_proteins: bool = field(
        default=True,
        metadata={
            "help":"Whether or not to use only PPI proteins for PPI protein negative sampling.",
        }
    )
    use_only_protein_protein_proteins: bool = field(
        default=True,
        metadata={
            "help":"Whether or not to use only PPI proteins for PPI protein negative sampling.",
        }
    )
    # use_only_pfam_go_gos: bool = field(
    #     default=True,
    #     metadata={
    #         "help":"Whether or not to use only Pfam GOs for Pfam-GO GO negative sampling."
    #     }
    # )
    # use_only_pfam_protein_proteins: bool = field(
    #     default=True,
    #     metadata={
    #         "help":"Whether or not to use only Pfam proteins for Pfam-protein protein negative sampling."
    #     }
    # )
    # # TODO improve help messages and/or naming for these (and above)
    # use_only_pfam_pfam_pfams: bool = field(
    #     default=True,
    #     metadata={
    #         "help":"Whether or not to use only Pfam Pfams for Pfam-Pfam Pfam negative sampling."
    #     }
    # )
    use_only_domain_go_gos: bool = field(
        default=True,
        metadata={
            "help":"Whether or not to use only domain-GO GOs for domain-GO GO negative sampling.",
        }
    )
    use_only_domain_go_domains: bool = field(
        default=True,
        metadata={
            "help":"Whether or not to use only domain-GO domains for domain-GO domain negative sampling.",
        }
    )
    use_only_domain_pfam_pfams: bool = field(
        default=True,
        metadata={
            "help":"Whether or not to use only domain-Pfam Pfams for domain-Pfam Pfam negative sampling.",
        }
    )
    use_only_domain_pfam_domains: bool = field(
        default=True,
        metadata={
            "help":"Whether or not to use only domain-Pfam domains for domain-Pfam domain negative sampling.",
        }
    )
    
    num_neg_samples_protein_go_per_go: int = field(
        default=64,
        metadata={
            "help":"Number of negative samples per GO end for protein-GO CL."
        }
    )
    num_neg_samples_protein_go_per_protein: int = field(
        default=2,
        metadata={
            "help":"Number of negative samples per protein for protein-GO CL."    
        }
    )
    num_neg_samples_protein_protein_per_protein: int = field(
        default=2,
        metadata={
            "help":"Number of negative samples per protein for protein-protein CL."    
        }
    )
    # num_neg_samples_pfam_go_per_go: int = field(
    #     default=128,
    #     metadata={
    #         "help":"Number of negative samples per GO end for Pfam-GO CL."
    #     }
    # )
    # num_neg_samples_pfam_protein_per_protein: int = field(
    #     default=2,
    #     metadata={
    #         "help":"Number of negative samples per protein for Pfam-protein CL."
    #     }
    # )
    # num_neg_samples_pfam_pfam_per_pfam: int = field(
    #     default=1,
    #     metadata={
    #         "help":"Number of negative samples per Pfam for Pfam-GO CL."
    #     }
    # )

    # Note: there are only 70+ component GO terms that have a domain-GO annotation. since during negative sampling we only select GOs belonging to the same namespace as negatives, and also mask out GOs with positive association with the domain, the possible candidate set of negative samples of GOs can go below 64.
    num_neg_samples_domain_go_per_go: int = field(
        default=32,
        metadata={
            "help":"Number of negative samples per GO end for domain-GO CL."
        }
    )
    num_neg_samples_domain_go_per_domain: int = field(
        default=2,
        metadata={
            "help":"Number of negative samples per domain for domain-GO CL."    
        }
    )
    num_neg_samples_domain_pfam_per_pfam: int = field(
        default=64,
        metadata={
            "help":"Number of negative samples per Pfam end for domain-Pfam CL."
        }
    )
    num_neg_samples_domain_pfam_per_domain: int = field(
        default=2,
        metadata={
            "help":"Number of negative samples per domain for domain-Pfam CL."    
        }
    )    
    
    go_sims_type: str = field(
        default="jaccard",
        metadata={
            "help":"Type of GO sims to use for negative sampling."
        }
    )
    protein_sims_type: str = field(
        default="esm2-650m_embeds_cosine",
        metadata={
            "help":"Type of protein sims to use for negative sampling.",
            "choices":["esm2-650m_embeds_cosine", "esm2-3b_embeds_cosine", "levenstein", "None"]
        }
    )
    domain_sims_type: str = field(
        default="esm2-650m_embeds_cosine",
        metadata={
            "help":"Type of domain sims to use for negative sampling.",
            "choices":["esm2-650m_embeds_cosine", "esm2-3b_embeds_cosine", "levenstein", "None"]
        }
    )
    pfam_sims_type: str = field(
        default="biogpt_embeds_cosine",
        metadata={
            "help":"Type of pfam sims to use for negative sampling. Note that this is used only for Pfam as text, but not aggregation of domains.",
            "choices":["biogpt_embeds_cosine", "dummy", "None"]
        }
    )
    
    # # Pfam data
    # num_domains_sampled_per_pfam: int = field(
    #     default=4,
    #     metadata={
    #         "help":"Number of domains sampled per pfam for constructing Pfam embeddings."
    #     }
    # )
    # num_pfam_neighbors_sampled_per_pfam: int = field(
    #     default=3,
    #     metadata={
    #         "help":"Number of pfam neighbors sampled per pfam for pfam-pfam CL."
    #     }
    # )
    # num_proteins_sampled_per_pfam: int = field(
    #     default=4,
    #     metadata={
    #         "help":"Number of proteins sampled per pfam for pfam-protein CL."
    #     }
    # )
    # num_gos_sampled_per_pfam: int = field(
    #     default=16,
    #     metadata={
    #         "help":"Number of GOs sampled per pfam for pfam-go CL."
    #     }
    # )

    # TODO: REPEATING TWO BELOW ATTRIBUTES DUE TO BUG, CHANGE LATER
    protein_mlm_probability: float = field(
        default=0.15,
        metadata={
            "help":"Probability of masking each token for protein MLM."
        }
    )
    protein_mlm_masking_strategy: str = field(
        default="esm2",
        metadata={
            "help":"Masking strategy for protein MLM."
        }
    )
    
    # MLM
    protein_mlm_probability: float = field(
        default=0.15,
        metadata={
            "help":"Probability of masking each token for protein MLM."
        }
    )
    protein_mlm_masking_strategy: str = field(
        default="esm2",
        metadata={
            "help":"Masking strategy for protein MLM."
        }
    )

    #Relation extraction
    relation_file: str = field(
        default = os.path.join(DATA_DIR, "integrated_data/relation2id_model_input.csv"),
        metadata = {
            "help":"File containing relational information, including symmetry requirements and number of relations.",
        }
    )

    def __post_init__(self):
        replace_data_dir(self)
    

@dataclass
class TrainArgs(TrainingArguments):
    """dataloading, training and optimization args"""
    # dataloading
    protein_mlm_num_workers: int = field(
        default=4,
        metadata={
            "help": "Number of workers to collate protein sequence dataset."
        }
    )
    text_cl_num_workers: int = field(
        default=4,
        metadata={
            "help": "Number of workers to collate protein sequence dataset."
        }
    )
    protein_go_num_workers: int = field(
        default=4,
        metadata={
            "help": "Number of workers to collate protein-go dataset."
        }
    )
    protein_protein_num_workers: int = field(
        default=4,
        metadata={
            "help": "Number of workers to collate protein-protein dataset."
        }
    )
    # pfam_num_workers: int = field(
    #     default=4,
    #     metadata={
    #         "help": "Number of workers to collate pfam relations dataset."
    #     }
    # )
    # go_go_dataloader_num_workers: int = field(
    #     default=1,
    #     metadata={"help": "Number of workers to collate go-go dataset."}
    # )
    domain_go_num_workers: int = field(
        default=4,
        metadata={
            "help": "Number of workers to collate domain-go dataset."
        }
    )
    domain_pfam_num_workers: int = field(
        default=4,
        metadata={
            "help": "Number of workers to collate domain-Pfam dataset."
        }
    )
    protein_mlm_batch_size: int = field(
        default=2,
        metadata={
            "help": "Batch size (num proteins) for protein MLM dataloader per GPU."
        }
    )
    text_cl_batch_size: int = field(
        default=6,
        metadata={
            "help": "Batch size (num gold sequences) for text CL dataloader per GPU. Not including augmented seqs or negatives."
        }
    )
    protein_go_batch_size: int = field(
        default=6,
        metadata={
            "help": "Batch size (num protein-GOs) for protein-go dataloader per GPU."
        }
    )
    protein_protein_batch_size: int = field(
        default=6,
        metadata={
            "help": "Batch size (num PPIs) for protein-protein dataloader per GPU."
        }
    )
    # pfam_batch_size: int = field(
    #     default=1,
    #     metadata={
    #         "help": "Batch size (num Pfams) for Pfam relations dataloader per GPU. MUST BE 1 FOR NOW."
    #     }
    # )
    # go_go_batch_size: int = field(
    #     default=8,
    #     metadata={
    #         "help": "Batch size (num GO-GOs) for GO-GO dataloader per GPU."
    #     }
    # )
    domain_go_batch_size: int = field(
        default=6,
        metadata={
            "help": "Batch size (num domain-GOs) for domain-go dataloader per GPU."
        }
    )
    domain_pfam_batch_size: int = field(
        default=6,
        metadata={
            "help": "Batch size (num domain-Pfams) for domain-Pfam dataloader per GPU."
        }
    )
    
    # loss
    mlm_loss_weight: float = field(
        default=.5,
        metadata={
            "help":"Weight (alpha) of MLM loss."
        }
    )
    text_cl_loss_weight: float = field(
        default=.1,
        metadata={
            "help":"Weight (alpha) of text_cl loss."
        }
    )
    kepler_margin: float = field(
        default=1.0,
        metadata={
            "help":"Margin for loss function."
        }
    )
    # TODO: Investigate this
    optimize_memory: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to optimize memory when computering the loss function of negative samples. "
        }
    )
    # NOTE: `gradient_accumulation_steps` is already a TrainingArguments attribute
    
    # optimization
    optimizer_type: str = field(
        default="adamw",
        metadata={
            "help":"Type of optimizer to use.",
            "choices":["adamw", "adafactor", "radam"]
        }
    )
    protein_encoder_lr: float = field(
        default=1e-5,
        metadata={
            "help":"Learning rate for protein encoder."
        }
    )
    # TODO: Replace with aaseq
    aaseq_encoder_lr: float = field(
        default=1e-5,
        metadata={
            "help":"Learning rate for protein encoder."
        }
    )
    text_encoder_lr: float = field(
        default=1e-5,
        metadata={
            "help":"Learning rate for text encoder."
        }
    )
    embedding_lr: float = field(
        default=1e-4,
        metadata={
            "help":"Learning rate for shallow embeddings (lookup tables)."
        }
    )
    decoder_lr: float = field(
        default=1e-3,
        metadata={
            "help":"Learning rate for decoder."
        }
    )
    weight_decay: float = field(
        default=0.01,
        metadata={
            "help":"Weight decay."
        }
    )
    
    # training
    max_steps: int = field(
        default=100000,
        metadata={
            "help":"Number of training steps."
        }
    )
    debug: bool = field(
        default=False,
        metadata={
            "help":"Whether or not to run in debug mode."
        }
    )
    overfit_first_batch: bool = field(
        default=False,
        metadata={
            "help":"Whether or not to run in use a single batch repeatedly (for testing)."
        }
    )
    log_interval: int = field(
        default=20,
        metadata={
            "help":"Number of steps between logging."
        }
    )
    checkpoint_steps: int = field(
        default=500,
        metadata={
            "help":"Number of steps between ephemeral checkpoint saving."
        }
    )
    eval_steps: int = field(
        default=5000,
        metadata={
            "help":"Number of steps between persistent checkpoint saving & evaluation."
        }
    )
    initial_eval_steps: int = field(
        default=1000,
        metadata={
            "help":"Number of steps between persistent checkpoint saving & evaluation until initial_eval_steps_limit."
        }
    )
    initial_eval_steps_limit: int = field(
        default=10000,
        metadata={
            "help":"Limit for initial_eval_steps"
        }
    )
    eval_on_first_step: bool = field(
        default=False,
        metadata={
            'help':"Whether to do checkpoint saving & evaluation before any training occurs."
        }
    )
    eval_text_batch_size: bool = field(
        default=64,
        metadata={
            'help':"Batch size to use when generating text embeddings for eval."
        }
    )

    # Early stopping (if enabled we check for early stopping every `eval_steps` (including `initial_eval_steps`), since eval is being performed anyway)
    early_stopping: bool = field(
        default=False,
        metadata={
            'help':"Whether to enable early stopping."
        }
    )
    early_stopping_patience: int = field(
        default=5000,
        metadata={
            'help':"Number of steps after last improvement mebefore early stopping (actual value also depends on measurement frequency as controlled by `eval_steps`)."
        }
    )
    early_stopping_delta: float = field(
        default=0.01,
        metadata={
            'help':"Minimum increase in metric over previous best to qualify as an improvement."
        }
    )
    
    # seed: int = field(
    #     default=42,
    #     metadata={
    #         "help":"Random seed."
    #     }
    # )
    # warmup_steps: int = field(
    #     default=200,
    #     metadata={
    #         "help":"Number of warmup steps."
    #     }
    # )   
    # lr_scheduler_type: str = field(
    #     default="linear",
    #     metadata={
    #         "help":"Type of learning rate scheduler.",
    #         "choices":["linear", "cosine", "constant"]
    #     }
    # )
    # warmup_ratio: float = field(
    #     default=0.1,
    #     metadata={
    #         "help":"Ratio of warmup steps."
    #     }
    # )
    # fp16: bool = field(
    #     default=False,
    #     metadata={
    #         "help":"Whether or not to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit."
    #     }
    # )
    # NOTE: `eval_steps`, `seed`, `warmup_steps`, `lr_scheduler_type`, `fp16`, and `warmup_ratio` are already in the base class `TrainingArguments`
    
    # logistics
    from_yaml: str = field(
        default=None,
        metadata={
            "help":"Whether or not to load arguments from a YAML file, and if so, where to load from."
        }
    )
    from_json: str = field(
        default=None,
        metadata={
            "help":"Whether or not to load arguments from a JSON file, and if so, where to load from."
        }
    )
    # save_path: str = field(
    #     default=f"{DATA_DIR}/model_outputs/pretrain/",
    #     metadata={
    #         "help":"Path to save model."
    #     }
    # )
    output_dir: str = field(
        default=os.path.join(DATA_DIR, "model_outputs/pretrain/"),
        metadata={
            "help":"Path to save model."
        }
    )

    run_name: str = field(
        default=None,
        metadata={
            "help":"Name of run."
        }
    )

    run_name_suffix: str = field(
        default=None,
        metadata={
            "help":"Suffix for name of run."
        }
    )

    model_type: str = field(
        default = "lm",
        metadata = {
            "help": "Type of model to use. ",
            "choices": ["lm", "linkpred"]
        }
    )

    # run_name: str | None = field(
    #     default=None,
    #     metadata={
    #         "help":"Name of run."
    #     }
    # )
    # resume_from_checkpoint: str = field(
    #     default=None,
    #     metadata={
    #         "help":"Path to checkpoint to resume from."
    #     }
    # )
    # NOTE: `run_name` and `resume_from_checkpoint` are already in the base class `TrainingArguments`
    
    resume_wandb_id: str = field(
        default=None,
        metadata={
            "help":"WandB id of existing run to resume"
        }
    )

    slurm_job_id: int = field(
        default=None,
        metadata={
            "help":"Automatically set in __post_init__ if running on a SLURM cluster."
        }
    )

    def __post_init__(self):
        super().__post_init__()
        replace_data_dir(self)
        self.warmup_steps = int(self.max_steps * self.warmup_ratio) if self.warmup_steps is None else self.warmup_steps

        # Set `self.slurm_job_id` (may be None)
        self.slurm_job_id = os.environ.get('SLURM_JOB_ID')


def to_dict(args):
    """ Adapted from transformers.TrainingArguments.to_dict()
    Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
    the token values by removing their value.
    """
    # filter out fields that are defined as field(init=False)
    d = dict((field.name, getattr(args, field.name)) for field in fields(args) if field.init)
    for k, v in d.items():
        if isinstance(v, Enum):
            d[k] = v.value
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
            d[k] = [x.value for x in v]
        if k.endswith("_token"):
            d[k] = f"<{k.upper()}>"
    return d
        

# FIXME: update
def get_hparams(args_tuple: Tuple[dataclass]):
    # keys_include = set([
    #     'freeze_protein_encoder',
    #     'protein_encoder_num_params',
    #     "go_split_method",
    #     "pfam_split_method",
    #     "text_encoder_fname",
    #     'freeze_text_encoder',
        
    #     # TODO: To replace
    #     'freeze_aaseq_encoder',
    #     'aaseq_encoder_num_params',
    #     'aaseq_tokenizer_name',
    #     'max_aaseq_len',
    #     'aaseq_encoder_lr',
    #     'use_aaseq_embeddings',
        
    #     # data
    #     'use_protein_mlm',
    #     'use_text_cl',
    #     'use_text_cl_unsupervised_only',
    #     'use_protein_go_cl',
    #     'use_protein_protein_cl',
    #     'use_domain_go_cl',
    #     'use_domain_pfam_cl',
    #     # 'use_pfam_cl',
    #     # 'use_pfam_pfam_cl',
    #     # 'use_pfam_protein_cl',
    #     # 'use_pfam_go_cl',
    #     'protein_tokenizer_name',
    #     'text_tokenizer_name',
    #     'max_protein_len',
    #     'max_text_len',
        
    #     # negative sampling
    #     "negative_sampling_strategy_protein_go",
    #     "negative_sampling_strategy_protein_protein",
    #     # "negative_sampling_strategy_pfam_go",
    #     # "negative_sampling_strategy_pfam_protein",
    #     # "negative_sampling_strategy_pfam_pfam",
    #     "negative_sampling_strategy_domain_go",
    #     "negative_sampling_strategy_domain_pfam",
    #     "use_only_goa_gos",
    #     "use_only_goa_proteins",
    #     "use_only_ppi_proteins",
    #     # "use_only_pfam_go_gos",
    #     # "use_only_pfam_protein_proteins",
    #     "use_only_domain_go_gos",
    #     "use_only_domain_go_domains",
    #     "use_only_domain_pfam_pfams",
    #     "use_only_domain_pfam_domains",

    #     "num_neg_samples_protein_go_per_go",
    #     "num_neg_samples_protein_go_per_protein",
    #     "num_neg_samples_protein_protein_per_protein",
    #     # "num_neg_samples_pfam_go_per_go",
    #     # "num_neg_samples_pfam_protein_per_protein",
    #     # "num_neg_samples_pfam_pfam_per_pfam",
    #     "num_neg_samples_domain_go_per_go",
    #     "num_neg_samples_domain_go_per_domain",
    #     "num_neg_samples_domain_pfam_per_pfam",
    #     "num_neg_samples_domain_pfam_per_domain",
    
    #     "go_sims_type",
    #     "protein_sims_type",
    #     "domain_sims_type",
    #     "pfam_sims_type",
    
    #     # # Pfam data
    #     # "num_domains_sampled_per_pfam",
    #     # "num_pfam_neighbors_sampled_per_pfam",
    #     # "num_proteins_sampled_per_pfam",
    #     # "num_gos_sampled_per_pfam",

    #     # protein mlm
    #     "protein_mlm_probability",
    #     "protein_mlm_masking_strategy",
        
    #     # dataloading
    #     "protein_mlm_batch_size",
    #     "text_cl_batch_size",
    #     "protein_go_batch_size",
    #     # "protein_protein_batch_size",
    #     "domain_go_batch_size",
    #     "domain_pfam_batch_size",

    #     # from saved embeddings:
    #     "use_protein_embeddings",
    #     "protein_seq_embeddings_path",
    #     "freeze_protein_embeddings",
    #     "domain_embeddings_path",
    #     "freeze_domain_embeddings",
    #     "use_text_embeddings",
    #     "go_embeddings_path",
    #     "pfam_embeddings_path"
    #     "freeze_text_embeddings",
    #     "use_domain_embeddings",
        
    #     # training
    #     "max_steps",
    #     "fp16",
    #     "local_rank",
    #     "world_size",
    #     "resume_from_checkpoint",
    #     "model_type",
        
    #     # loss and optimization
    #     "mlm_loss_weight",
    #     "kepler_margin",
    #     "protein_encoder_lr",
    #     "text_encoder_lr",
    #     "decoder_lr",
    #     "embedding_lr",
    #     "eval_steps",
    #     "seed",
    #     "warmup_steps",
    #     "lr_scheduler_type",
    #     "warmup_ratio",
    #     "weight_decay",

    #     # PEFT arguments:
    #     "aaseq_lora_alpha",
    #     "aaseq_lora_r",
    #     "aaseq_adapter_rank",
        
    #     "text_lora_alpha",
    #     "text_lora_r",
    #     "text_adapter_rank"
        
    #     # TODO: @Tom, @Owen, add more model args here
    # ])
    
    hparams = dict()
    for args in args_tuple:
        args_dict = to_dict(args)
        for key, value in args_dict.items():
            # if key in keys_include:
                hparams[key] = value
    
    return hparams

