import os
from enum import Enum
from dataclasses import dataclass, field, fields
from transformers.training_args import TrainingArguments
from typing import Tuple
from dataclasses import asdict

from procyon.data.data_utils import DATA_DIR, HOME_DIR
from procyon.data.constants import EXPERTISE_LEVEL, REPHRASE_ENTITY_LEVEL, REPHRASE_TASK_DEF_LEVEL


def fill_modelargs_from_dict(d):
    margs = ModelArgs()
    for k, v in d.items():
        setattr(margs, k, v)
    return margs


def replace_custom_dir(data_obj):
    for field, value in asdict(data_obj).items():
        if isinstance(value, str):
            setattr(data_obj, field, value.replace('DATA_DIR', DATA_DIR))
            setattr(data_obj, field, value.replace("HOME_DIR", HOME_DIR))


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
    protein_encoder_debug: bool = field(
        default=False,
        metadata={
            "help": "Whether or not the Pfam terms are already tokenized."
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
        default="llama-2-7b-hf",
        metadata={
            "help": "fname of text sequence pretrained model weights (stored in DATA_DIR/model_weights/).",
            "choices": ["llama-2-7b-hf", "llama-3-8b"]
        }
    )
    text_encoder_debug: bool = field(
        default=False,
        metadata={
            "help": "debug mode of text encoder"
        }
    )
    text_tokenizer_name: str = field(
        default="llama-2-7b-hf",
        metadata={
            "help": "Text tokenizer name"
        }
    )
    max_text_len: int = field(
        default=1024,
        metadata={
            "help": "Max text length to input to model"
        }
    )
    text_pooling_opt: str = field(
        default = "special_token",
        metadata = {
            "help": "Chosen option for text pooling.",
            "choices": ["mean", "max", "special_token"],
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
    ret_token_access: str = field(
        default = 'all',
        metadata={
            "help": "How to access ret token - used for internal testing",
            "choices": ["last", "all"]
        }
    )

    model_splitting: bool = field(
        default=False,
        metadata={
            "help": "Whether or not use model splitting"
        }
    )

    n_model_pieces: int = field(
        default=2,
        metadata={
            "help": "Split model into n gpus"
        }
    )

    use_lora: bool = field(
        default=False,
        metadata={
            "help": "Whether or not use lora"
        }
    )

    use_q_lora: bool = field(
        default=False,
        metadata={
            "help": "Whether or not use q-lora"
        }
    )

    attention_type: str = field(
        default='vanilla',
        metadata={
            "help": "vanilla/flash-attetion in LLM",
            "choices": ["vanilla", "flash_attn_v1", "streaming_llm"],
        }
    )

    protein_attention_type: str = field(
        default='vanilla',
        metadata={
            "help": "vanilla/flash-attetion in LLM",
            "choices": ["vanilla", "flash_attn"],
        }
    )

    streaming_llm_max_gen_len: int = field(
        default=50,
        metadata={
            "help": "The maximum length of generated text"
        }
    )

    protein_lora_parameters: str = field(
        default='default',
        metadata={
            "help": "default/full in LLM",
            "choices": ["attn", 'mlp', "full"],
        }
    )

    protein_task_spc_lora: bool = field(
        default=False,
        metadata={
            "help": "Whether or not use task specific lora in protein encoder"
        }
    )

    protein_task_spc_lora_num: int = field(
        default=2
    )

    text_task_spc_lora: bool = field(
        default=False,
        metadata={
            "help": "Whether or not use task specific lora in test encoder"
        }
    )

    text_task_spc_lora_num: int = field(
        default=2
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
        default = False,
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
    use_drug_embeddings: bool = field(
        default = False,
        metadata = {
            "help": "If True, use the drug embeddings in the model from pre-saved embeddings"
        }
    )
    use_protein_struct: bool = field(
        default = False,
        metadata = {
            "help": "If true, models protein structure alongside aa-seq when forward passing proteins are used",
        }
    )
    protein_struct_dropout: float = field(
        default = 0.5,
        metadata = {
            "help": "Dropout level for protein structures if use_protein_struct==True. Represents probability of dropout, i.e., 0.0 would mean never dropout the protein structure token."
        }
    )
    protein_seq_embeddings_path: str = field(
        default = os.path.join(DATA_DIR, "generated_data/node_embeddings/protein/protein_esm2-3b_mean.pt"),
        metadata = {
            "help": "[If model_args.use_protein_embeddings is True] The path to retrieve protein embeddings from.",
        }
    )
    protein_struct_embeddings_path: str = field(
        default = os.path.join(DATA_DIR, "generated_data/node_embeddings/protein/protein_gearnet.pt"),
        metadata = {
            "help": "If use_protein_struct is True, this should contain structural embeddings for all proteins, as aligned by index."
        }
    )
    protein_embeddings_idmap_path: str = field(
        default = os.path.join(DATA_DIR, 'generated_data/node_embeddings/protein/protein_esm2-3b_mean.pkl'),
        metadata = {
            "help": "[If model_args.use_protein_embeddings is True] The path to retrieve protein embedding idmap from.",
        }
    )
    drug_struct_embeddings_path: str = field(
        default = os.path.join(DATA_DIR, "integrated_data/v1/drugbank/drugbank_embeds.pt"),
        metadata = {
            "help": "If use_drug_embeddings is True, this should contain embeddings for all drugs, as aligned by index."
        }
    )
    domain_embeddings_path: str = field(
        default = os.path.join(DATA_DIR, "generated_data/node_embeddings/domain/domain_esm2-3b_mean.pt"),
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
    peptide_embeddings_path: str = field(
        default = None,
        metadata = {
            "help": "The path to retrieve peptide embeddings from.",
        }
    )
    peptide_embeddings_idmap_path: str = field(
        default = None,
        metadata = {
            "help": "The path to retrieve peptide embedding idmap from.",
        }
    )
    mouse_ortholog_embeddings_path: str = field(
        default = os.path.join(DATA_DIR, "generated_data/node_embeddings/mouse_ortholog/mouse_ortholog_esm2-3b_max.pt"),
        metadata = {
            "help": "[If model_args.use_mouse_ortholog_embeddings is True] The path to retrieve mouse protein embeddings from.",
        }
    )
    mouse_ortholog_embeddings_idmap_path: str = field(
        default=os.path.join(DATA_DIR, 'generated_data/node_embeddings/mouse_ortholog/mouse_ortholog_esm2-650m.pkl'),
        metadata={
            "help": "[If model_args.use_mouse_ortholog_embeddings is True] The path to retrieve mouse protein embedding idmap from.",
        }
    )

    ######################## Model Training / Freezing ########################

    # TODO: make this more robust to different model sizes
    freeze_protein_encoder: str = field(
        default=None,
        metadata={
            "help":"Whether or not to freeze the protein encoder. Numbers are TOP-END, meaning if you choose 10, layers greater than 10 will be tuned, ones less than 10 frozen.",
            "choices":[None, 'None', "embed", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "all", "lora", "qlora", "adapter", "prefix"]
        }
    )
    # TODO: make this more robust to different model sizes
    freeze_text_encoder: str = field(
        default=None,
        metadata={
            "help":"Whether or not to freeze the text encoder.",
            "choices":[None, 'None', "embed", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', "all", "lora", "qlora", "adapter", "prefix"]
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
            "help":"Whether or not to freeze the protein and domain shallow embeddings."
        }
    )

    ######################## PEFT parameters ########################
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

    ########################## Contrastive learning method ##########################

    cl_method: str = field(
        default = 'infonce',
        metadata = {
            'help': 'Type of contrastive learning method used for [PROT] token optimization.',
            "choices": ['infonce'],
        }
    )

    use_projection_cl: bool = field(
        default = False,
        metadata = {
            'help': 'Whether to use a projection layer (adapter a la SimCLR) in InfoNCE loss.'
        }
    )

    causal_qa: bool = field(
        default = True,
        metadata = {
            'help': 'Whether or not to use the causal language generation system for QA. If not, uses a separate MLP trained on the hidden states.'
        }
    )

    train_retrieval_lm: bool = field(
        default = False,
        metadata = {
            'help': 'Whether to train language modeling objective when training retrieval'
        }
    )

    train_qa_full_lm: bool = field(
        default = False,
        metadata = {
            "help": "If True, trains each QA task with a full language modeling objective rather than just on answers"
        }
    )

    ########################## Projection layer arguments ##########################
    num_layers_token_projector: int = field(
        default = 1,
        metadata = {
            "help": "Number of layers in protein to token projection",
        }
    )
    hidden_size_token_projector: int = field(
        default = 256,
        metadata = {
            "help": "Size of hidden layers in token projector",
        }
    )
    num_layers_shared_projector: int = field(
        default = 1,
        metadata = {
            "help": "Number of layers in protein to shared projection",
        }
    )
    hidden_size_shared_projector: int = field(
        default = 256,
        metadata = {
            "help": "Size of hidden layers in shared projector",
        }
    )
    num_layers_lm_projector: int = field(
        default = 1,
        metadata = {
            "help": "Number of layers in [PROT] to shared projection",
        }
    )
    hidden_size_lm_projector: int = field(
        default = 256,
        metadata = {
            "help": "Size of hidden layers in lm projector",
        }
    )
    roll_num: int = field( # TEST------------
        default = -1,
        metadata = {
            "help": "How far to roll the retrieved [PROT] indices from input. Remember that the labels are shifted to the left (-1) automatically, so -1 is standard."
        }
    )

    residual_dropout: float = field(
        default=0.0
    )

    ## TESTING: Duplicate argument from DataArgs:
    negative_sampling_strategy_retrieval: str = field(
        default='in_batch',
        metadata={
            "choices":["go_only", "protein_go_both", "protein_only", "in_batch"],
            "help":"Negative sampling strategy for protein-protein CL."
        }
    )

    context_crop_sampling: bool = field(
        default = False,
        metadata = {
            "help": "If True, crops the context of given inputs"
        }
    )

    context_crop_sampling_qa: bool = field(
        default = False,
        metadata = {
            "help": "If True, crops the context of inputs in QA tasks. If context_crop_sampling is True, this is overwritten to also be true."
        }
    )

    context_crop_sampling_retrieval: bool = field(
        default = False,
        metadata = {
            "help": "If True, crops the context of inputs in retrieval tasks. If context_crop_sampling is True, this is overwritten to also be true."
        }
    )

    context_crop_sampling_caption: bool = field(
        default = False,
        metadata = {
            "help": "If True, crops the context of inputs in caption tasks. If context_crop_sampling is True, this is overwritten to also be true."
        }
    )

    enforce_checkpoint_architecture_strict: bool = field(
        default = False,
        metadata = {
            "help": "If True, enforces that architecture of checkpoint config and target config match exactly"
        }
    )

    contrastive_global: bool = field(
        default = False,
        metadata = {
            "help": "If True, gathers negatives from across all GPUs to assist in batch expansion for contrastive learning"
        }
    )

    lora_specific_style: str = field(
        default='specific',
        metadata={
            'choices': ['specific', 'single_lora', 'qa_retrieval_share']
        }
    )

    filter_negatives_by_id_contrastive: bool = field(
        default = False,
        metadata = {
            "help": "If True, filters contrastive learning negatives by text and protein IDs. Advised to only use this with contrastive_global."
        }
    )

    protein_pooling_correction_option: bool = field(
        default = False,
        metadata = {
            "help": "Makes a correction for the previous method we used to generate protein embeddings. "
                    "Necessary if you're switching a model instance to using ESM module if pretrained on embedding library (e.g., using load_plm_directly in from_pretrained). "
                    "If questions, ask Owen."
        }
    )

    def __post_init__(self):
        #super().__post_init__()
        replace_custom_dir(self)

@dataclass
class DataArgs:
    """dataset and data collator instantiation args"""
    it_data_config_yml: str = field(
        default = None,
        metadata = {
            "help": "Path to yaml configuration of instruction tuning datasets "
                    "to use. Note that this will override other arguements "
                    "regarding which datasets to use.",
        },
    )

    # ablations
    use_protein_mlm: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to use protein MLM."
        }
    )
    use_qa: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to use text CL (supervised SimCSE)."
        }
    )
    use_retrieval: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to use protein-go CL."
        }
    )
    use_caption: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use protein-protein CL."
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
        default="sample_random_random_go_centric",
        metadata={
            "help": "The method to split GO terms into CL train, CL val, eval pt-ft, eval few-shot, and eval zero-shot sets.",
            "choices": ["sample_random_random_go_centric", "sample_random_random_pair_centric", "sample_aware_random_go_centric", "sample_random_time_aware_go_centric", "sample_aware_time_aware_go_centric", "sample_random_ontology_aware_go_centric", "sample_aware_ontology_aware_go_centric"]
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
        #default="description_combined",
        default='desc_concise_summary', # TODO: Change this default later when old models are deprecated
        metadata={
            "help": "The name of the text to use for GO descriptions during training.",
            "choices": ["description_combined", "standard", "name_def", "def_only"]
        }
    )
    pfam_def_col: str = field(
        default="description_combined",
        metadata={
            "help": "The name of the column to use for Pfam descriptions in X-Pfam CL (for generated_data/node_data/go/go_descriptions.pkl [TODO: this file does not exist yet]).",
        }
    )
    text_variant_type: str = field(
        default='standard',
        metadata={
            "help": "The type of description to use for text.",
        }
    )
    # negative sampling
    negative_sampling_strategy_qa: str = field(
        default="text_only",
        metadata={
            "choices":["text_only", "aaseq_text_both", "aaseq_only"],
            "help":"Negative sampling strategy for protein-go CL."
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

    num_neg_samples_qa: int = field(
        default = 1,
        metadata = {
            "help": "Number of negative samples across all datasets for Q&A instruction type."
        }
    )
    num_neg_samples_retrieval: int = field(
        default = 2,
        metadata = {
            "help": "Number of negative samples across all datasets for retrieval instruction type."
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

    aaseq_subset_tsv_path: str = field(
        default=None,
        metadata={
            "help": "Path to a TSV file containing aaseq IDs to subset dataset to. This file must "
                    "contain at least one column named `seq_id`.",
        },
    )

    # PPI
    ppi_store_reverse_edges: bool = field(
        default=False,
        metadata={
            "help": "Whether to store reverse of each PPI edge as a separate relation. Typically relation file is "
                    "assumed to be undirected, so if set to True will store (a1, a2) and (a2, a1) even if only "
                    "(a1, a2) occurs in the file. This can be useful in eval settings where we want to make sure "
                    "each protein in the pair shows up as a query."
        },
    )
    ppi_edge_swap_prob: bool = field(
        default=0.5,
        metadata={
            "help": "Probability of 'swapping' a PPI relation such that the target becomes the query and vice-versa. "
                    "Automatically set to 0 if `ppi_store_reverse_edges` is True, but otherwise defaults to 0.5. "
                    "Should be set to 0 for evaluations."
        },
    )

    val_split_type: str = field(
        default = 'pt_ft',
        metadata = {
            'help': 'Type of split to use for validation. Temporary until we combine validation dataframes.',
            'choices': ['pt_ft', 'five_shot', 'zero_shot']
        }
    )

    # Instruction tuning-specific parameters: -------------------------------------------
    num_instruction_examples: int = field(
        default = None,
        metadata = {
            "help": "Upper limit on number of instruction examples to use. If None, uses the default, preset number."
        }
    )

    sample_num_instruction_examples: bool = field(
        default = False,
        metadata = {
            "help": "If True, samples the number of instructions during each batch construction"
        }
    )

    # Composition options:
    use_entity_compositions: bool = field(
        default = False,
        metadata = {
            "help": "If True, uses the compositions for entity descriptions found for each dataset"
        }
    )
    sample_entity_compositions: str = field(
        default="uniform",
        metadata={
            "help": "How to sample entity compositions. Several options given.",
            "choices": ["uniform"]
        }
    )

    #
    use_instructions: bool = field(
        default=True,
        metadata={
            "help": "Instruction tuning if True"
        }
    )

    use_perplexity_filtered_set: bool = field(
        default=False,
        metadata={
            "help": "If true, attempt to use the perplexity filtered set for all datasets"
        }
    )

    qa_subset_version: int = field(
        default = None,
        metadata = {
            "help": "For qa training, define the subset (as defined in procyon.data.constants) of fields to use",
            "choices": [None, 1] # Add to options here if we change it later
        }
    )

    retrieval_subset_version: int = field(
        default = None,
        metadata = {
            "help": "For retrieval training, define the subset (as defined in procyon.data.constants) of fields to use",
            "choices": [None, 1] # Add to options here if we change it later
        }
    )

    caption_subset_version: int = field(
        default = None,
        metadata = {
            "help": "For caption training, define the subset (as defined in procyon.data.constants) of fields to use",
            "choices": [None, 1, 2] # Add to options here if we change it later
        }
    )

    insert_disease_function_context: bool = field(
        default = False,
        metadata = {
            "help": "If True, inserts functional information (from UniProt) for disease datasets",
        }
    )

    disease_function_context_dropout: float = field(
        default = None,
        metadata = {
            "help": "If > 0, drops out disease function context at this rate, i.e., higher dropout leads to dropping out more often",
        }
    )

    insert_go_ontology_context: bool = field(
        default = False,
        metadata = {
            "help": "If True, inserts ontology context for Gene Ontology dataset"
        }
    )

    go_ontology_rag_level_upper_limit: int = field(
        default = 5,
        metadata = {
            "help": "Defines the maximum level that an ancestor can be for GO ontology context insertion (RAG)"
        }
    )

    go_ontology_rag_num_context: int = field(
        default = 3,
        metadata = {
            "help": "Defines the number of in-context ontology terms for GO RAG. If go_ontology_rag_sample_num_context is True, then this is an upper bound"
        }
    )

    go_ontology_rag_sample_num_context: bool = field(
        default = False,
        metadata = {
            "help": "If True, samples the number of in-context information for GO ontology RAG"
        }
    )

    use_go_ontology_level_groups: bool = field(
        default = True,
        metadata = {
            "help": "If True, uses discrete groups as defined in ONTOLOGY_RAG_LEVEL_GROUPS instead of numerical levels for GO"
        }
    )

    insert_reactome_ontology_context: bool = field(
        default = False,
        metadata = {
            "help": "If True, inserts ontology context for Reactome dataset"
        }
    )

    reactome_ontology_rag_level_upper_limit: int = field(
        default = 5,
        metadata = {
            "help": "Defines the maximum level that an ancestor can be for Reactome ontology context insertion (RAG)"
        }
    )

    reactome_ontology_rag_num_context: int = field(
        default = 3,
        metadata = {
            "help": "Defines the number of in-context ontology terms for GO RAG. If reactome_ontology_rag_sample_num_context is True, then this is an upper bound"
        }
    )

    reactome_ontology_rag_sample_num_context: bool = field(
        default = False,
        metadata = {
            "help": "If True, samples the number of in-context information for Reactome ontology RAG"
        }
    )

    insert_go_ontology_level: bool = field(
        default = False,
        metadata = {
            "help": "If True, inserts the level (binned into three categories) for GO dataset"
        }
    )

    insert_reactome_ontology_level: bool = field(
        default = False,
        metadata = {
            "help": "If True, inserts the level (binned into three categories) for Reactome dataset"
        }
    )

    use_reactome_ontology_level_groups: bool = field(
        default = True,
        metadata = {
            "help": "If True, uses discrete groups as defined in ONTOLOGY_RAG_LEVEL_GROUPS instead of numerical levels for Reactome"
        }
    )

    use_drug_context_augmentation: bool = field(
        default = False,
        metadata = {
            "help": "If True, inserts additional information to the context for drug tasks"
        }
    )
    exclude_levels_in_ontology_captioning: bool = field(
        default = False,
        metadata = {
            "help": "If True, does not include 'Level:N' prefix for ontology-based datasets in the caption target text. Should set this to True, set to False here for backwards compatibility."
        }
    )
    shuffle_seed_metadataset: int = field(
        default = None,
        metadata = {
            "help": "If None, doesn't shuffle. If given an integer, shuffles the metadataset indices by the given seed."
        }
    )

    # REPHRASINGS ARGUMENTS --------------------------------------------------------------
    use_entity_rephrasings: bool = field(
        default = False,
        metadata = {
            "help": "If True, use rephrased entities (descriptions from datasets)"
        }
    )
    use_task_def_rephrasings: bool = field(
        default = False,
        metadata = {
            "help": "If True, use rephrased task descriptions (descriptions with each instruction)"
        }
    )
    rephrasing_sample_prob: float = field(
        default = 0.5,
        metadata = {
            "help": "Probability of sampling one of the rephrased entities during training when using rephrased entities"
        }
    )
    rephrase_caption_entities: bool = field(
        default = False,
        metadata = {
            "help": "If True, samples entity description for captioning task"
        }
    )
    use_personality_prompts_rephrasing: bool = field(
        default = False,
        metadata = {
            "help": "If True, use prompts that indicate personality for the type of rephrase"
        }
    )
    fixed_rephrasing_expertise_level: str = field(
        default = None,
        metadata = {
            "help": "If provided, fixes the expertise for the sampled rephrasings at both the task definition and text entity (both are tied).",
            "choices": EXPERTISE_LEVEL, # ["junior", "mid", "senior"]
        }
    )
    fixed_rephrasing_entity_rephrase_level: str = field(
        default = None,
        metadata = {
            "help": "If provided, fixes the sampled entity rephrasings at this level (rephrasing level).",
            "choices": REPHRASE_ENTITY_LEVEL, # ["rephrasing", "summarisation"]
        }
    )
    fixed_rephrasing_task_def_rephrase_level: str = field(
        default = None,
        metadata = {
            "help": "If provided, fixes the sampled task definition rephrasing at this level (rephrasing level).",
            "choices": REPHRASE_TASK_DEF_LEVEL, # ["rephrasing", "summarisation", 'simplification']
        }
    )
    # EXPERTISE_LEVEL, REPHRASE_ENTITY_LEVEL, REPHRASE_TASK_DEF_LEVEL

    def __post_init__(self):
        #super().__post_init__()
        replace_custom_dir(self)


@dataclass
class TrainArgs(TrainingArguments):
    """dataloading, training and optimization args"""
    # dataloading
    num_workers: int = field(
        default = 1,
        metadata = {
            "help": "Number of workers to collate all datasets."
        }
        # NOTE: Applies to all datasets, unlike previous arguments which broke them up by dataset
    )

    deepspeed_config: str = field(
        default = None,
        metadata = {
            "help": "Deepspeed config file",
        }
    )

    # Batch sizes:
    protein_mlm_batch_size: int = field(
        default=2,
        metadata={
            "help": "Batch size (num proteins) for protein MLM dataloader per GPU."
        }
    )
    qa_batch_size: int = field(
        default=4,
        metadata={
            "help": "Batch size for each instruction dataloaders per GPU."
        }
    )
    retrieval_batch_size: int = field(
        default=8,
        metadata={
            "help": "Batch size for each instruction dataloaders per GPU."
        }
    )
    caption_batch_size: int = field(
        default=8,
        metadata={
            "help": "Batch size for each instruction dataloaders per GPU."
        }
    )

    # Loss hyperparameters:
    mlm_loss_weight: float = field(
        default=.5,
        metadata={
            "help":"Weight (alpha) of MLM loss."
        }
    )
    # Loss hyperparameters:
    qa_loss_weight: float = field(
        default=1.0,
        metadata={
            "help":"Weight (alpha) of QA LM loss."
        }
    )
    # Loss hyperparameters:
    retrieval_loss_weight: float = field(
        default=1.0,
        metadata={
            "help":"Weight (alpha) of retrieval loss."
        }
    )
    # Loss hyperparameters:
    caption_loss_weight: float = field(
        default=1.0,
        metadata={
            "help":"Weight (alpha) of caption LM loss."
        }
    )

    caption_loss_rescale_version: int = field(
        default = None,
        metadata = {
            "help": "If specified, scales the loss for captioning based on options provided in constants. Current choice: 0 or None (i.e., if None, doesn't use rescaling)."
        }
    )

    qa_epoch_multiplier: int = field(
        default = 1,
        metadata = {
            "help": "If above 1, repeats QA beyond the number of given epochs"
        }
    )

    retrieval_epoch_multiplier: int = field(
        default = 1,
        metadata = {
            "help": "If above 1, repeats retrieval beyond the number of given epochs"
        }
    )

    caption_epoch_multiplier: int = field(
        default = 1,
        metadata = {
            "help": "If above 1, repeats caption beyond the number of given epochs"
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
    projection_lr: float = field(
        default=1e-3,
        metadata={
            "help":"Learning rate for projection layers in model."
        }
    )
    contrastive_lr: float = field(
        default=1e-4, # TODO: examine if this default LR is useful - can look at SimCLR, other CL papers w adapters
        metadata={
            "help":"Learning rate for temperature parameter and (potentially) projection layers in contrastive head."
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

    # Evaluation:
    eval_on_the_fly: bool = field(
        default=True,
        metadata={
            "help":"If True, evaluates on the fly with validation data."
        }
    )
    eval_steps: int = field(
        default=5000,
        metadata={
            "help":"Number of steps between checkpoint saving & evaluation."
        }
    )
    initial_eval_steps: int = field(
        default=1000,
        metadata={
            "help":"Number of steps between checkpoint saving & evaluation until initial_eval_steps_limit."
        }
    )
    initial_eval_steps_limit: int = field(
        default=5000,
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
    eval_batch_size: bool = field(
        default=16,
        metadata={
            'help':"Batch size to use when generating text embeddings for eval. By default, is set to 2x training batch size."
        }
    )
    # TODO: Implement below if we need efficiency
    eval_max_number: bool = field(
        default = None,
        metadata = {
            "help": "Maximum number of samples from the validation set to run at each evaluation. Set lower for more efficiency. Only applies to QA, captioning."
        }
    )
    eval_retrieval_k: bool = field( # TODO: Could break this across datasets bc different datasets may need different values
        default = 25,
        metadata = {
            "help": "K value used for retrieval evaluation"
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

    # Save:
    output_dir: str = field(
        default=os.path.join(DATA_DIR, "model_outputs/pretrain/"),
        metadata={
            "help":"Path to save model."
        }
    )
    save_steps: int = field( # TODO: Implement
        default=5000,
        metadata={
            "help": "Steps to save model. Controls frequency of saving."
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

    group_name: str = field(
        default = None,
        metadata={
            "help": "Name of distributed run",
        }
    )

    base_name: str = field(
        default = None,
        metadata = {
            "help": "Set if you want to use the find_latest_checkpoint option"
        }
    )

    distributed_wandb_logging: bool = field(
        default = False,
        metadata={
            "help": "If True, logs wandb across all processes, i.e., all nodes, all ranks"
        }
    )

    resume_wandb_id: str = field(
        default=None,
        metadata={
            "help":"WandB id of existing run to resume"
        }
    )

    resume_wandb_id_config: str = field(
        default=None,
        metadata={
            "help": "Path to config file with all run ID's across the distributed wandb logging. Must be json-parseable"
        }
    )

    local_rank: int = field(
        default=-1,
        metadata={
            "help": "For distributed training: local_rank process will distribute corpus across devices. "
        },
    )

    watch_gradient: bool = field(
        default = True,
        metadata = {
            "help": "If true, logs the gradient values with weights and biases."
        }
    )

    gradient_log_frequency: int = field(
        default = 500,
        metadata = {
            "help": "Logs gradients at this frequency; based on steps not global steps."
        }
    )

    use_deepspeed: bool = field(
        default = True,
        metadata = {
            "help": "Use deepspeed training. Set False if debugging."
        }
    )

    importance_sample_datasets: bool = field(
        default = False,
        metadata = {
            "help": "If True, performs importance sampling to balance the datasets."
        }
    )

    # Resume arguments:
    resume_data_args: bool = field(
        default = False,
        metadata = {
            "help": "If True, uses the old data args from the model checkpoint that you're loading"
        }
    )
    resume_model_args: bool = field(
        default = False,
        metadata = {
            "help": "If True, uses the old model args from the model checkpoint that you're loading"
        }
    )
    resume_train_args: bool = field(
        default = False,
        metadata = {
            "help": "If True, uses the old training args from the model checkpoint that you're loading"
        }
    )
    resume_training_progress: bool = field(
        default = True,
        metadata = {
            "help": "If True and resume_from_checkpoint is not None, resumes progress from the checkpoint being loaded. Should set to false if changing the training args."
        }
    )
    force_checkpoint_load_consolidation: bool = field(
        default = False,
        metadata = {
            "help": "If True, upon loading the checkpoint, consolidates through fp32 checkpoint loading. Will not resume deepspeed optimizer args. Use this if checkpoint's world size was different from new training instance's world size."
        }
    )
    def __post_init__(self):
        super().__post_init__()
        replace_custom_dir(self)
        self.warmup_steps = int(self.max_steps * self.warmup_ratio) if self.warmup_steps is None else self.warmup_steps


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

    model_args_variables = [
        "protein_encoder_num_params",
        "protein_encoder_debug",
        "aaseq_encoder_num_params",
        "protein_tokenizer_name",
        "aaseq_tokenizer_name",
        "max_protein_len",
        "max_aaseq_len",
        "long_protein_strategy",
        "long_aaseq_strategy",
        "is_protein_tokenized",
        "is_aaseq_tokenized",
        "protein_pooling_opt",
        "aaseq_pooling_opt",
        "protein_enc_batch_limit",
        "aaseq_enc_batch_limit",
        "text_encoder_fname",
        "text_encoder_debug",
        "text_tokenizer_name",
        "max_text_len",
        "text_pooling_opt",
        "is_go_tokenized",
        "is_pfam_tokenized",
        "ret_token_access",
        "model_splitting",
        "n_model_pieces",
        "use_lora",
        "attention_type",
        "streaming_llm_max_gen_len",
        "decoder_dim",
        "decoder_nlayers",
        "protein_text_combine_strategy",
        "aaseq_text_combine_strategy",
        "use_text_embeddings",
        "go_embeddings_path",
        "pfam_embeddings_path",
        "drugbank_embeddings_path",
        "reactome_embeddings_path",
        "omim_embeddings_path",
        "ec_embeddings_path",
        "use_aaseq_embeddings",
        "use_drug_embeddings",
        "use_protein_struct",
        "protein_struct_dropout",
        "protein_seq_embeddings_path",
        "protein_struct_embeddings_path",
        "protein_embeddings_idmap_path",
        "drug_struct_embeddings_path",
        "domain_embeddings_path",
        "domain_embeddings_idmap_path",
        "mouse_ortholog_embeddings_path",
        "mouse_ortholog_embeddings_idmap_path",
        "freeze_protein_encoder",
        "freeze_text_encoder",
        "freeze_text_embeddings",
        "freeze_aaseq_embeddings",
        "aaseq_lora_alpha",
        "aaseq_lora_r",
        "aaseq_adapter_rank",
        "text_lora_alpha",
        "text_lora_r",
        "text_adapter_rank",
        "cl_method",
        "use_projection_cl",
        "causal_qa",
        "train_retrieval_lm",
        "train_qa_full_lm",
        "num_layers_token_projector",
        "num_layers_shared_projector",
        "num_layers_lm_projector",
        "roll_num",
        "negative_sampling_strategy_retrieval",
        "context_crop_sampling",
        "context_crop_sampling_qa",
        "context_crop_sampling_retrieval",
        "context_crop_sampling_caption",
        "enforce_checkpoint_architecture_strict"
    ]

    data_args_variables = [
        "it_data_config_yml",
        "use_protein_go_dataset",
        "use_domain_go_dataset",
        "use_pfam_dataset",
        "use_omim_dataset",
        "use_disgenet_dataset",
        "use_ec_dataset",
        "use_opentargets_dataset",
        "use_huri_dataset",
        "use_string_dataset",
        "use_clingen_dataset",
        "use_pharmgkb_dataset",
        "use_ctb_dataset",
        "use_reactome_dataset",
        "use_drugbank_dataset",
        "use_mouse_dataset",
        "use_protein_mlm",
        "use_qa",
        "use_retrieval",
        "use_caption",
        "data_dir",
        "go_split_method",
        "pfam_split_method",
        "go_def_col",
        "pfam_def_col",
        "text_variant_type",
        "negative_sampling_strategy_qa",
        "use_only_goa_gos",
        "use_only_protein_go_gos",
        "use_only_goa_proteins",
        "use_only_protein_go_proteins",
        "use_only_ppi_proteins",
        "use_only_protein_protein_proteins",
        "use_only_domain_go_gos",
        "use_only_domain_go_domains",
        "use_only_domain_pfam_pfams",
        "use_only_domain_pfam_domains",
        "num_neg_samples_qa",
        "num_neg_samples_retrieval",
        "go_sims_type",
        "protein_sims_type",
        "domain_sims_type",
        "pfam_sims_type",
        "protein_mlm_probability",
        "protein_mlm_masking_strategy",
        "protein_mlm_probability",
        "protein_mlm_masking_strategy",
        "relation_file",
        "val_split_type",
        "use_old_data",
        "num_instruction_examples",
        "sample_num_instruction_examples",
        "use_entity_compositions",
        "sample_entity_compositions",
        "use_instructions",
        "use_perplexity_filtered_set"
    ]

    train_args_variables = [
        "num_workers",
        "deepspeed_config",
        "protein_mlm_batch_size",
        "qa_batch_size",
        "retrieval_batch_size",
        "caption_batch_size",
        "mlm_loss_weight",
        "qa_loss_weight",
        "retrieval_loss_weight",
        "caption_loss_weight",
        "optimize_memory",
        "optimizer_type",
        "protein_encoder_lr",
        "aaseq_encoder_lr",
        "text_encoder_lr",
        "embedding_lr",
        "projection_lr",
        "contrastive_lr",
        "weight_decay",
        "max_steps",
        "debug",
        "overfit_first_batch",
        "log_interval",
        "eval_on_the_fly",
        "eval_steps",
        "initial_eval_steps",
        "initial_eval_steps_limit",
        "eval_on_first_step",
        "eval_batch_size",
        "eval_max_number",
        "eval_retrieval_k",
        "early_stopping",
        "early_stopping_patience",
        "early_stopping_delta",
        "from_yaml",
        "from_json",
        "output_dir",
        "save_steps",
        "run_name",
        "run_name_suffix",
        "resume_wandb_id",
        "local_rank",
        "watch_gradient",
        "gradient_log_frequency",
        "use_deepspeed",
        "importance_sample_datasets",
        "resume_data_args",
        "resume_model_args",
        "resume_train_args",
        "resume_training_progress"
    ]

    # TODO: fill in below - ignored for now
    # TODO: @Tom, @Owen, add more model args here
    keys_include = set(model_args_variables + data_args_variables + train_args_variables)

    hparams = dict()
    for args in args_tuple:
        if not isinstance(args, dict):
            args_dict = to_dict(args) # Convert to dict if needed
        else:
            args_dict = args
        for key, value in args_dict.items():
            #if key in keys_include:
            hparams[key] = value # Just include all of them

    return hparams

def update_model_args_data_dir(model_args: ModelArgs, prev_data_dir: str):
    if not isinstance(model_args, ModelArgs):
        raise ValueError(f"expected ModelArgs, got: {type(model_args)}")

    if DATA_DIR == prev_data_dir:
        return
    print(f"updating model args DATA_DIR from {prev_data_dir} -> {DATA_DIR}")
    for field, curr_val in asdict(model_args).items():
        if field.endswith("path"):
            if curr_val is None:
                continue
            if curr_val.startswith(prev_data_dir):
                suffix = curr_val.replace(prev_data_dir, "").lstrip("/")
                print(f"updating stale DATA_DIR for model arg: {field}")
                setattr(model_args, field, os.path.join(DATA_DIR, suffix))

def update_data_args_data_dir(data_args: DataArgs):
    if not isinstance(data_args, DataArgs):
        raise ValueError(f"expected DataArgs, got: {type(data_args)}")

    prev_data_dir = data_args.data_dir
    if DATA_DIR == prev_data_dir:
        return
    print(f"updating data args DATA_DIR from {prev_data_dir} -> {DATA_DIR}")
    data_args.data_dir = DATA_DIR

def postprocess_args(train_args, data_args, model_args):

    if model_args.context_crop_sampling: # Override if this is True
        model_args.context_crop_sampling_qa = True
        model_args.context_crop_sampling_retrieval = True
        model_args.context_crop_sampling_caption = True

    data_args.negative_sampling_strategy_retrieval = model_args.negative_sampling_strategy_retrieval
    return train_args, data_args, model_args
