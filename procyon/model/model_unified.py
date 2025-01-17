import os
import json
import random

from itertools import chain
from typing import List

import deepspeed
import torch
import transformers

from esm.data import Alphabet
from torch import nn
from torch.distributed import all_gather as all_gather_no_backprop
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from procyon.data.data_utils import (
    load_aaseq_embeddings,
    load_protein_struct_embeddings,
    load_drug_structure_embeddings,
    DATA_DIR,
)

from procyon.model.contrastive import InfoNCE, InfoNCEInBatch, MaxMarginContrastiveLoss
from procyon.model.pmc_llama import LlamaPostTokenization
from procyon.model.esm import ESM_PLM
from procyon.model.model_utils import create_mlp, check_architecture_args, compute_conflict_matrix, left_pad_tensors

from procyon.training.training_args_IT import ModelArgs, update_model_args_data_dir
from procyon.training.train_utils import unwrap_model, barrier
from procyon.data.constants import DATASET_ID

DEFAULT_PRETRAINED_WEIGHTS_DIR = f'{DATA_DIR}/model_weights/'

SAVE_TRAINING_STATE_FNAME = 'training_state.pt'
SAVE_CONFIG_FNAME = 'procyon_config.json'


def mask_before(full_labels, answer_idx, before_last_answer = False):
    '''
    Takes full_labels that is already masked from
    '''
    answer_found = (full_labels == answer_idx).nonzero()
    if not before_last_answer:
        if torch.any((full_labels == answer_idx).sum(dim=1) > 1):
            raise ValueError('More than one {} token detected in an input'.format(answer_idx))
        found_map = answer_found
    else:
        # Need the last instance of each [ANSWER] token
        # Approach: for each full_labels instance (one sample in the batch), find the maximum index of the answer tokens found in that sample
        found_map = []
        for i in range(full_labels.shape[0]):
            highest_ind = answer_found[answer_found[:,0] == i,1].max().item() # Finds maximum column with answer_idx that was found in row i
            found_map.append((i,highest_ind))
        found_map = torch.tensor(found_map, device = full_labels.device)

    arange_expanded = torch.arange(full_labels.shape[1], device = full_labels.device).unsqueeze(0).repeat(full_labels.shape[0],1)
    ind_expanded = found_map[:,1].unsqueeze(1).repeat(1,full_labels.shape[1])
    mask = ind_expanded >= arange_expanded
    return mask

def find_first_ret_token(ret_indication):
    # Input: (B, L) boolean tensor, indicates where certain tokens are found
    # Get only the first [RET] detected:
    imax = ret_indication.float().argmax(dim=-1)
    if_ret_found = ((ret_indication).sum(dim=-1) > 0)
    ic = torch.cat([torch.arange(ret_indication.shape[0]).unsqueeze(1), imax.unsqueeze(1)])[if_ret_found,:]
    # Index of tensors, size (B~, 2) NOTE: B~ is possibly not B, if no RET token was found

    mask = torch.zeros_like(ret_indication).bool()
    mask[ic[:,0],ic[:,1]] = 1 # Mask in found indexes

    return mask, if_ret_found # Return size same as ret_indication for mask,

def mid_list_insert(a, b, replace_token):
    # From ChatGPT
    index = a.index(replace_token)
    # Create a new list that contains the elements before the index,
    # the new list, and the elements after the index
    result = a[:index] + b + a[index + 1:]
    return result

def multi_replace_tokens(a, b, replace_token, eval = False):
    # Find all occurrences of replace_token in a
    occurrences = (torch.tensor(a) == replace_token).nonzero(as_tuple=True)[0].tolist()

    # Check if the number of occurrences matches the length of b
    if len(occurrences) != len(b):
        raise ValueError("Number of occurrences of replace_token does not match the length of b")

    # Nothing to replace
    if len(occurrences) == 0:
        return a

    # Iterate through occurrences and replace with elements from b
    result = a[:occurrences[0]]
    for i, occurrence in enumerate(occurrences):
        if i == (len(occurrences) - 1) and eval:
            pass # Skip this step if we're evaluating
        else:
            result += b[i]

        if i == (len(occurrences) - 1):
            result += a[(occurrence+1):]
        else:
            result += a[(occurrence+1):occurrences[i+1]]

    return result


class UnifiedProCyon(nn.Module):

    def __init__(self,
            config: ModelArgs,
            pretrained_weights_dir = DEFAULT_PRETRAINED_WEIGHTS_DIR,
            for_pretraining = True,
        ):
        super(UnifiedProCyon, self).__init__()
        self.config = config
        self.pretrained_weights_dir = pretrained_weights_dir

        # Support for models that don't have this variable TODO: change later
        try:
            self.causal_qa = self.config.causal_qa
        except:
            print('Causal QA not found in config, setting to True')
            self.causal_qa = True

        try:
            self.train_qa_full_lm = self.config.train_qa_full_lm
        except:
            print('Setting train qa full LM')
            self.train_qa_full_lm = True

        assert self.causal_qa, 'Non-causal QA currently not working, causal_qa must be set to true'

        if config.text_encoder_fname.lower().startswith('llama'):
            if config.freeze_text_encoder == "lora":
                config.use_lora = True
                use_q_lora = False
            else:
                config.use_lora = False
                use_q_lora = False
            if config.freeze_text_encoder == "qlora":
                config.use_lora = False
                use_q_lora = True

            self.text_encoder = LlamaPostTokenization(
                model_path=f'{DATA_DIR}/model_weights/{config.text_encoder_fname}',
                model_splitting = config.model_splitting,
                n_model_pieces = config.n_model_pieces,
                use_lora = config.use_lora,
                attention_type=config.attention_type,
                max_gen_len = config.streaming_llm_max_gen_len,
                use_q_lora=use_q_lora,
                use_task_spc_lora=config.text_task_spc_lora,
                lora_num = config.text_task_spc_lora_num,
                for_pretraining = for_pretraining,
            )
            if config.text_encoder_debug == True:
                if config.use_lora:
                    self.text_encoder.model.model.model.layers = self.text_encoder.model.model.model.layers[: 2]
                else:
                    self.text_encoder.model.model.layers = self.text_encoder.model.model.layers[: 2]
            self._init_tokenizer()
            self.text_encoder.model.resize_token_embeddings(len(self.tokenizer) - 1)

            self.text_embed_dim = self.text_encoder.model.config.hidden_size
            self.input_embeddings = self.text_encoder.model.get_input_embeddings()
            #self.text_encoder.model._set_gradient_checkpointing(self.text_encoder.model.model, True)

            if config.freeze_text_encoder == "all":
                for pn, p in self.text_encoder.model.named_parameters():
                    if "lm_head" in pn:
                        p.requires_grad_(True)
                    else:
                        p.requires_grad_(False)
            elif config.freeze_text_encoder == "lora":
                for pn, p in self.text_encoder.model.named_parameters():
                    if not 'lora' in pn:
                        p.requires_grad_(False)
            elif config.freeze_text_encoder != "qlora":
                for pn, p in self.text_encoder.model.named_parameters():
                    p.requires_grad_(True)
        else:
            raise ValueError(f'Unrecognized text encoder: {config.text_encoder_fname}')

        # Protein encoder definition ----------------------------------- :
        if config.use_aaseq_embeddings:
            self.protein_seq_encoder = None

            if not hasattr(config, 'mouse_ortholog_embeddings_path'):
                config.mouse_ortholog_embeddings_path = f'{DATA_DIR}/generated_data/node_embeddings/mouse_ortholog/mouse_ortholog_esm2-3b_max.pt'
                config.mouse_ortholog_embeddings_idmap_path = f'{DATA_DIR}/generated_data/node_embeddings/mouse_ortholog/mouse_ortholog_esm2-3b_max.pkl'
            if not hasattr(config, 'protein_struct_embeddings_path'):
                config.protein_struct_embeddings_path = os.path.join(DATA_DIR, "generated_data/node_embeddings/protein/protein_gearnet.pt")
            config.protein_embeddings_idmap_path = f'{DATA_DIR}/generated_data/node_embeddings/protein/protein_esm2-3b_mean.pkl'
            config.domain_embeddings_idmap_path = f'{DATA_DIR}/generated_data/node_embeddings/domain/domain_esm2-3b_mean.pkl'

            protein_embeddings = load_aaseq_embeddings(config.protein_seq_embeddings_path, config.protein_embeddings_idmap_path, aaseq_type='protein')
            domain_embeddings = load_aaseq_embeddings(config.domain_embeddings_path, config.domain_embeddings_idmap_path, aaseq_type='domain')

            self.protein_seq_embeddings = nn.Embedding.from_pretrained(protein_embeddings, freeze=config.freeze_aaseq_embeddings)
            self.domain_embeddings = nn.Embedding.from_pretrained(domain_embeddings, freeze=config.freeze_aaseq_embeddings)

            if config.peptide_embeddings_path is not None:
                assert config.peptide_embeddings_idmap_path is not None
                peptide_embeddings = load_aaseq_embeddings(config.peptide_embeddings_path, config.peptide_embeddings_idmap_path, aaseq_type='peptide')
                self.peptide_embeddings = nn.Embedding.from_pretrained(peptide_embeddings, freeze=config.freeze_aaseq_embeddings)

            self.protein_embed_dim = protein_embeddings.shape[1]

            assert domain_embeddings.shape[1] == self.protein_embed_dim

        else:
            protein_tokenizer = Alphabet.from_architecture(config.protein_tokenizer_name)
            self.protein_seq_encoder = ESM_PLM(
                pretrained_weights_dir = pretrained_weights_dir,
                num_params = config.protein_encoder_num_params,
                pooling_method = config.protein_pooling_opt,
                padding_idx = protein_tokenizer.padding_idx,
                eos_idx = protein_tokenizer.eos_idx,
                long_protein_strategy = config.long_protein_strategy,
                max_protein_len = config.max_protein_len,
                max_batch_forward_pass = config.protein_enc_batch_limit,
                use_lora=(config.freeze_protein_encoder == 'lora'),
                lora_alpha=config.aaseq_lora_alpha,
                lora_r=config.aaseq_lora_r,
                use_adapter=(config.freeze_protein_encoder == 'adapter'),
                adapter_rank=config.aaseq_adapter_rank,
                use_prefix=(config.freeze_protein_encoder == 'prefix'),
                lora_parameters = config.protein_lora_parameters,
                use_task_spc_lora = config.protein_task_spc_lora,
                lora_num=config.protein_task_spc_lora_num,
                protein_pooling_correction_option=config.protein_pooling_correction_option,
            )
            if config.protein_encoder_debug == True:
                self.protein_seq_encoder.model.layers = self.protein_seq_encoder.model.layers[: 1]
                self.protein_seq_encoder.repr_layer = 0

            self.protein_embed_dim = self.protein_seq_encoder.embedding_size

            if config.freeze_protein_encoder == "all":
                for pn, p in self.protein_seq_encoder.model.named_parameters():
                    print(pn, p.requires_grad)
                    if "lm_head" in pn:
                        p.requires_grad_(True)
                    else:
                        p.requires_grad_(False)
            elif config.freeze_protein_encoder == "lora":
                for pn, p in self.protein_seq_encoder.model.named_parameters():
                    if not 'lora' in pn:
                        p.requires_grad_(False)
                    print(pn, p.requires_grad)
            elif config.freeze_protein_encoder != "qlora":
                for pn, p in self.protein_seq_encoder.model.named_parameters():
                    p.requires_grad_(True)
                    print(pn, p.requires_grad)

        self.token_projectors = nn.ModuleDict({
            "aaseq": create_mlp(
                n_layers = self.config.num_layers_token_projector,
                in_features = self.protein_embed_dim,
                out_features = self.input_embeddings.weight.shape[-1],
                hidden_features = self.config.hidden_size_token_projector,
            )
        })

        # Get structure embeddings if we're using them:
        if self.config.use_protein_struct:
            protein_struct_embeddings = load_protein_struct_embeddings(config.protein_struct_embeddings_path)
            self.protein_struct_embeddings = nn.Embedding.from_pretrained(protein_struct_embeddings, freeze=True)
            self.protein_struct_embed_dim = protein_struct_embeddings.shape[1]

            self.token_projectors.update({"prot_structure": create_mlp(
                n_layers = self.config.num_layers_token_projector,
                in_features = self.protein_struct_embed_dim,
                out_features = self.input_embeddings.weight.shape[-1],
                hidden_features = self.config.hidden_size_token_projector,
            )})

        else:
            self.protein_struct_embeddings = None

        if self.config.use_drug_embeddings:
            drug_embeddings = load_drug_structure_embeddings(config.drug_struct_embeddings_path)
            self.drug_structure_embeddings = nn.Embedding.from_pretrained(drug_embeddings, freeze=True)
            self.drug_embed_dim = drug_embeddings.shape[1]

            self.token_projectors.update({"drug": create_mlp(
                n_layers = self.config.num_layers_token_projector,
                in_features = self.drug_embed_dim,
                out_features = self.input_embeddings.weight.shape[-1],
                hidden_features = self.config.hidden_size_token_projector
            )})
        else:
            self.drug_structure_embeddings = None


        # Projector definitions: --------------------------------------------------:
        # Static design choice (can change later): use protein encoder dimension as shared dimension size
        #   - This is directly following FROMAGe architecture

        # Terminology:
        #   1. token_projectors: Projects from modality-specific encoders to token space
        #   2. shared_projectors: Projects from modality-specific encoders to shared latent space
        #   3. lm_projectors: Projects from the text embedding output to shared latent space

        self.aaseq_shared_projector = create_mlp(
            n_layers = self.config.num_layers_shared_projector,
            in_features = self.protein_embed_dim,
            out_features = self.protein_embed_dim,
            hidden_features = self.config.hidden_size_shared_projector,
        )

        self.aaseq_lm_projector = create_mlp(
            n_layers = self.config.num_layers_lm_projector,
            in_features = self.text_embed_dim,
            out_features = self.protein_embed_dim,
            hidden_features = self.config.hidden_size_lm_projector
        )

        # Setup contrastive learning method:
        assert self.config.negative_sampling_strategy_retrieval == 'in_batch'
        if self.config.cl_method.lower() == 'infonce':
            if self.config.negative_sampling_strategy_retrieval == 'in_batch':
                self.contrastive_head = InfoNCEInBatch(
                    input_embed_dim = self.protein_embed_dim,
                    use_projection = self.config.use_projection_cl,
                    all_gather_version = self.config.contrastive_global,
                )
            else:
                self.contrastive_head = InfoNCE(input_embed_dim = self.protein_embed_dim, use_projection = self.config.use_projection_cl)
        elif self.config.cl_method.lower() == 'maxmargin':
            self.contrastive_head = MaxMarginContrastiveLoss(
                protein_embed_dim = self.protein_embed_dim,
                text_embed_dim = self.text_embed_dim,
                margin = 0.0, use_projection = self.config.use_projection_cl)
        else:
            raise NotImplementedError

        if "llama-3" in self.config.text_encoder_fname.lower():
            self.yes_token = self.tokenizer.encode(" yes", add_special_tokens=False)[0]
            self.no_token = self.tokenizer.encode(" no", add_special_tokens=False)[0]
        else:
            self.yes_token = self.tokenizer.encode('yes', add_special_tokens=False)[0]
            self.no_token = self.tokenizer.encode('no', add_special_tokens=False)[0]

        self.context_crop_sampling = self.config.context_crop_sampling
        self.struct_dropout_prob = self.config.protein_struct_dropout

    def _preprocessing(self,
            inputs,
            aaseq_type = 'protein',
            exclude_protein_structure = False,
            crop_off = False,
            no_pad = False,
            retrieval=False,
            left_pad=False,
        ):

        # Pass all sequence tokens to protein encoder
        aaseq_embeddings = None
        if inputs["data"]["seq"] is not None:
            if self.config.use_aaseq_embeddings:
                if aaseq_type == 'protein':
                    aaseq_ret_embeddings = self.protein_seq_embeddings(inputs["data"]["seq"])
                    aaseq_token_embeddings = self.protein_seq_embeddings(inputs["data"]["seq"])
                elif aaseq_type == 'domain':
                    aaseq_ret_embeddings = self.domain_embeddings(inputs["data"]["seq"])
                    aaseq_token_embeddings = self.domain_embeddings(inputs["data"]["seq"])
                elif aaseq_type == 'peptide':
                    aaseq_ret_embeddings = self.peptide_embeddings(inputs["data"]["seq"])
                    aaseq_token_embeddings = self.peptide_embeddings(inputs["data"]["seq"])
            else:
                if self.config.protein_task_spc_lora and (self.config.lora_specific_style == 'space_specific'):
                    # Breakdown by input and output spaces:
                    if retrieval:
                        # Flag to consider proteins in the retrieval space
                        self.protein_seq_encoder.set_prot_lora_group(0) # Zero marks the token-space LoRA
                        aaseq_token_embeddings, _ = activation_checkpoint(self.protein_seq_encoder, inputs["data"]["seq"], True)
                        self.protein_seq_encoder.set_prot_lora_group(1) # One marks the retrieval-space LoRA
                        aaseq_ret_embeddings, _ = activation_checkpoint(self.protein_seq_encoder, inputs["data"]["seq"], True)
                    else:
                        # Everything is token space
                        self.protein_seq_encoder.set_prot_lora_group(0) # Zero marks the token-space LoRA
                        aaseq_embeddings, _ = self.protein_seq_encoder(inputs["data"]["seq"], aggregate = True)
                        aaseq_token_embeddings = aaseq_embeddings
                        aaseq_ret_embeddings = aaseq_embeddings
                else:
                    aaseq_embeddings, _ = self.protein_seq_encoder(inputs["data"]["seq"], aggregate = True)
                    aaseq_token_embeddings = aaseq_embeddings
                    aaseq_ret_embeddings = aaseq_embeddings
        else:
            aaseq_token_embeddings, aaseq_ret_embeddings = None, None

        # Combine seq's into text_input and text_target:
        if inputs["input"]["seq"] is not None:
            # Flattens list of lists into one list for indexing
            #   Below line should work for arbitrarily-sized lists - need this for PPI construction in the future
            full_index = list(chain.from_iterable(inputs["input"]["seq"]))
            pz_inputs = aaseq_token_embeddings[full_index]

            # Run the protein embeddings through the projection:
            protein_soft_tokens = self.token_projectors['aaseq'](pz_inputs)
        else:
            protein_soft_tokens = None

        if self.config.use_drug_embeddings and (inputs["data"]["drug"] is not None):
            # Replace in drug embeddings - treat like protein embeddings not like protein structure embeddings
            full_index = list(chain.from_iterable(inputs["input"]["drug"]))
            drug_z = self.drug_structure_embeddings(inputs["data"]["drug"])[full_index]
            drug_soft_tokens = self.token_projectors["drug"](drug_z)
        else:
            drug_soft_tokens = None

        # Assume inputs["input"]["text"] is never None
        # Iterates over lists of lists
        text_inputs = [[inputs["data"]["text"][i] for i in inp_list] for inp_list in inputs["input"]["text"]]

        # If getting structure, go through each string and replace "<|protein|>" with "<|protein|> <|struct|>" at 1-dropout probability
        instruction_list = inputs['instructions']
        protein_struct_tokens = []

        # REMINDER: structure only goes in input, so if no sequence in input, don't insert structure tokens
        if (not exclude_protein_structure) and \
            (self.config.use_protein_struct) and \
            (inputs["input"]["seq"]):

            # Method 1:
            # Mask for dropout by sample:
            # If zero, then dropout that sample
            include_mask = torch.bernoulli(torch.full((len(instruction_list),), 1-self.struct_dropout_prob))

            all_row_indices = []
            for i in include_mask.nonzero(as_tuple=True)[0].tolist():
                instruction_list[i] = instruction_list[i].replace("<|protein|>", "<|protein|> <|struct|>")
                row_index = torch.cat([inputs["data"]["seq_idx"][j].unsqueeze(0) for j in inputs["input"]["seq"][i]])
                all_row_indices.append(row_index)

            all_row_indices = torch.stack(all_row_indices, dim = 0)
            ari_unique, ari_inverse = all_row_indices.unique(return_inverse=True)

            # Ideally only want one call to these modules
            if (aaseq_type == "protein"):
                struct_z = self.protein_struct_embeddings(ari_unique)
            else:
                struct_z = torch.zeros_like(self.protein_struct_embeddings(torch.arange(ari_unique.shape[0], device=ari_unique.device)))
            struct_token_z = self.token_projectors["prot_structure"](struct_z)

            # Re-organize unique embeddings back to original positions
            token_z_expand = struct_token_z[ari_inverse]

            # Account for dropout in struct tokens by appending empty list in missing spots
            protein_struct_tokens = []
            for i, val in enumerate(include_mask):
                if val:
                    protein_struct_tokens.append(token_z_expand[i,...])
                else:
                    protein_struct_tokens.append([])

        # Create input embeddings by replacing protein_soft_tokens in the correct spots
        input_ids, attn_masks = self._prepare_text_inputs_and_tokenize(
            instruction_list,
            text_inputs,
            crop_off=crop_off,
            retrieval=retrieval,
            no_pad=no_pad,
            left_pad=left_pad,
        )
        input_ids = input_ids.to(self.input_embeddings.weight.device)
        attn_masks = attn_masks.to(self.input_embeddings.weight.device)
        # Get input embeds:
        input_embeds, ret_output_indices = self._prepare_input_embeddings(
            input_ids,
            protein_soft_tokens=protein_soft_tokens,
            protein_struct_tokens=protein_struct_tokens,
            drug_soft_tokens=drug_soft_tokens,
        )

        return input_embeds, input_ids, attn_masks, ret_output_indices, aaseq_token_embeddings, aaseq_ret_embeddings

    def forward(self,
            inputs,
            return_mlm: bool = False,
            retrieval: bool = False,
            get_full_labels: bool = False,
            aaseq_type: str = 'protein',
            exclude_protein_structure = False,
            crop_off = False,
            output_attentions = False,
        ):
        '''
        Performs a forward pass specifically with the output from the Trainer/collators

        Returns:
            out: dict with keys:
                outputs: CausalLMOutputs object from Huggingface
                text_toks: equivalent to input_ids; text that is input to the model
                full_labels:
                contrastive_out:
                contrastive_loss:
        '''

        if return_mlm:
            # Doesn't access the text encoder at all
            tok_src = inputs["data"]["seq"] # TODO: May need to change
            outputs, logits = self.protein_seq_encoder(tok_src, aggregate=False)
            return {"mlm": logits} # No pooling b/c MLM

        # Ignores protein structure upon retrieval (bc retrieved proteins are not structure-based)
        ignore_protein_struct = (not (inputs["target"]["seq"] is None)) and self.training and not (exclude_protein_structure) # Detects if retrieval and training - eval might have different rules

        input_embeds, input_ids, attn_masks, ret_output_indices, protein_token_embeddings, protein_ret_embeddings = self._preprocessing(
            inputs,
            aaseq_type = aaseq_type,
            crop_off = crop_off,
            retrieval=retrieval,
            exclude_protein_structure = ignore_protein_struct)

        full_labels = None
        if not retrieval:
            full_labels = input_ids.clone()
            # Mask-out anything before the answer token
            # Don't compute on pads, replacement, drug, structure, or retrieval indices:
            all_masks = (full_labels == self.tokenizer.pad_token_id) \
                | (full_labels == self.prot_replacement_idx) \
                | (full_labels == self.prot_retrieval_idx) \
                | (full_labels == self.drug_idx) \
                | (full_labels == self.struct_idx)
            # Cannot mask out sep bc we need this for ending the model output: | (full_labels == self.tokenizer.sep_token_id)
            # Temporary hack to get around llama appending sep to the end: (can we make this more general?)
            if self.use_llama_tokenizer:
                all_masks[:,-1] = True # Must mask last bc it appends a sep token
            if not self.train_qa_full_lm:
                mbefore_mask = mask_before(full_labels, self.answer_idx, before_last_answer = True)
                all_masks |= mbefore_mask
            full_labels = torch.where(all_masks, -100, full_labels) # Already on device

        outputs = self.text_encoder(
            input_embeds = input_embeds,
            attn_masks = attn_masks,
            full_labels = full_labels,
            output_attentions = output_attentions,
        )

        # Construct output dictionary:
        out_dict = {
            'outputs': outputs,
            'text_toks': input_ids,
            'full_labels': full_labels if get_full_labels else None,
            'contrastive_out': None,
            'contrastive_loss': None,
        }

        if retrieval:
            contrastive_out = {"positive": {}, "negative": {}}

            # Now proceed to process text
            if self.config.ret_token_access == 'all':
                # FROMAGe sums across all the hidden states, which is different than they claim in the paper
                #   - See https://github.com/kohjingyu/fromage/blob/92c6d6f6ea9cea38f0b0a12bcdb0cf3915d0e774/fromage/models.py#L310
                pooled_hidden_states = torch.stack(outputs.hidden_states, dim = -1).sum(dim=-1) # Pool hidden states -> (B, L, d)
            elif self.config.ret_token_access == 'last':
                pooled_hidden_states = outputs.hidden_states[-1]
            else:
                raise NotImplementedError("Invalid option {} for ret_token_access".format(self.config.ret_token_access))

            # Index the retrieved proteins via the original placement of the [PROT] token
            # Use hidden layer output in index of ret token from input
            extracted_ret = pooled_hidden_states[ret_output_indices]
            # We know there is one ret per input, so this operation holds, even though it flattens along the two dimensions of the boolean mask

            # Index the hidden states by above
            # Collapses first two dimensions, preserves last (perfect for ret_to_shared)
            shared_lm_output = self.aaseq_lm_projector(extracted_ret)

            if inputs["target"]["text"] is None:
                contrastive_out["positive"]["text"] = shared_lm_output
            else:
                contrastive_out["positive"]["text"] = shared_lm_output[inputs["target"]["text"]["positive"]]

                if (inputs["target"]["text"]["negative"] is not None):
                    raise NotImplementedError

            if inputs["target"]["seq"] is not None: # Have the option to exclude this if you don't need to process targets
                # Extract target protein embeddings,
                shared_plm_output = self.aaseq_shared_projector(protein_ret_embeddings)
                contrastive_out["positive"]["sequence"] = shared_plm_output[inputs["target"]["seq"]["positive"]]

                if (inputs['target']["seq"]["negative"] is not None):
                    raise NotImplementedError

                conflict_mat = None

                if self.config.filter_negatives_by_id_contrastive and (self.training):

                    # Order ID's by appearance in contrastive embeddings
                    if (inputs["target"]["text"] is None):
                        if any(inputs["input"]["text"]):
                            text_inds_local = [row[-1] for row in inputs["input"]["text"]]
                            text_ids = torch.LongTensor([inputs["data"]["text_idx"][i] for i in text_inds_local]).to(shared_plm_output.device)
                        else:
                            # "text_ids" becomes equivalent to input sequence in PPI datasets
                            # Clever trick: set to (-1 - N) because this will never overlap with text IDs (which are [0,N], so use -1 - N to make sure that upper bound is -1)
                            input_prot_ids_local = [row[-1] for row in inputs["input"]["seq"]]
                            text_ids = torch.LongTensor([(-1 - inputs["data"]["seq_idx"][i]) for i in input_prot_ids_local]).to(shared_plm_output.device) # Is a PPI dataset
                        # This batch size will be equivalent across ranks
                    else:
                        raise NotImplementedError

                    prot_ids = torch.LongTensor([inputs["data"]["seq_idx"][i] for i in inputs["target"]["seq"]["positive"]]).to(shared_plm_output.device)
                    # This batch size above will be equivalent across ranks

                    # All-gather text_ids
                    all_text_ids = None
                    if self.config.contrastive_global: # Gather across ranks
                        # OK to use this in conjunction with a separate all_gather call since they remain sorted by rank
                        # Text ID:
                        barrier()
                        WORLDSIZE = torch.distributed.get_world_size()
                        text_ids_list = [torch.empty_like(text_ids) for _ in range(WORLDSIZE)]
                        all_gather_no_backprop(text_ids_list, text_ids)
                        text_ids = torch.cat(text_ids_list, dim=0)

                        # Prot ID:
                        barrier()
                        prot_ids_list = [torch.empty_like(prot_ids) for _ in range(WORLDSIZE)]
                        all_gather_no_backprop(prot_ids_list, prot_ids)
                        prot_ids = torch.cat(prot_ids_list, dim=0)

                    # Compute and gather aaseq types:
                    if aaseq_type == 'protein': # Refers to the input type of aaseq
                        aaseq_indicator = torch.zeros_like(prot_ids)
                    elif aaseq_type == 'domain':
                        aaseq_indicator = torch.ones_like(prot_ids)
                    elif aaseq_type == 'peptide': # Shouldn't interact with others
                        aaseq_indicator = torch.full_like(prot_ids, 2)

                    # Compute aaseq conflict:
                    aa_r0 = aaseq_indicator.repeat(aaseq_indicator.shape[0],1)
                    aa_r1 = aaseq_indicator.unsqueeze(1).repeat(1,aaseq_indicator.shape[0])
                    aaseq_overlap = (aa_r0 == aa_r1) # Goes to False where the amino acid sequence type does not match

                    # Compute conflict matrices:
                    text_conflict = compute_conflict_matrix(text_ids, prot_ids)

                    prot_conflict = compute_conflict_matrix(prot_ids, text_ids)
                    # Only consider conflicts of IDs across types of amino acid sequences (domain IDs and protein IDs could be mismatched)
                    prot_conflict = aaseq_overlap & prot_conflict

                    # Only consider conflicts if they occur in the same dataset wrt text ID
                    if "dataset_id" in inputs.keys():
                        # Compute dataset ID matrices:
                        dset_ids = inputs["dataset_id"].to(shared_plm_output.device)

                        if self.config.contrastive_global:
                            WORLDSIZE = torch.distributed.get_world_size()
                            dset_ids_list = [torch.empty_like(dset_ids) for _ in range(WORLDSIZE)]
                            dset_ids = torch.cat(dset_ids_list, dim=0)

                        #print("Dset ids gathered")

                        d_r0 = dset_ids.repeat(dset_ids.shape[0],1)
                        d_r1 = dset_ids.unsqueeze(1).repeat(1,dset_ids.shape[0])
                        dset_overlap = (d_r0 == d_r1)

                        # Compute input protein overlap if we have a protein-protein dataset
                        # Make adjustment in text for protein_protein datasets:
                        # Compute dset ID presence for STRING:
                        ppi_dset = (dset_ids == DATASET_ID["protein"])
                        ppi_dset_r0 = ppi_dset.repeat(ppi_dset.shape[0],1)
                        ppi_dset_r1 = ppi_dset.unsqueeze(1).repeat(1,ppi_dset.shape[0])
                        ppi_dset_matrix = (ppi_dset_r0 == ppi_dset_r1)

                        # Combine with text conflict matrix
                        text_conflict = dset_overlap & text_conflict

                        text_conflict[ppi_dset_matrix] = False # Set to False - there can be no text conflict in PPI dataset overlaps bc there is no text

                    # Compute two-sided conflict matrix via OR
                    conflict_mat = ~(text_conflict | prot_conflict)

                    conflict_mat.requires_grad = False

                # Only need to compute if sequence targets are provided
                # Text part in contrastive_out will already be filled in
                if self.training:
                    contrastive_loss = self.contrastive_head(contrastive_out, negatives_mask = conflict_mat)
                else:
                    contrastive_loss = -999.0 # Very low number to indicate that this is not being ran outside of training

                out_dict['contrastive_loss'] = contrastive_loss # Fill in the None, if not used will be None

            # WARNING: We do not project via the contrastive head for the embeddings

            out_dict['contrastive_out'] = contrastive_out

        return out_dict

    @torch.no_grad()
    def _generate_beam_search(
        self,
        input_embeds,
        attn_mask,
        max_len=64,
        beam_size=5,
        beam_group_size=5,
        diversity_penalty=0.8,
    ):
        """Beam search for text generation.

        Implements diverse beam search as described in "Diverse Beam Search: Decoding Diverse Solutions
        from Neural Sequence Models" (https://arxiv.org/abs/1610.02424). We use the Hamming distance
        diversity function here due to its simplicity and authors' findings that it generates better
        oracle metrics (CIDEr) compared to more complex diversity functions.

        Note that default settings run vanilla beam search, decrease `beam_group_size` to run diverse
        beam search.

        Args:
          - input_embeds: input embeddings as a tensor of (num_inputs X seq_len X embedding_dim)
          - max_len: maximum generation length
          - beam_size: number of beam search hypotheses to maintain per input
          - beam_group_size: used for controlling diverse beam search. Size of sub-groups within
                             each input's beam search that are allowed to extend without diversity
                             penalty. Must evenly divide `beam_size`. Setting this to `beam_size`
                             is equivalent to vanilla beam search, and setting this to 1 results in
                             maximum diversity but limited exploration along a single beam.
          - diversity_penalty: weight given to the Hamming distance penalty term applied when
                               calculating beam hypotheses log-likelihoods across groups.

        Returns three tensors:
        - out: tensor of output token IDs (num_inputs X beam_size X max_len)
        - current_log_probs: tensor of total log-probability of generated sequences (num_inputs X beam_size)
        - output_logits: tensor of logits per position, per output (num_inputs X beam_size X max_len X vocab_size)
        """
        device = input_embeds.device
        orig_batch_size = input_embeds.shape[0]
        beam_batch_size = orig_batch_size * beam_size
        vocab_size = self.text_encoder.model.vocab_size

        if beam_size % beam_group_size != 0:
            raise ValueError("beam_group_size must evenly divide beam_size, got: "
                             f"{beam_size} % {beam_group_size} != 0")
        groups_per_input = beam_size // beam_group_size

        # General approach here is to keep track of beam candidates as a single flattened dim of size
        # num_inputs * beam_size, where the rows from [i:i+beam_size] correspond to the beam candidates
        # for input i.
        embeds_repeated = torch.repeat_interleave(input_embeds, repeats=beam_size, dim=0)
        attn_mask_repeated = torch.repeat_interleave(attn_mask, repeats=beam_size, dim=0)

        current_log_probs = torch.zeros((beam_batch_size,), device=device)
        out = torch.zeros(beam_batch_size, max_len, dtype=torch.int64, device=device)

        past_key_values = None
        output_logits = None
        sm = torch.nn.LogSoftmax(dim=-1)
        for i in range(max_len):
            if i == 0:
                output = self.text_encoder(
                    input_embeds=embeds_repeated,
                    attn_masks=attn_mask_repeated,
                    use_cache=True,
                    past_key_values=None,
                )
            else:
                output = self.text_encoder(input_ids=out[:, i-1].unsqueeze(-1), use_cache=True, past_key_values=past_key_values)
            logits = output.logits[:,-1,:]
            past_key_values = output.past_key_values

            iter_logits = logits.detach().clone().cpu().unsqueeze(1)
            if output_logits is None:
                output_logits = iter_logits
            else:
                output_logits = torch.cat(
                    [output_logits, iter_logits],
                    dim=1,
                )

            log_probs = sm(logits) + current_log_probs[:, None]

            for input_idx in range(orig_batch_size):
                beam_start = input_idx * beam_size

                for group_idx in range(groups_per_input):
                    if i == 0:
                        # If this is the first iteration, all beams are currently
                        # the same, so only want to take top k from a single beam
                        # otherwise we'll just duplicate the same top-1 option from
                        # all the identical beams.
                        check_end_inc = 1
                    else:
                        check_end_inc = beam_group_size

                    group_start = beam_start + (group_idx * beam_group_size)
                    group_end = group_start + beam_group_size
                    check_end = group_start + check_end_inc

                    # Collect token probabilities and ids across all candidates for
                    # this input group.
                    log_probs_for_group = log_probs[group_start:check_end]

                    # Apply diversity penalty based on tokens selected for previous
                    # groups from the same input.
                    if group_idx != 0:
                        previous_group_tokens = out[beam_start:group_start, i]
                        token_frequency = (torch.bincount(
                                               previous_group_tokens,
                                               minlength=vocab_size)
                                          .to(device))
                        log_probs_for_group -= diversity_penalty * token_frequency

                    # Get top-k across all candidates for this group.
                    top_candidate_likelihoods, top_candidate_idxs = log_probs_for_group.ravel().topk(beam_group_size)
                    selected_tokens = top_candidate_idxs % vocab_size

                    # Map back to which original candidate for this input we're extending.
                    orig_candidate_idxs = (top_candidate_idxs // vocab_size) + group_start

                    # Reorder current outputs to reflect selected beams for next iteration. Add in
                    # selected tokens for each beam.
                    out[group_start:group_end] = out[orig_candidate_idxs]
                    out[torch.arange(group_start, group_end), i] = selected_tokens
                    current_log_probs[group_start:group_end] = top_candidate_likelihoods
                    output_logits[group_start:group_end] = output_logits[orig_candidate_idxs.cpu()]

                    # Also need to reorder the past key-value caches.
                    for attn_idx in range(len(past_key_values)):
                        past_key_values[attn_idx][0][group_start:group_end] = past_key_values[attn_idx][0][orig_candidate_idxs]
                        past_key_values[attn_idx][1][group_start:group_end] = past_key_values[attn_idx][1][orig_candidate_idxs]
            if torch.all((out == self.tokenizer.eos_token_id).any(dim=1)).item():
                print(f"all beams reached EOS at iter {i}")
                break

        # Go from flat dim across all beams to nested dims across inputs X num_beams
        out = out.detach().cpu().unflatten(0, (orig_batch_size, beam_size))
        current_log_probs = current_log_probs.detach().cpu().unflatten(0, (orig_batch_size, beam_size))
        output_logits = output_logits.unflatten(0, (orig_batch_size, beam_size))

        return out.detach().cpu(), current_log_probs.detach().cpu(), output_logits

    def _get_nucleus_mask(
        self,
        probs,
        nucleus_prob,
    ):
        remove_prob = 1-nucleus_prob
        sorted_vals, indices = probs.sort(dim=-1, descending=False)
        keep_vals = (sorted_vals.cumsum(dim=-1) >= remove_prob)
        keep_idxs = keep_vals.nonzero(as_tuple=True)
        keep_token_ids = indices[keep_idxs]

        mask = torch.zeros_like(probs)
        mask[keep_idxs[0], keep_token_ids] = 1

        return mask

    @torch.no_grad()
    def _generate_sampling(
        self,
        input_embeds,
        attn_masks,
        max_len=64,
        num_text_per_instance=1,
        temperature=1.0,
        greedy=False,
        nucleus_prob=None,
    ):
        assert nucleus_prob is None or (nucleus_prob > 0 and nucleus_prob < 1)
        num_inputs = len(input_embeds)
        out_list, log_probs_list, output_logits_list = [], [], []
        log_sm = torch.nn.LogSoftmax(dim=-1)

        for _ in range(num_text_per_instance):

            out = None
            past_key_values = None
            output_logits = []
            total_log_prob = torch.zeros(num_inputs, device=input_embeds.device)

            for i in range(max_len):
                if i == 0:
                    output = self.text_encoder(input_embeds=input_embeds, attn_masks=attn_masks, use_cache=True, past_key_values=None)
                else:
                    output = self.text_encoder(input_ids = out[:,-1:], use_cache=True, past_key_values=past_key_values)

                past_key_values = output.past_key_values

                logits = output.logits[:,-1,:]
                output_logits.append(logits.detach().clone().cpu())
                log_probs = log_sm(logits)

                if greedy:
                    next_token = torch.argmax(logits, keepdim=True, dim=-1).long().to(input_embeds.device)
                else:
                    # Logits enter at [N,vocab_size] shape
                    if nucleus_prob is not None:
                        probs = logits.softmax(dim=-1)
                        probs *= self._get_nucleus_mask(probs, nucleus_prob)
                    else:
                       probs =  (logits / temperature).softmax(dim=-1)

                    next_token = torch.multinomial(probs, 1) # (N,1)
                total_log_prob += log_probs[torch.arange(num_inputs), next_token.squeeze()]

                if out is not None:
                    out = torch.cat([out, next_token], dim = -1)
                else:
                    out = next_token

            out_list.append(out)
            output_logits_list.append(torch.stack(output_logits, 1))
            log_probs_list.append(total_log_prob.detach().clone().cpu())

        out_tokens = torch.stack(out_list, dim=1).detach().cpu()
        out_logits = torch.stack(output_logits_list, dim=1)
        # texts_per_input X num_inputs -> num_inputs X num_texts_per_input
        log_probs = torch.stack(log_probs_list).T
        return out_tokens, log_probs, out_logits

    @torch.no_grad()
    def generate(
        self,
        inputs,
        max_len = 64,
        aaseq_type = 'protein',
        method="sampling",
        temperature = 1.0,
        greedy = False,
        num_text_per_instance = 1,
        return_all_internals = False,
        beam_size = 5,
        beam_group_size = 5,
        diversity_penalty = 0.8,
        exclude_protein_structure = False,
        nucleus_prob = 0.9,
        truncate_on_eos = True,
    ):
        '''
       Generate text from a trained model via greedy decoding, sampling, or beam search.

       For description of beam search arguments, see docstring of `_generate_beam_search` above.
        '''

        assert method in ["sampling", "temperature", "greedy", "beam", "nucleus"]
        if method == "beam":
            num_text_per_instance = beam_size
            if beam_group_size == 1:
                print("WARNING: received beam_group_size = 1, note that a group size of 1 corresponds "
                      "to running `beam_size` greedy searches in parallel with a diversity penalty (i.e. "
                      "losing the look-ahead benefit of beam search)")
        elif method == "greedy":
            greedy = True
        elif method in ["sampling", "nucleus"]:
            temperature = 1
        if temperature < 1e-8: # i.e. == 0
            greedy = True

        self.text_encoder.eval()
        if self.protein_seq_encoder is not None:
            self.protein_seq_encoder.eval()
        (input_embeds,
         input_ids,
         attn_masks,
         ret_output_indices,
         protein_token_embeddings,
         protein_ret_embeddings) = self._preprocessing(
            inputs,
            aaseq_type=aaseq_type,
            crop_off = True,
            no_pad=True,
            exclude_protein_structure=exclude_protein_structure,
            left_pad=True,
        )

        # Get input ID's for whole instruction:
        whole_instructions = self.tokenizer.batch_decode(input_ids)
        # Ground truth text:
        if inputs["target"]["text"] is not None:
            gt_text = [inputs["data"]["text"][i] for i in inputs["target"]["text"]]
        else:
            gt_text = None

        batch_size = input_embeds.shape[0]

        if method == "beam":
            output_tokens, log_probs, output_logits = self._generate_beam_search(
                input_embeds,
                attn_masks,
                max_len=max_len,
                beam_size=beam_size,
                diversity_penalty=diversity_penalty,
                beam_group_size=beam_group_size,
            )
        else:
            output_tokens, log_probs, output_logits = self._generate_sampling(
                input_embeds,
                max_len,
                num_text_per_instance,
                temperature,
                greedy,
                nucleus_prob,
            )

        flattened = torch.flatten(output_tokens, start_dim=0, end_dim=1)
        generated_text_list = self.tokenizer.batch_decode(flattened)
        if truncate_on_eos:
          generated_text_list = [x.split(self.tokenizer.eos_token)[0].strip() for x in generated_text_list]
        generated_text_list = [generated_text_list[i*num_text_per_instance : (i+1)*num_text_per_instance] for i in range(batch_size)]

        if return_all_internals:
            # Output is dictionary:
            return_dict = {
                "out_tokens": output_tokens,
                "out_logits": output_logits,
                "out_log_probs": log_probs,
                "text": generated_text_list,
                "input_instructions": whole_instructions,
                "ground_truth_text": gt_text,
                "text_references": inputs["reference_indices"]["target"]["text"],
                "seq_references": [inputs["reference_indices"]["input"]["seq"][j][-1] for j, _ in enumerate(inputs["input"]["seq"])]
            }
            return return_dict

        return output_tokens, log_probs, output_logits, generated_text_list

    def forward_sequences(self, seq_input, get_soft_tokens = False, aaseq_type = "protein"):
        '''
        Forward pass but only for the protein encoder
            - Runs the projection layers to obtain shared latent space representations
        '''

        if isinstance(seq_input, dict):
            seq_input = seq_input["data"] # Decomposes if using collator


        # Pass all sequence tokens to protein encoder
        if self.config.use_aaseq_embeddings: #TODO: implement seq_type argument for embeddings
            if aaseq_type == 'protein':
                protein_embeddings = self.protein_seq_embeddings(seq_input)
            elif aaseq_type == 'domain':
                protein_embeddings = self.domain_embeddings(seq_input)
            elif aaseq_type == 'peptide':
                protein_embeddings = self.peptide_embeddings(seq_input)

            protein_shared_embeddings = self.aaseq_shared_projector(protein_embeddings)
        else:
            if self.config.protein_task_spc_lora and (self.config.lora_specific_style == 'space_specific'):
                protein_embeddings = None

                self.protein_seq_encoder.set_prot_lora_group(1) # One marks the retrieval-space LoRA
                protein_ret_embeddings, _ = self.protein_seq_encoder(seq_input, aggregate=True)

                # Projection layer:
                protein_shared_embeddings = self.aaseq_shared_projector(protein_ret_embeddings)

            else:
                protein_embeddings, _ = self.protein_seq_encoder(seq_input, aggregate = True)
                # Projection layer:
                protein_shared_embeddings = self.aaseq_shared_projector(protein_embeddings)

        output = {
            "original": protein_embeddings,
            "shared": protein_shared_embeddings,
            "token": None,
        }
        '''
        What do these mean?
            - original: raw output proteins from the protein language model
            - shared: output of the PLM outputs fed to the projection layers
            - token: input token embeddings for the model (might be none if you don't specify argument above)
        '''

        if get_soft_tokens:
            if self.config.protein_task_spc_lora and (self.config.lora_specific_style == 'space_specific'):
                self.protein_seq_encoder.set_prot_lora_group(0)
                protein_embeddings, _ = self.protein_seq_encoder(seq_input, aggregate=True)
                protein_soft_tokens = self.token_projectors['aaseq'](protein_embeddings)
            else:
                protein_soft_tokens = self.token_projectors['aaseq'](protein_embeddings)

            output["token"] = protein_soft_tokens

        return output

    def _init_tokenizer(self):
        if self.config.text_encoder_fname.lower().startswith('llama'):
            if "llama-3" in self.config.text_encoder_fname.lower():
                llama_path = os.getenv("LLAMA3_PATH")
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(llama_path)
                self.tokenizer.padding_side = 'right'
            else:
                self.tokenizer = transformers.LlamaTokenizer.from_pretrained(os.path.join(self.pretrained_weights_dir, self.config.text_encoder_fname))
            self.use_llama_tokenizer = True
        else:
            raise ValueError(f'No tokenizer defined for text encoder: {self.config.text_encoder_fname}')

        if self.tokenizer.sep_token == None:
            self.tokenizer.add_tokens("[CLS]")
            self.tokenizer.sep_token = '[CLS]'
            self.tokenizer.sep_token_id = self.tokenizer(self.tokenizer.sep_token, add_special_tokens=False).input_ids[0]
        if self.tokenizer.pad_token == None:  # Fixes padding issue with LLaMA tokenizers automatically
            self.tokenizer.add_tokens("[PAD]")
            self.tokenizer.pad_token = "[PAD]"
            self.tokenizer.pad_token_id = self.tokenizer(self.tokenizer.pad_token, add_special_tokens=False).input_ids[0]

        # Protein replacement token (i.e. to indicate that a protein has been inserted into the input text arbitrarily)
        #   Protein embeddings will be placed here as a sort of soft prompt to the LM
        self.tokenizer.add_tokens("<|protein|>")
        self.prot_replacement_idx = self.tokenizer("<|protein|>", add_special_tokens=False).input_ids[0]

        # Protein retrieval token - similar to [RET] token used by FROMAGe to compute embedding in shared embedding space
        self.tokenizer.add_tokens("[PROT]")
        self.prot_retrieval_idx = self.tokenizer("[PROT]", add_special_tokens=False).input_ids[0]

        # Insert special token for denoting the answer:
        self.tokenizer.add_tokens("[ANSWER]")
        self.answer_idx = self.tokenizer("[ANSWER]", add_special_tokens=False).input_ids[0]

        # Protein structure token:
        self.tokenizer.add_tokens("<|struct|>")
        self.struct_idx = self.tokenizer("<|struct|>", add_special_tokens=False).input_ids[0]

        # Drug structure token:
        self.tokenizer.add_tokens("<|drug|>")
        self.drug_idx = self.tokenizer("<|drug|>", add_special_tokens=False).input_ids[0]

        # Insert a special token for inserting descriptions:
        # MUST ALWAYS COME LAST:
        self.tokenizer.add_tokens("[EXT]") # EXT for external
        self.ext_idx = self.tokenizer("[EXT]",add_special_tokens=False).input_ids[0]

    def _prepare_input_embeddings(self,
            input_ids,
            protein_soft_tokens = None,
            protein_struct_tokens = [],
            drug_soft_tokens = None,
        ):
        '''
        Function processes input embeddings such that we have one input sequence to the language model
        Don't need to provide protein_soft_tokens, will not try to replace if not provided
        '''
        # Step 1: tokenize sequence inputs - already input
        # Step 2: Embed the text input
        z_inputs = self.input_embeddings(input_ids)

        # Step 3: find where the "<|protein|>" token is located (via torch.where)
        # Step 4: insert the protein embeddings where needed for each input in the batch

        if protein_soft_tokens is not None:
            replacement_indices = (input_ids == self.prot_replacement_idx)
            assert replacement_indices.sum() == protein_soft_tokens.shape[0]
            z_inputs[replacement_indices] = protein_soft_tokens

        if len(protein_struct_tokens) > 0:
            replacement_indices = (input_ids == self.struct_idx)
            for i in range(z_inputs.shape[0]):
                if replacement_indices[i,:].sum() > 0: # Only replace if there are structural tokens in the input
                    assert replacement_indices[i,:].sum().item() == protein_struct_tokens[i].shape[0], f"expected: {replacement_indices[i,:].sum().item()} got: {protein_struct_tokens[i].shape[0]}"
                    z_inputs[i,replacement_indices[i,:],:] = protein_struct_tokens[i]

        if drug_soft_tokens is not None:
            replacement_indices = (input_ids == self.drug_idx)
            assert replacement_indices.sum() == drug_soft_tokens.shape[0], f"expected: {replacement_indices.sum()} got: {drug_soft_tokens.shape[0]}"
            z_inputs[replacement_indices] = drug_soft_tokens

        # Step 5: find where the [ret] token was, return those indices
        if self.config.roll_num != 0:
            ret_output_indices = (input_ids == self.prot_retrieval_idx).roll(self.config.roll_num,1) # Rolls the indices along the 1st dimension to the specified amount
        else:
            ret_output_indices = (input_ids == self.prot_retrieval_idx)

        return z_inputs, ret_output_indices

    def _prepare_text_inputs_and_tokenize(
        self,
        instructions: List[str],
        text_input_list: List[List[str]],
        crop_off: bool = False,
        retrieval: bool = False,
        no_pad: bool = False,
        left_pad: bool = False,
    ):
        # Note: biomedgpt tokenizer does not automatically append any special token to end of input, so do it manually:
        # We choose to use the sep token here (see https://huggingface.co/docs/transformers/model_doc/biogpt)
        # (we also have to check for case when sequence is truncated)

        # Ensure we do not append sep tokens twice
        assert all([t[-1] != self.tokenizer.sep_token for t in instructions])

        # Tokenize:
        instruction_tokens = self.tokenizer(
            instructions,
            padding = False,
            truncation = True,
            add_special_tokens = True,
            max_length = self.config.max_text_len,
            # Don't set return_tensors - need lists on which you can insert
        )['input_ids']

        max_len = max([len(l) for l in instruction_tokens])
        joint_tokens = []
        attention_masks = []

        for i, text_input in enumerate(text_input_list): # Over instructions

            num_text_inputs = len(text_input)

            # Have to handle case where we don't have any descriptions to tokenize/replace,
            # e.g. for PPI instructions. (huggingface tokenizer crashes with empty list)
            if len(text_input) != 0:
                # Samples a crop in the input text to let it fit in the given context window. Is a bit more stochastic
                #   -> should allow for more information to be in the text than just truncating the end
                # In the else, just truncates at the end to fit in the max context length
                for i_sub in range(len(text_input)):
                    if not isinstance(text_input[i_sub], str):
                        print("SETTING NULL *********************************************************** Line 1263 model_unified")
                        text_input[i_sub] = "null"

                text_input_tokens = self.tokenizer(
                    text_input,
                    padding = False,
                    truncation=False,
                    add_special_tokens = False,
                    # Don't set return_tensors - need lists that you can insert
                )['input_ids'] # Don't pad, will pad at the end

                max_len_for_sample = (self.config.max_text_len - max_len) // num_text_inputs

                # We do our own truncation, mainly to control for drug tokens and other things added at the end
                # For loop not a big deal - only O(batch_size) - linear in batch size
                for j in range(len(text_input_tokens)):
                    if self.drug_idx in text_input_tokens[j]:
                        where_drug = text_input_tokens[j].index(self.drug_idx) - 3 # -3 is a hack to include "Drug: <>"
                        drug_add = text_input_tokens[j][(where_drug-3):]

                        text_input_tokens[j] = text_input_tokens[j][:(where_drug-3)]
                    else:
                        drug_add = None

                    # CROP SAMPLING IMPLEMENTED HERE
                    if self.training and self.context_crop_sampling and (not crop_off): # Only do this when training, not when testing
                        top_end = len(text_input_tokens[j]) - max_len_for_sample
                        if top_end <= 0: # If here, we have lots of room to spare, so set the start at first token
                            start_i = 0
                        else:
                            start_i = random.randint(0, top_end)
                    else:
                        start_i = 0 # If not performing crop, this is the only thing that changes

                    end_i = start_i + max_len_for_sample
                    if drug_add is not None: # Adjust other tokens you'll have to add on
                        end_i -= len(drug_add)
                    text_input_tokens[j] = text_input_tokens[j][start_i:end_i] # Crop based on sample
                    # If len(text tokens) < max_len_for_sample, gets the whole text

                    if drug_add is not None:
                        text_input_tokens[j] = text_input_tokens[j] + drug_add # Add it at the end bc it would overlap with "Description"
            else:
                text_input_tokens = []

            # Replace in tokens from descriptions to instructions:
            L = multi_replace_tokens(instruction_tokens[i], text_input_tokens, self.ext_idx, eval=False)
            if no_pad:
                L = torch.tensor(L)
            else:
                L = torch.tensor(L + [self.tokenizer.eos_token_id] + ([self.tokenizer.pad_token_id] * (max(self.config.max_text_len - len(L) - 1, 0))))
            joint_tokens.append(L)

            attention_masks.append((L != self.tokenizer.pad_token_id).int())

            assert not torch.any(L == self.ext_idx), 'ERROR [EXT] found in input'

        if left_pad:
            input_ids, attention_masks = left_pad_tensors(
                joint_tokens,
                pad_value=self.tokenizer.pad_token_id,
            )
        else:
            try:
                input_ids = torch.stack(joint_tokens, dim = 0)
            except RuntimeError as e:
                print(
                    "WARNING: exception of tensor with unequal size can currently be caused by "
                    "calling `model.generate` with context augmentation (due to unequal length) "
                    "of prompts. This should be solved by setting `left_pad=True`."
                )
                raise e
            attention_masks = torch.stack(attention_masks, dim = 0)

        return input_ids, attention_masks

    @staticmethod
    def from_pretrained(*,
            pretrained_weights_dir = DEFAULT_PRETRAINED_WEIGHTS_DIR,
            checkpoint_dir = DEFAULT_PRETRAINED_WEIGHTS_DIR,
            model: nn.Module = None,
            config_only = False,
            config = None,
            state_dict_relative_path: str = "txllm_model_ckpt.pt",
            strict_load = False,
            load_plm_directly = False,
            protein_pooling_correction_option = False,
        ):
        '''
        Load pretrained model from checkpoint.
        If config_only is True then only the config is returned
        NOTE: if model is supplied then the model will be updated with the pretrained weights AND CONFIG WILL BE IGNORED (so nothing is returned)

        Parameters:
            pretrained_weights_dir: str
                -
            checkpoint_dir: str
                - Path to directory containing checkpoint information
                - Final-level directory be something like "checkpoint-N"
            config_only: bool
                - If True, only get config, not the model
                - In this case, return
            config: ModelArgs
                - Provided can change some internal model arguments
                - Set enforce_checkpoint_architecture_strict=True if you want to assert that architecture arguments align
        '''
        config_checkpoint = torch.load(os.path.join(checkpoint_dir, "model_args.pt"))

        if config is None:
            config = config_checkpoint
        else:
            assert check_architecture_args(config, config_checkpoint) or (not config.enforce_checkpoint_architecture_strict)

        if config_only:
            return None, config

        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint, load_state_dict_from_zero_checkpoint

        config.n_model_pieces = 1 # Manual to disable deepspeed pipelinemodule initialization
        config.model_splitting = False

        if load_plm_directly and config.use_aaseq_embeddings: # Only need modifications if the model you're loading was trained with the embeddings
            # Decompose name of embeddings
            assert config.protein_seq_embeddings_path is not None
            protein_seq_embeddings_path = os.path.basename(config.protein_seq_embeddings_path)

            # Excludes .pt and gets "aaseq_name", "esm2-nparams", "pooling method"
            decomp_name = protein_seq_embeddings_path.split(".")[0].split("_")
            aaseq_type, nparams_name, pooling_method = decomp_name

            assert pooling_method in ["max", "mean"]

            nparams = None
            if nparams_name == "esm2-3b":
                nparams = "3b"
            elif nparams_name == "esm-650m":
                nparams = "650m"
            else:
                raise NotImplementedError("Invalid number of parameters")

            # Change the arguments to use the correct configuration of ESM:
            config.use_aaseq_embeddings = False
            # Keep config.protein_tokenizer_name fixed
            config.freeze_protein_encoder = "all"
            config.protein_encoder_num_params = nparams
            config.protein_pooling_opt = pooling_method
            config.long_protein_strategy = "split" # Fixed, we don't have other ones
            config.max_protein_len = 1024 # Default
            config.protein_enc_batch_limit = None # Set to default - meant for batched inputs
            config.protein_pooling_correction_option = protein_pooling_correction_option

        if model is None:
            update_model_args_data_dir(config)
            model = UnifiedProCyon(pretrained_weights_dir = pretrained_weights_dir, config = config, for_pretraining = False)

            # Check if state_dict has been consolidated locally:
            if os.path.exists(os.path.join(checkpoint_dir, state_dict_relative_path)):
                print('Via relative path')
                with open(os.path.join(checkpoint_dir, state_dict_relative_path), "rb") as sd:
                    model.load_state_dict(torch.load(sd), strict=strict_load)
            else:
                state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir = checkpoint_dir)
                model.load_state_dict(state_dict, strict = strict_load)
            return model, config
        else:
            if os.path.exists(os.path.join(checkpoint_dir, state_dict_relative_path)):
                with open(os.path.join(checkpoint_dir, state_dict_relative_path), "rb") as sd:
                    state_dict = torch.load(sd)
                strict = False
            else:
                state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir = checkpoint_dir)
                strict = True

            model.load_state_dict(state_dict, strict = (strict or strict_load))
            return

    @staticmethod
    def get_checkpoint_configs(resume_from_checkpoint):
        '''
        Loads all checkpoints from config
        Returns are alphabetically sorted, hopefully that helps with remembering haha
        '''
        model_args = torch.load(os.path.join(resume_from_checkpoint, "model_args.pt"))
        data_args = torch.load(os.path.join(resume_from_checkpoint, "data_args.pt"))
        train_args = torch.load(os.path.join(resume_from_checkpoint, "training_args.pt"))

        return data_args, model_args, train_args

    def save_pretrained(self, output_dir=None):
        '''
        Prepare model for saving to a checkpoint file
        - if output_dir is specified: saves (wrapped) state_dict and config to output_dir
        - otherwise: returns (state_dict, config)
        '''
        model_to_save = unwrap_model(self)
        state_dict = model_to_save.state_dict()

        if output_dir is None:
            return state_dict, self.config
        else:
            # Wrap and save state dict
            training_state = {'model_state_dict': state_dict}
            torch.save(training_state, os.path.join(output_dir, SAVE_TRAINING_STATE_FNAME))

            # Save config
            with open(os.path.join(output_dir, SAVE_CONFIG_FNAME), 'w') as f:
                json.dump(self.config, f)

    def freeze_protein_encoder(self, protein_encoder_mode: str = 'all'):
        if protein_encoder_mode != 'all':
            # freeze params in protein encoder
            if protein_encoder_mode == 'embed':
                for name, param in self.protein_seq_encoder.named_parameters():
                    if name.startswith('model.embed'):
                        param.requires_grad_(False)
            elif protein_encoder_mode == 'lora':
                trainable_pp = 0
                all_pp = 0
                for name, param in self.protein_seq_encoder.named_parameters():
                    all_pp += param.numel()
                    if ('lora' not in name) and (not name.startswith('model.lm_head')):
                        param.requires_grad_(False)
                    else:
                        trainable_pp += param.numel()
                print(f"Trainable Param: {trainable_pp}, All Param: {all_pp}, Trainable Ratio: {(trainable_pp / all_pp)*100:.4f}%")
            elif protein_encoder_mode == 'adapter':
                for name, param in self.protein_seq_encoder.named_parameters():
                    if (not 'adapter' in name) and (not name.startswith('model.lm_head')):
                        param.requires_grad_(False)
            elif protein_encoder_mode == 'prefix':
                for name, param in self.protein_seq_encoder.named_parameters():
                    if (not 'prefix' in name) and (not name.startswith('model.lm_head')):
                        param.requires_grad_(False)
            else:
                max_freeze_layer = int(protein_encoder_mode) - 1
                for name, param in self.protein_seq_encoder.named_parameters():
                    if name.startswith('model.embed'):
                        param.requires_grad_(False)
                    elif name.startswith('model.layers'):
                        layer = int(name.split('.')[2])
                        if layer <= max_freeze_layer:
                            param.requires_grad_(False)
        else:
            # freeze all params in protein encoder (apart from lm head)
            for name, param in self.protein_seq_encoder.named_parameters():
                if not name.startswith('model.lm_head'):
                    param.requires_grad_(False)

    def freeze_text_encoder(self, text_encoder_mode: str = 'all'):
        # Note BioGPT param names have an extra 'biogpt.' prefix
        # TODO: deprecated since BioGPT has been removed - update to work with LLaMA

        if text_encoder_mode != 'all':
            # freeze params in text encoder
            if text_encoder_mode == 'embed':
                for name, param in self.text_encoder.named_parameters():
                    if name.startswith('model.biogpt.embed'):
                        param.requires_grad_(False)
            elif text_encoder_mode == 'lora':
                for name, param in self.text_encoder.named_parameters():
                    if not 'lora' in name:
                        param.requires_grad_(False)
            elif text_encoder_mode == 'adapter':
                for name, param in self.text_encoder.named_parameters():
                    if not 'adapter' in name:
                        param.requires_grad_(False)
            elif text_encoder_mode == 'prefix':
                raise NotImplementedError
            else:
                max_freeze_layer = int(text_encoder_mode) - 1
                for name, param in self.text_encoder.named_parameters():
                    if name.startswith('model.biogpt.embed'):
                        param.requires_grad_(False)
                    elif name.startswith('model.biogpt.layers'):
                        layer = int(name.split('.')[3])
                        if layer <= max_freeze_layer:
                            param.requires_grad_(False)
        else:
            # freeze all params in text encoder (apart from lm head)
            for name, param in self.text_encoder.named_parameters():
                # Note: biogpt has no lm_head (no text MLM (yet))
                param.requires_grad_(False)

def deepspeed_init_with_checkpoint(
        train_args,
        model_args,
        data_args,
        source_zero_stage: int = None,
        target_zero_stage: int = None,
        model = None,
        logger = None,
    ):
    '''
    NOTE: This function works for deepspeed engine target loading, NOT inference model target
    Loads a deepspeed checkpoint based on state of the checkpoint directory
    Need to control for ZeRO 3 and ZeRO 2 loading in source and target
        - Target type == source type -> deepspeed checkpoint load
        - Target_type != source type -> fp32 pooling load, no deepspeed checkpoint load

    Return:
        (model_engine, optimizer)
    '''
    if (source_zero_stage == target_zero_stage) and (not train_args.force_checkpoint_load_consolidation):
        if logger is not None:
            logger.info(f"ZeRO stages match at {source_zero_stage}, loading via deepspeed checkpoint")

        model_engine, optimizer, _, _ = deepspeed.initialize(
            train_args,
            model,
        )

        # Load checkpoint through normal mechanism:
        load_path = model_engine.load_checkpoint(train_args.resume_from_checkpoint, load_module_strict=False)
    else:
        if logger is not None:
            logger.info(f"Source ZeRO stage ({source_zero_stage}) and target ZeRO stage ({target_zero_stage}) do not match, loading pooled fp32 weights")
        # Need cross loading
        model_engine, optimizer, _, _ = deepspeed.initialize(
            train_args,
            model,
        )

    return model_engine, optimizer
