import torch
from torch import nn
import numpy as np
from transformers.modeling_outputs import BaseModelOutput

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os 
from procyon.model.external.biogpt.modeling_biogpt import BioGptForCausalLM 
from procyon.model.external.biogpt.tokenization_biogpt import BioGptTokenizer

from dotenv import load_dotenv
load_dotenv()
DATA_DIR = os.getenv('DATA_DIR')
assert DATA_DIR is not None, "DATA_DIR must be set in .env file"

from typing import List
from procyon.model import simcse
from procyon.model.simcse import Pooler, MLPLayer, Similarity
from transformers.modeling_outputs import BaseModelOutputWithPooling


# TODO: convert to arg?
# Note: this is only used when already_tokenized=False
TOKENIZER_MAX_LENGTH = 1024 

class BioPrefix(nn.Module):
    def __init__(self, config, prefix_dropout=0.0, prefix_attn_bn=30, prefix_attn_composition="add", prefix_mid_dim=800):
        super().__init__()

        # self.match_n_layer = config.decoder_layers if args.num_bias_layers < 0 else args.num_bias_layers
        self.match_n_layer = config.num_hidden_layers
        self.match_n_head = config.num_attention_heads
        self.n_embd = config.hidden_size

        self.match_n_embd = self.n_embd // self.match_n_head

        self.mid_dim = prefix_mid_dim
        self.attn_bn = prefix_attn_bn
        self.prefix_dropout = prefix_dropout

        self.dropout = nn.Dropout(self.prefix_dropout)

        self.input_tokens = torch.arange(self.attn_bn).long()
        self.wte = nn.Embedding(self.attn_bn, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        self.wte_enc = nn.Embedding(self.attn_bn, self.n_embd)
        self.control_trans_enc = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        self.wte2 = nn.Embedding(self.attn_bn, self.n_embd)
        self.control_trans2 = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        # if args.lisa_option == "cross_attn":
        #     self.apply(init_lisa_params)

    def forward(self, bsz, nsamples=1, device="gpu"):
        old_bsz = bsz
        bsz = bsz * nsamples
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        temp_control2 = self.wte2(input_tokens)
        past_key_values2 = self.control_trans2(temp_control2)  # bsz, seqlen, layer*emb
        past_key_values2 = past_key_values2.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                 self.match_n_embd)
        past_key_values2 = self.dropout(past_key_values2)
        past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)

        input_tokens_enc = self.input_tokens.unsqueeze(0).expand(old_bsz, -1).to(device)
        temp_control_enc = self.wte_enc(input_tokens_enc)
        past_key_values_enc = self.control_trans_enc(temp_control_enc)  # bsz, seqlen, layer*emb
        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                       self.match_n_embd)
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp_dict = {'self': {"prev_key": key_val[0].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                  "prev_value": key_val[1].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                  "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val.device) #bsz, attn_bn
                                  },
                         }
            temp_dict = [key_val[0].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                         key_val[1].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                         torch.zeros(bsz, seqlen).to(key_val.device)]
            """
            key_val2 = past_key_values2[i]
            temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                            "prev_value": key_val2[1].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                            "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val2.device)
                                            }
            key_val_enc = past_key_values_enc[i]
            # at generation time, this is expanded automatically to the beam size
            temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous().view(old_bsz*self.match_n_head, -1, self.match_n_embd),
                                    "prev_value": key_val_enc[1].contiguous().view(old_bsz*self.match_n_head, -1, self.match_n_embd),
                                    "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen).to(key_val_enc.device)
                                    }
            """
            result.append(temp_dict)
        return result

class BioGPT(nn.Module):
    def __init__(self,
            pooler_type: str,
            already_tokenized = 'ignore',
            model_path = f'{DATA_DIR}/model_weights/BioGPT-Large',
            use_lora = False,
            lora_alpha = 8,
            lora_r = 8,
            use_adapter=False,
            adapter_rank=8,
            use_prefix=False,
            prefix_dropout=0.0,
            prefix_mid_dim=800,
            prefix_attn_bn=30,
        ):
        super(BioGPT, self).__init__()
        
        self.model_path = model_path
        self.use_lora = use_lora
        self.lora_alpha = lora_alpha

        self.tokenizer = BioGptTokenizer.from_pretrained(model_path)
        assert self.tokenizer.padding_side == "right"
        assert self.tokenizer.truncation_side == "right"

        config = AutoConfig.from_pretrained(model_path)
        config.use_lora = use_lora
        config.lora_alpha = lora_alpha
        config.lora_r = lora_r
        config.use_adapter = use_adapter
        config.adapter_rank = adapter_rank
        self.use_prefix = use_prefix
        
        self.model = BioGptForCausalLM.from_pretrained(model_path, config=config)
        if self.use_prefix:
            self.prefix_model = BioPrefix(config, prefix_dropout, prefix_attn_bn, prefix_mid_dim=prefix_mid_dim)

        self.pooler_type = pooler_type
        self.pooler = Pooler(self.pooler_type)
        if self.pooler_type == "cls":
            self.mlp = MLPLayer(self.model.config.hidden_size)
        self.sim = Similarity()


    def forward(self, input_ids = None, attn_masks = None, text_list = None, pooling='special_token', do_pooling=True):
        '''
        Args:
            input_ids: provide only if already_tokenized=False
            attn_masks: provide only if already_tokenized=False
            text_list: provide only if already_tokenized=True
        '''
        
        if text_list is None:
            assert input_ids is not None and attn_masks is not None 
        else:
            assert input_ids is None and attn_masks is None         
            # Prepare input by appending sep tokens and tokenizing
            input_ids, attn_masks, text_list  = self.prepare_inputs_and_tokenize(text_list, self.tokenizer, TOKENIZER_MAX_LENGTH)
        
        input_ids, attn_masks = input_ids.to(self.model.device), attn_masks.to(self.model.device)

        # Verify correct tokenization
        final_token_idxs = np.arange(len(input_ids)), simcse.get_final_token_indices(attn_masks)
        assert all(input_ids[final_token_idxs] == self.tokenizer.sep_token_id)

        if self.use_prefix:
            bsz = input_ids.shape[0]
            prefix_states = self.prefix_model(bsz, 1, self.prefix_model.wte2.weight.device)
        else:
            #prefix_states = {'self': None}
            prefix_states = None
        outputs = self.model(input_ids = input_ids, attention_mask = attn_masks, output_hidden_states = True, prefix_states=prefix_states)

        outputs = BaseModelOutput(
            last_hidden_state=outputs.hidden_states[-1],
            hidden_states=outputs.hidden_states,
        )

        if do_pooling:
            pooler_output = self.pooler(attn_masks, outputs)
            if self.pooler_type == "cls":
                pooler_output = self.mlp(pooler_output)

            outputs = BaseModelOutputWithPooling(
                pooler_output=pooler_output,
                **outputs,
            )
    
        return outputs


    def cl_forward(self, input_ids, attn_masks):
        return simcse.cl_forward(self, input_ids, attn_masks)
    

    @staticmethod
    def prepare_inputs_and_tokenize(text_list: List[str], tokenizer, max_text_len):
        # Note: biomedgpt tokenizer does not automatically append any special token to end of input, so do it manually:
        # We choose to use the sep token here (see https://huggingface.co/docs/transformers/model_doc/biogpt)
        # (we also have to check for case when sequence is truncated)
    
        # Ensure we do not append sep tokens twice
        assert all([t[-1] != tokenizer.sep_token for t in text_list])
        # Append sep tokens
        text_list = [t + tokenizer.sep_token for t in text_list]

        # Tokenize 
        inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt", max_length=max_text_len)
        
        # If we have truncated sequences, we need to replace the last token with sep token:
        overflow_rows = torch.where((inputs['input_ids'][:, -1] != 2) & (inputs['input_ids'][:, -1] != 1))
        if len(overflow_rows[0]) > 0:
            inputs['input_ids'][overflow_rows[0], -1] = tokenizer.sep_token_id

        return inputs['input_ids'], inputs['attention_mask'], text_list    


class BioGPTPostTokenization(nn.Module):
    '''
    BioGPT class as above but with no tokenizer
        - Used for multimodal architecture because tokenization/embedding process needs special consideration
    Also outputs more autoregressive-LM-based outputs, no pooling
    '''
    def __init__(self,
            model_path = f'{DATA_DIR}/model_weights/BioGPT-Large',
            use_lora = False,
            lora_alpha = 8,
            lora_r = 8,
            use_adapter=False,
            adapter_rank=8,
            use_prefix=False,
            prefix_dropout=0.0,
            prefix_mid_dim=800,
            prefix_attn_bn=30,
        ):
        super(BioGPTPostTokenization, self).__init__()
        
        self.model_path = model_path
        self.use_lora = use_lora
        self.lora_alpha = lora_alpha

        config = AutoConfig.from_pretrained(model_path)
        config.use_lora = use_lora
        config.lora_alpha = lora_alpha
        config.lora_r = lora_r
        config.use_adapter = use_adapter
        config.adapter_rank = adapter_rank
        self.use_prefix = use_prefix
        
        self.model = BioGptForCausalLM.from_pretrained(model_path, config=config)
        # if self.use_prefix:
        #     self.prefix_model = BioPrefix(config, prefix_dropout, prefix_attn_bn, prefix_mid_dim=prefix_mid_dim)

        assert not self.use_prefix, "Prefix not yet implemented for this version of BioGPT, need to extend"

        # self.pooler_type = "final" # TODO: generalise
        # self.pooler = Pooler(self.pooler_type)
        # if self.pooler_type == "cls":
        #     self.mlp = MLPLayer(self.model.config.hidden_size)
        #self.sim = Similarity()


    def forward(self, input_embeds = None, attn_masks = None, full_labels = None):
        '''
        Args:
            input_ids: provide only if already_tokenized=False
            attn_masks: provide only if already_tokenized=False
            text_list: provide only if already_tokenized=True
        '''
        
        #input_ids, attn_masks = input_ids.to(self.model.device), attn_masks.to(self.model.device)

        # Verify correct tokenization
        # final_token_idxs = np.arange(len(input_ids)), simcse.get_final_token_indices(attn_masks)
        # assert all(input_ids[final_token_idxs] == self.tokenizer.sep_token_id)

        # TODO: Figure out prefix states
        prefix_states = None

        outputs = self.model(
            inputs_embeds = input_embeds, 
            attention_mask = attn_masks, 
            output_hidden_states = True, 
            prefix_states=prefix_states,
            labels = full_labels,
        )

        # Add some additional variables to the original outputs:
        # outputs = BaseModelOutput(
        #     last_hidden_state=outputs_org.hidden_states[-1],
        #     **outputs_org
        # )
    
        return outputs


    def cl_forward(self, input_ids, attn_masks):
        raise NotImplementedError('Not implemented for BioGPT post-tokenized version')
        #return simcse.cl_forward(self, input_ids, attn_masks)
