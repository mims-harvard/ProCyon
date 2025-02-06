import esm
import torch
import time
from functools import partial
from torch import nn
from procyon.training.train_utils import batched_split_long_seq, reverse_batched_split, concat_tensor_dict

#from procyon.model.external.esm import esm


from transformers import (
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)

from transformers.models.esm.modeling_esm import *

from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

from procyon.model.mlora import MoLoRAConfig, get_moepeft_model

#from flash_attn import flash_attn_func

def set_attention_type(model: nn.Module, attention_type: str):
    for mn, m in model.named_modules():
        if hasattr(m, 'attention_type'):
            m.attention_type = attention_type
    return model

def set_lora_group(model: nn.Module, index):

    for mn, m in model.named_modules():
        if hasattr(m, 'setting_lora_group'):
            m.setting_lora_group(index)


class ESMPrefix(nn.Module):
    def __init__(self, num_hidden_layers, decoder_attention_heads, d_model, prefix_dropout=0.0, prefix_attn_bn='concat', prefix_attn_composition="add", prefix_mid_dim=800):
        super().__init__()

        # self.match_n_layer = config.decoder_layers if args.num_bias_layers < 0 else args.num_bias_layers
        self.match_n_layer = num_hidden_layers
        self.match_n_head = decoder_attention_heads
        self.n_embd = d_model

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

    def forward(self, bsz, nsamples=1):
        old_bsz = bsz
        bsz = bsz * nsamples
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.wte2.weight.device)
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

        input_tokens_enc = self.input_tokens.unsqueeze(0).expand(old_bsz, -1).to(self.wte2.weight.device)
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
            result.append(temp_dict)
        return result

class ProteinPooler(nn.Module):
    '''
    Similar setup to PfamPooler but with different options
    - Uses flattening along sequence dimension before pooling multiple tokens on batch key
    '''
    def __init__(self, pooling_method = 'mean', protein_pooling_correction_option = True):
        super(ProteinPooler, self).__init__()
        self.pooling_method = pooling_method.lower()
        self.protein_pooling_correction_option = protein_pooling_correction_option

        if self.pooling_method == 'max':
            self.pooler = lambda x: x.max(dim=-2)[0]
        elif self.pooling_method == 'mean':
            if self.protein_pooling_correction_option:
                self.pooler = lambda x: x[1:-1,:].nanmean(dim=-2)
            else:
                self.pooler = lambda x: x.nanmean(dim=-2)
            #self.pooler = lambda x: x.nanmean(dim=-2)
        elif self.pooling_method == 'cls_token': # IF YOU USE SPLITTING OF SEQUENCES, DO NOT USE CLS TOKEN
            self.pooler = lambda x: x[:,0,:] # TODO: Can pool across multiple sub-splits of proteins by max/mean pooling of CLS tokens
        else:
            raise NotImplementedError(f'Protein pooling method {self.pooling_method} is not implemented')

    def forward(self, protein_embeds, batch_keys = None, padding_mask = None):
        # start = time.time()
        if (padding_mask is not None) and (self.pooling_method == 'max'):
            protein_embeds[padding_mask] = -float("inf") # Set to neg inf so that max ignores it
        if batch_keys is not None:
            max_ind = batch_keys.max().item()
            pooled_reps = []
            for i in range(max_ind + 1):
                iship = (batch_keys == i)
                if iship.sum() == 0: # Allow for breaks in continuously increasing integers
                    continue
                # Reshape (#,S,d) -> (1,S + #,d), an essential flattening along dimension 1
                common_prot = protein_embeds[batch_keys == i,:,:]
                common_prot = common_prot.reshape(1, -1, protein_embeds.shape[-1]).squeeze(0)
                if self.pooling_method == 'mean':
                    pad_whole_seq = padding_mask[batch_keys == i].reshape(1,-1).squeeze(0)
                    common_prot = common_prot[~pad_whole_seq]

                rep = self.pooler(common_prot)
                pooled_reps.append(rep)

            result1 = torch.stack(pooled_reps)
        # end = time.time()
        # print(f'Pooler time: {end - start}')

        # start = time.time()
        # if padding_mask is not None:
        #     protein_embeds[padding_mask] = -float("inf") # Set to neg inf so that max ignores it
        # if batch_keys is not None:
        #     # Get indices of unique batch keys and their counts
        #     start = time.time()
        #     pooled_reps = scatter_max(protein_embeds, batch_keys.to(protein_embeds.device), dim=0, out=-torch.ones(torch.unique(batch_keys).shape[0], protein_embeds.shape[1], protein_embeds.shape[-1]).to(protein_embeds.device)*1e7)[0].max(1)[0]
        #     end = time.time()
        #     print(f'Pooler 3 time: {end - start}')

        # result3 = pooled_reps

        # if batch_keys is not None:
        #     start = time.time()
        #     # Get indices of unique batch keys and their counts
        #     unique_keys, counts = batch_keys.unique(return_counts=True)
        #     end = time.time()
        #     print(f'unique key time: {end - start}')

        #     # Compute pooled representations for each unique batch key
        #     start = time.time()
        #     start_indices = torch.cat((torch.tensor([0]), counts[:-1].cumsum(dim=0)))
        #     end_indices = counts.cumsum(dim=0)
        #     end = time.time()
        #     print(f'indices time: {end - start}')

        #     start = time.time()
        #     pooled_reps = torch.stack([protein_embeds[start:end].masked_select(~padding_mask[start:end].unsqueeze(-1).repeat(1, 1, protein_embeds.shape[-1])).reshape(-1, protein_embeds.shape[-1]).max(0)[0] for start, end in zip(start_indices, end_indices)])
        #     end = time.time()
        #     print(f'pool time: {end - start}')

        #     # return pooled_reps

        # # else:
        #     # return self.pooler(protein_embeds)

        # result2 = pooled_reps

        return result1


class ESM_PLM_basic(torch.nn.Module):
    def __init__(self, num_params = '35m', **kwargs):
        super(ESM_PLM_basic, self).__init__()

        self.num_params = num_params.lower()

        if self.num_params == '35m':
            self.model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            self.repr_layers = 12
            self.embedding_size = 480
        elif self.num_params == '650m':
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.repr_layers = 33
            self.embedding_size = 1280
        elif self.num_params == '3b':
            self.model, _ = esm.pretrained.esm2_t36_3B_UR50D()
            self.repr_layers = 36
            self.embedding_size = 2560


        # Add other numbers of parameter models here (based on https://github.com/facebookresearch/esm#available-models-and-datasets-)

        self.batch_converter = self.alphabet.get_batch_converter()

    def forward(self, batch, aggregate = True):
        # ESM API to process forward pass
        # IF aggregate=True, return is shape (B,E), else (B,len,E)

        _, _, batch_tokens = self.batch_converter(batch)
        batch_tokens = batch_tokens.to(self.model.device)
        results = self.model(batch_tokens, repr_layers=[self.repr_layers], return_contacts = False)

        z = results['representations'][self.repr_layers]

        if aggregate:
            # Reduce to per-sequence token through mean:
            aggmask_nanmask = (batch_tokens == self.alphabet.padding_idx)
            z[aggmask_nanmask] = torch.nan
            z = z.nanmean(dim=1)

        return z

class EsmSelfOutputQuant(EsmSelfOutput):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # print(hidden_states.dtype, input_tensor.dtype)
        # hidden_states += input_tensor
        ret_hidden_states = hidden_states + input_tensor
        return ret_hidden_states

class EsmOutputQuant(EsmOutput):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # hidden_states += input_tensor
        ret_hidden_states = hidden_states + input_tensor
        return ret_hidden_states

class EsmAttentionQuant(EsmAttention):
    def __init__(self, config):
        super().__init__(config)
        self.output = EsmSelfOutputQuant(config)

class EsmLayerQuant(EsmLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = EsmAttentionQuant(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise RuntimeError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = EsmAttentionQuant(config)
        self.output = EsmOutputQuant(config)

class ESMEncoderQuant(EsmEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([EsmLayerQuant(config) for _ in range(config.num_hidden_layers)])

class EsmModelQuant(EsmModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.encoder = ESMEncoderQuant(config)

class EsmForMaskedLMQuant(EsmForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.esm = EsmModelQuant(config, add_pooling_layer=False)


class ESM_PLM(torch.nn.Module):
    def __init__(
        self,
        pretrained_weights_dir,
        num_params = '3b',
        pooling_method = 'max',
        padding_idx = 1,
        eos_idx = 2,
        max_protein_len = 1024,
        long_protein_strategy = 'split',
        max_batch_forward_pass = None,
        use_lora = False,
        use_q_lora = False,
        use_task_spc_lora = False,
        lora_alpha = 8,
        lora_r = 8,
        use_adapter = False,
        adapter_rank = 8,
        use_prefix=False,
        prefix_dropout=0.0,
        prefix_mid_dim=800,
        prefix_attn_bn=30,
        protein_attention_type = 'vanilla',
        lora_parameters = 'default',
        lora_num = 2,
        protein_pooling_correction_option = False,
    ):
        super(ESM_PLM, self).__init__()

        self.num_params = num_params.lower()
        self.pooling_method = pooling_method
        self.padding_idx = padding_idx
        self.protein_pooling_correction_option = protein_pooling_correction_option
        self.pooler = ProteinPooler(
            pooling_method = self.pooling_method,
            protein_pooling_correction_option = self.protein_pooling_correction_option,
        )
        self.long_protein_strategy = long_protein_strategy
        self.padding_idx, self.eos_idx = padding_idx, eos_idx
        self.max_protein_len = max_protein_len
        self.max_batch_forward_pass = max_batch_forward_pass

        self.use_prefix = use_prefix

        self.use_task_spc_lora = use_task_spc_lora

        self.seq_proc = partial(batched_split_long_seq,
            padding_idx = self.padding_idx,
            eos_idx = self.eos_idx,
            long_protein_strategy = self.long_protein_strategy,
            max_protein_len = self.max_protein_len)
        # extra_model_kwargs = {
        #     'use_lora': use_lora,
        #     'lora_alpha': lora_alpha,
        #     'lora_r': lora_r,
        #     'use_adapter': use_adapter,
        #     'adapter_rank': adapter_rank
        # }
        assert not ((pooling_method == 'cls_token') and (long_protein_strategy == 'split')), 'Cannot use CLS token with split strategy'

        if self.num_params == '8m':
            self.model, _ = esm.pretrained.esm2_t6_8M_UR50D()
            #self.model, _ = esm.pretrained.load_model_and_alphabet_local(pretrained_weights_dir+'/esm2_t6_8M_UR50D.pt')
            self.repr_layer = 6
            self.embedding_size = 320
        elif self.num_params == '35m':
            self.model, _ = esm.pretrained.esm2_t12_35M_UR50D()
            #self.model, _ = esm.pretrained.load_model_and_alphabet_local(pretrained_weights_dir+'/esm2_t12_35M_UR50D.pt')
            self.repr_layer = 12
            self.embedding_size = 480
        elif self.num_params == '650m':
            self.model, _ = esm.pretrained.esm2_t33_650M_UR50D()
            # self.model, _ = esm.pretrained.load_model_and_alphabet_local(pretrained_weights_dir+'/esm2_t33_650M_UR50D.pt')
            self.repr_layer = 33
            self.embedding_size = 1280
        elif self.num_params == '3b':
            self.model, _ = esm.pretrained.esm2_t36_3B_UR50D()
            # FIXME: This local loading is not working: https://github.com/facebookresearch/esm/discussions/514.  Investigate later.
            # self.model, _ = esm.pretrained.load_model_and_alphabet_local(pretrained_weights_dir+'/esm2_t36_3B_UR50D.pt')
            self.repr_layer = 36
            self.embedding_size = 2560
        elif self.num_params == '15b':
            self.model, _ = esm.pretrained.esm2_t48_15B_UR50D()
            # self.model, _ = esm.pretrained.load_model_and_alphabet_local(pretrained_weights_dir+'/esm2_t48_15B_UR50D.pt')
            self.repr_layer = 48
            self.embedding_size = 5120
        elif 'official' in self.num_params:
            if '650m' in self.num_params:
                model_name = "facebook/esm2_t33_650M_UR50D"
                self.repr_layer = 33
                self.embedding_size = 1280
            elif '3b' in self.num_params:
                model_name = "facebook/esm2_t36_3B_UR50D"
                self.repr_layer = 36
                self.embedding_size = 2560
            elif '15b' in self.num_params:
                model_name = "facebook/esm2_t48_15B_UR50D"
                self.repr_layer = 48
                self.embedding_size = 5120
            elif '150m' in self.num_params:
                model_name = "facebook/esm2_t30_150M_UR50D"
                self.repr_layer = 30
                self.embedding_size = 640
            else:
                raise ValueError("Invalid number of parameters for ESM '{}'".format(self.num_params))

            if lora_parameters == 'attn':
                target_lora_modules = ["query", "key", "value"]
            elif lora_parameters == 'mlp':
                target_lora_modules = ["dense"]
            else:
                target_lora_modules = ["query", "key", "value", "dense"]

            if not use_task_spc_lora:

                peft_config = LoraConfig(
                    task_type=TaskType.TOKEN_CLS,
                    inference_mode=False,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=target_lora_modules, # also try "dense_h_to_4h" and "dense_4h_to_h"
                    lora_dropout=0.1,
                    bias="none" # or "all" or "lora_only"
                )

                if use_q_lora and not use_lora:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit = True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type='nf4',
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
                else:
                    bnb_config = None

                self.model = EsmForMaskedLMQuant.from_pretrained(model_name, quantization_config=bnb_config)
                self.model = set_attention_type(self.model, protein_attention_type)


                if use_lora and not use_q_lora:
                    self.model = get_peft_model(self.model, peft_config)
                elif use_q_lora:

                    self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False)

                    self.model = get_peft_model(self.model, peft_config)
            else:
                peft_config = MoLoRAConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=target_lora_modules,
                    lora_dropout=0.1,
                    bias='none',
                    task_type="CAUSAL_LM",
                    moe_num_experts=lora_num
                )
                if use_q_lora and not use_lora:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit = True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type='nf4',
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
                else:
                    bnb_config = None
                self.model = EsmForMaskedLMQuant.from_pretrained(model_name, quantization_config=bnb_config)
                self.model = set_attention_type(self.model, protein_attention_type)

                if use_lora and not use_q_lora:
                    self.model = get_moepeft_model(self.model, peft_config)
                elif use_q_lora:

                    self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False)

                    self.model = get_moepeft_model(self.model, peft_config)

        else:

            raise ValueError(f'ESM model with {self.num_params} parameters is not implemented')

        if self.use_prefix:
            self.prefix_model = ESMPrefix(self.repr_layer, self.model.attention_heads, self.embedding_size, prefix_dropout, prefix_attn_bn, prefix_mid_dim=prefix_mid_dim)

    def set_prot_lora_group(self, index):
        set_lora_group(self.model, index)

    def forward(self, tokens, aggregate = True):
        # Modified forward from ESM_PLM_basic
        # ESM API to process forward pass
        # IF aggregate=True, return is shape (B,E), else (B,len,E)

        # Split into chunks here ------:
        batch_tokens, batch_keys, eos_loc = self.seq_proc(tokens)
        # batch_keys will be None if we don't need to split the tokens (allows you to detect when reverse splitting is needed)
        if self.use_prefix:
            bs = batch_tokens.shape[0]
            prefix_states = self.prefix_model(bs)
        else:
            prefix_states = {'self': None}
        if self.max_batch_forward_pass is not None:
            # Restrict the batch size of a forward pass by the user-given parameter
            res_list = []
            split_tokens = torch.split(batch_tokens, self.max_batch_forward_pass, dim = 0)
            for i in range(len(split_tokens)):
                if "official" in self.num_params:
                    r = self.model(batch_tokens, output_hidden_states = True)
                    r['representations'] = r['hidden_states']
                else:
                    r = self.model(batch_tokens, repr_layers=[self.repr_layer], return_contacts = False)
                res_list.append(r)
            results = concat_tensor_dict(res_list)
        else:
            # FIXME
            # Everything passed in one batch
            if "official" in self.num_params:
                results = self.model(batch_tokens, output_hidden_states = True)
                results['representations'] = results['hidden_states']
            else:
                results = self.model(batch_tokens, repr_layers=[self.repr_layer], return_contacts = False)

        z = results['representations'][self.repr_layer]
        #z = results['representations'][self.repr_layer][:,:245,:] # For testing
        # print(results['representations'].keys())
        # z = results['representations'][-1]

        if aggregate:
            # Reduce to per-sequence token through mean:
            padmask = (batch_tokens == self.padding_idx)
            z = self.pooler(z, batch_keys = batch_keys, padding_mask = padmask)
            logits = results['logits']
            # WARNING: Pooling with split long_protein_strategy will pool all extra CLS and EOS tokens across pooler
        else: # If not aggregating, don't touch logits
            # Map split sequences back to original size:
            if batch_keys is not None:
                z = reverse_batched_split(z, batch_keys, eos_locs = eos_loc)
                logits = reverse_batched_split(results['logits'], batch_keys, eos_locs = eos_loc)
            else:
                logits = results['logits']

        #print('logits', results['logits'].shape)
        return z, logits # Logits are for masked language modeling