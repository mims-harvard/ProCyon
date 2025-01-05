from typing import List, Optional, Tuple, Union
from deepspeed.runtime.activation_checkpointing import checkpointing
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from deepspeed.pipe import PipelineModule
import deepspeed.runtime.utils as ds_utils
from deepspeed import comm as dist
from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_varlen_func, flash_attn_func
# from deepspeed.runtime.utils import logger
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from transformers import BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import add_start_docstrings_to_model_forward, LLAMA_INPUTS_DOCSTRING, LlamaModel, LlamaDecoderLayer, LlamaAttention
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from procyon.model.mlora import MoLoRAConfig, get_moepeft_model
import math

load_dotenv()
DATA_DIR = os.getenv('DATA_DIR')
assert DATA_DIR is not None, "DATA_DIR must be set in .env file"

# logger = logging.get_logger(__name__)

def set_lora_group(model: nn.Module, index):

    for mn, m in model.named_modules():
        if hasattr(m, 'setting_lora_group'):
            m.setting_lora_group(index)

class PipelineModuleLamma(PipelineModule):
    def __init__(self, layers, num_stages=None, topology=None, loss_fn=None, seed_layers=False, seed_fn=None, base_seed=1234, partition_method='parameters', activation_checkpoint_interval=0, activation_checkpoint_func=checkpointing.checkpoint, checkpointable_layers=None):
        super().__init__(layers, num_stages, topology, loss_fn, seed_layers, seed_fn, base_seed, partition_method, activation_checkpoint_interval, activation_checkpoint_func, checkpointable_layers)

    def forward(self, forward_input, past_key_values, all_hidden_states: Tuple, output_hidden_states, gradient_checkpointing, attention_mask, position_ids, output_attentions, use_cache, all_self_attns, next_decoder_cache):
        self.micro_offset += 1
        self.all_hidden_states = all_hidden_states
        self.all_self_attns = all_self_attns
        self.next_decoder_cache = next_decoder_cache
        def exec_range_func(start, end):
            ''' Helper function to be used with checkpoint()
            Adapted from torch.utils.checkpoint:checkpoint_sequential()
            '''
            local_micro_offset = self.micro_offset + 1

            def exec_func(*inputs):
                # Single tensor inputs need to be unwrapped
                if len(inputs) == 1:
                    inputs = inputs[0]
                for idx, layer in enumerate(self.forward_funcs[start:end]):
                    self.curr_layer = idx + self._local_start
                    # print("########", start, end, dist.get_rank())
                    if self.seed_layers:
                        new_seed = (self.base_seed * local_micro_offset) + self.curr_layer
                        if self.seed_fn:
                            self.seed_fn(new_seed)
                        else:
                            ds_utils.set_random_seed(new_seed)
                    if output_hidden_states:
                        self.all_hidden_states += (inputs, )
                    past_key_value = past_key_values[idx] if past_key_values is not None else None
                    if gradient_checkpointing and self.training:
                        def create_custom_forward(module):
                            def custom_forward(*cinputs):
                                # None for past_key_value
                                return module(*cinputs, past_key_value, output_attentions)
                            return custom_forward
                        layer_outputs = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(layer),
                            inputs,
                            attention_mask,
                            position_ids,
                        )
                    else:
                        layer_outputs = layer(inputs,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids,
                                            past_key_value=past_key_value,
                                            output_attentions=output_attentions,
                                            use_cache=use_cache)
                    inputs = layer_outputs[0]
                    if use_cache:
                        self.next_decoder_cache += (layer_outputs[2 if output_attentions else 1], )
                    if output_attentions:
                        self.all_self_attns += (layer_outputs[1], )
                    # inputs = layer(inputs)
                return inputs

            return exec_func

        if self.activation_checkpoint_interval == 0:
            func = exec_range_func(0, len(self.forward_funcs))
            x = func(forward_input)
        else:
            num_layers = len(self.forward_funcs)
            x = forward_input
            for start_idx in range(0, num_layers, self.activation_checkpoint_interval):
                end_idx = min(start_idx + self.activation_checkpoint_interval, num_layers)

                funcs = self.forward_funcs[start_idx:end_idx]
                # Since we either pass tensors or tuples of tensors without unpacking, we
                # need to be careful not to double-wrap tensors with tuple.
                if not isinstance(x, tuple):
                    x = (x, )

                if self._is_checkpointable(funcs):
                    x = self.activation_checkpoint_func(exec_range_func(start_idx, end_idx), *x)
                else:
                    x = exec_range_func(start_idx, end_idx)(*x)
        return x, self.all_hidden_states, self.all_self_attns, self.next_decoder_cache

    # def state_dict(self, *args, destination=None, prefix='', keep_vars=False):

    #     # Overload of save_state_dict - https://github.com/microsoft/DeepSpeed/blob/3e94f8c75116377d4b1c32b8c674368a27fb2a77/deepspeed/runtime/pipe/module.py#L567
    #     # NOTE: exclude_frozen_params is handled upstream, so we set to False
    #     exclude_frozen_params=False
    #     save_dir = None
    #     checkpoint_engine = None

    #     # Processes having the same model parallel rank on different data parallel instances
    #     # have identical layer weights.  We can distribute the task of saving the layer weights
    #     # among the data parallel ranks.  For example, if a pipeline stage has 9 layers and
    #     # if there are 2 data parallel instances, rank 0 will save the first 5 layers and
    #     # rank 1 will save the last 4.
    #     dp_rank = self._grid.data_parallel_id
    #     dp_size = self._grid.data_parallel_size
    #     num_layers = len(self.forward_funcs)
    #     # if self.checkpoint_parallel_write_pipeline:
    #     #     # spread layers evenly across data parallel ranks
    #     #     offsets = ds_utils.partition_uniform(num_layers, dp_size)
    #     #     start, end = offsets[dp_rank], offsets[dp_rank + 1]
    #     #else:
    #     # data parallel rank 0 writes all layers
    #     if dp_rank != 0:
    #         return
    #     start, end = 0, num_layers
    #     layer_list = self.forward_funcs[start:end]

    #     total_state_dict = {}

    #     for idx, layer in enumerate(layer_list):
    #         if not hasattr(layer, 'state_dict'):
    #             continue

    #         orig_state_dict = layer.state_dict(prefix = prefix + layer.name + ".")
    #         final_state_dict = clone_tensors_for_torch_save(orig_state_dict)

    #         print(list(final_state_dict.keys()))
    #         print()
    #         #assert len(set(total_state_dict.keys()).intersection(set(final_state_dict.keys()))) > 0, "Conflict in dictionary keys already found"

    #         total_state_dict.update(final_state_dict)

    #     return total_state_dict

class FlashLlamaAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # attn start
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        #     raise ValueError(
        #         f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
        #         f" {attn_weights.size()}"
        #     )

        # if attention_mask is not None:
        #     if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
        #         raise ValueError(
        #             f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
        #         )
        #     attn_weights = attn_weights + attention_mask

        # # upcast attention to fp32
        # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # attn_output = torch.matmul(attn_weights, value_states)

        # if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        #     raise ValueError(
        #         f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
        #         f" {attn_output.size()}"
        #     )

        # attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # attn end
        if output_attentions:
            attn_output, attn_weights, _ = flash_attn_func(query_states, key_states, value_states, return_attn_probs = True)
        else:
            attn_output = flash_attn_func(query_states, key_states, value_states, return_attn_probs = False)
            attn_weights = None
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class FlashLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.self_attn = FlashLlamaAttention(config=config)


class PiplineLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig, attention_type = 'vanilla'):
        super().__init__(config)
        if attention_type == 'vanilla':
            self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        elif attention_type == 'flash_attn_v1':
            self.layers = nn.ModuleList([FlashLlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        else:
            raise NotImplementedError



    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.Tensor = None, position_ids: torch.LongTensor = None, past_key_values: List[torch.FloatTensor] = None, inputs_embeds: torch.FloatTensor = None, use_cache: bool = None, output_attentions: bool = None, output_hidden_states: bool = None, return_dict: bool = None) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                # logger.warning_once(
                #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                # )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        if type(self.layers) is PipelineModuleLamma:

            hidden_states, all_hidden_states, all_self_attns, next_decoder_cache = self.layers(
                hidden_states,
                past_key_values, all_hidden_states, output_hidden_states, self.gradient_checkpointing, attention_mask, position_ids, output_attentions, use_cache, all_self_attns, next_decoder_cache
            )
        else:
            for idx, decoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                past_key_value = past_key_values[idx] if past_key_values is not None else None

                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module(*inputs, past_key_value, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(decoder_layer),
                        hidden_states,
                        attention_mask,
                        position_ids,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class PiplineLlamaForCausalLM(transformers.LlamaForCausalLM):
    def __init__(self, config, attention_type = 'vanilla'):
        super().__init__(config)
        self.model = PiplineLlamaModel(config, attention_type = attention_type)



class LlamaPostTokenization(nn.Module):
    '''
    BioGPT class as above but with no tokenizer
        - Used for multimodal architecture because tokenization/embedding process needs special consideration
    Also outputs more autoregressive-LM-based outputs, no pooling
    '''
    def __init__(self,
            model_path = f'{DATA_DIR}/model_weights/llama',
            model_splitting = False,
            n_model_pieces = 2,
            use_lora = False,
            attention_type = 'flash_attn_v1',
            max_gen_len = 50,
            use_q_lora = False,
            residual_dropout = 0.2,
            lora_r = 16,
            lora_alpha=8,
            use_task_spc_lora = False,
            lora_num = 2,
            for_pretraining = True,
        ):
        super(LlamaPostTokenization, self).__init__()

        self.model_path = model_path
        # self.use_lora = use_lora
        # self.lora_alpha = lora_alpha
        self.attention_type = attention_type
        if "llama-3" in model_path:
            config = None
        else:
            config = transformers.AutoConfig.from_pretrained(model_path)
        # config.use_lora = use_lora
        # config.lora_alpha = lora_alpha
        # config.lora_r = lora_r
        # config.use_adapter = use_adapter
        # config.adapter_rank = adapter_rank
        # self.use_prefix = use_prefix
        self.use_task_spc_lora = use_task_spc_lora

        self.kv_cache = None
        self.attention_type = attention_type
        # self.model = BioGptForCausalLM.from_pretrained(model_path, config=config)
        if use_q_lora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            assert not ("llama-3" in model_path), "q-lora not supported currently with llama-3"
        else:
            bnb_config = None
        # self.model = BioGptForCausalLM.from_pretrained(model_path, config=config)
        if model_splitting:
            if "flash" in attention_type:
                self.model = PiplineLlamaForCausalLM.from_pretrained(
                    model_path, config=config, attention_type = attention_type, quantization_config=bnb_config
                )
            else:
                self.model = PiplineLlamaForCausalLM.from_pretrained(
                    model_path, config=config, attention_type = attention_type, quantization_config=bnb_config
                )
        else:
            if "llama-3" in model_path:
                llama_path = os.getenv("LLAMA3_PATH")
                if for_pretraining:
                    self.model = transformers.AutoModelForCausalLM.from_pretrained(llama_path)
                else:
                    llama_config = transformers.AutoConfig.from_pretrained(llama_path)
                    self.model = transformers.AutoModelForCausalLM.from_config(llama_config)
                #self.model = transformers.AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B', cache_dir = f"{DATA_DIR}/model_weights/llama-3-8b")
            else:
                self.model = transformers.LlamaForCausalLM.from_pretrained(model_path, config=config, quantization_config=bnb_config)
        # for pn, p in self.model.named_parameters():
        #     print(pn, p.requires_grad)

        if not use_task_spc_lora:

            if use_lora and not use_q_lora:
                peft_config = LoraConfig(
                    task_type = TaskType.CAUSAL_LM,
                    inference_mode = False,
                    r = lora_r,
                    lora_alpha = lora_alpha,
                    lora_dropout = 0.1
                )
                self.model.gradient_checkpointing_enable()
                self.model = get_peft_model(self.model, peft_config)
            elif use_q_lora:
                self.model.gradient_checkpointing_enable()
                self.model = prepare_model_for_kbit_training(self.model)
                peft_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=0.1,
                    inference_mode = False,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM
                )
                self.model = get_peft_model(self.model, peft_config)

            if model_splitting:
                if use_lora or use_q_lora:
                    self.model.model.model.layers = PipelineModuleLamma(layers = self.model.model.model.layers, num_stages=n_model_pieces)
                else:
                    self.model.model.layers = PipelineModuleLamma(layers = self.model.model.layers, num_stages=n_model_pieces)
        else:
            peft_config = MoLoRAConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=0.1,
                    bias='none',
                    task_type="CAUSAL_LM",
                    moe_num_experts=lora_num
                )
            if use_lora and not use_q_lora:
                self.model = get_moepeft_model(self.model, peft_config)
            elif use_q_lora:

                self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False)

                self.model = get_moepeft_model(self.model, peft_config)
            if model_splitting:
                if use_lora or use_q_lora:
                    self.model.model.model.layers = PipelineModuleLamma(layers = self.model.model.model.layers, num_stages=n_model_pieces)
                else:
                    self.model.model.layers = PipelineModuleLamma(layers = self.model.model.layers, num_stages=n_model_pieces)

    def set_text_lora_group(self, index):
        set_lora_group(self.model, index)

    def forward(self,
            input_embeds = None,
            input_ids = None,
            attn_masks = None,
            full_labels = None,
            past_key_values = None,
            use_cache = False,
            output_attentions = None,
        ):
        '''
        Args:
            input_ids: provide only if already_tokenized=False
            attn_masks: provide only if already_tokenized=False
            text_list: provide only if already_tokenized=True
        '''

        assert ((not(input_embeds is None)) != (not (input_ids is None))), "Only one of input_embeds or input_ids can be provided"

        #input_ids, attn_masks = input_ids.to(self.model.device), attn_masks.to(self.model.device)

        # Verify correct tokenization
        # final_token_idxs = np.arange(len(input_ids)), simcse.get_final_token_indices(attn_masks)
        # assert all(input_ids[final_token_idxs] == self.tokenizer.sep_token_id)

        if input_embeds is not None:
            outputs = self.model(
                inputs_embeds = input_embeds,
                attention_mask = attn_masks,
                output_attentions = output_attentions,
                output_hidden_states = True,
                labels = full_labels,
                past_key_values = past_key_values,
                use_cache = use_cache,
            )
        else:
            outputs = self.model(
                input_ids = input_ids,
                attention_mask = attn_masks,
                output_hidden_states = True,
                labels = full_labels,
                past_key_values = past_key_values,
                use_cache = use_cache,
            )
        # print("########", dist.get_rank())
        # for pn, p in self.model.named_parameters():
            # if p.grad is not None:
            # print(pn, p)
            # if "model.layers.30.mlp.down_proj.weight" == pn:
            #   print("######", pn, p[0,0])

        return outputs
