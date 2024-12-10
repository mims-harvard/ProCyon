import os
from deepspeed.runtime.activation_checkpointing import checkpointing
import torch
import torch.nn as nn
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from deepspeed.pipe import PipelineModule
import deepspeed.runtime.utils as ds_utils

from typing import List, Optional, Tuple, Union
from accelerate import Accelerator
# Imports specific to the custom peft lora model
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType

from transformers.models.esm.modeling_esm import EsmModel, EsmEncoder, BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutputWithPastAndCrossAttentions
from transformers.models.esm.modeling_esm import logger

class PipelineModuleEsm(PipelineModule):
    def __init__(self, layers, num_stages=None, topology=None, loss_fn=None, seed_layers=False, seed_fn=None, base_seed=1234, partition_method='parameters', activation_checkpoint_interval=0, activation_checkpoint_func=checkpointing.checkpoint, checkpointable_layers=None):
        super().__init__(layers, num_stages, topology, loss_fn, seed_layers, seed_fn, base_seed, partition_method, activation_checkpoint_interval, activation_checkpoint_func, checkpointable_layers)
        
    def forward(self, forward_input, all_hidden_states, 
                all_self_attentions, all_cross_attentions, next_decoder_cache,
                use_cache, output_hidden_states, head_mask, past_key_values,
                gradient_checkpointing, training, output_attentions,
                attention_mask, encoder_hidden_states, encoder_attention_mask,
                add_cross_attention):
        # We need to offset the seed by the microbatch ID. Save it in a local var to
        # ensure it is preserved in the closure. Otherwise checkpointed forward funcs
        # will see a different offset.
        self.micro_offset += 1
        self.all_hidden_states = all_hidden_states
        self.all_self_attentions = all_self_attentions
        self.all_cross_attentions = all_cross_attentions
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
                    if self.seed_layers:
                        new_seed = (self.base_seed * local_micro_offset) + self.curr_layer
                        if self.seed_fn:
                            self.seed_fn(new_seed)
                        else:
                            ds_utils.set_random_seed(new_seed)

                    if output_hidden_states:
                        self.all_hidden_states = self.all_hidden_states + (inputs, )
                    
                    layer_head_mask = head_mask[self.curr_layer] if head_mask is not None else None
                    past_key_value = past_key_values[self.curr_layer] if past_key_values is not None else None
                    
                    if gradient_checkpointing and training:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs, past_key_value, output_attentions)

                            return custom_forward

                        layer_outputs = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(layer),
                            inputs,
                            attention_mask,
                            layer_head_mask,
                            encoder_hidden_states,
                            encoder_attention_mask,
                        )
                    else:
                        layer_outputs = layer(
                            inputs,
                            attention_mask,
                            layer_head_mask,
                            encoder_hidden_states,
                            encoder_attention_mask,
                            past_key_value,
                            output_attentions,
                        )
                    inputs = layer_outputs[0]
                    if use_cache:
                        self.next_decoder_cache += (layer_outputs[-1], )
                    if output_attentions:
                        self.all_self_attentions = self.all_self_attentions + (layer_outputs[1], )
                        if add_cross_attention:
                            self.all_cross_attentions = self.all_cross_attentions + (layer_outputs[2], )
                        
                    
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
        return x, self.all_cross_attentions, self.all_hidden_states, self.all_self_attentions, self.next_decoder_cache

class EsmEncoderPipline(EsmEncoder):
    def __init__(self, config):
        super().__init__(config)
        # if splitting:
        #     self.layer = PipelineModuleEsm(layers = self.layer, num_stages=n_model_pieces, splitting = splitting)
            # self.layer = nn.ModuleList([EsmLayer(config) for _ in range(config.num_hidden_layers)])
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                    "`use_cache=False`..."
                )
                use_cache = False
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        if type(self.layer) is PipelineModuleEsm:
            hidden_states, all_cross_attentions, all_hidden_states, all_self_attentions, next_decoder_cache = self.layer(hidden_states,
                                                                                                          all_hidden_states, 
                                                                                                          all_self_attentions,
                                                                                                          all_cross_attentions,
                                                                                                          next_decoder_cache,
                                                                                                          use_cache,
                                                                                                          output_hidden_states,
                                                                                                          head_mask,
                                                                                                          past_key_values,
                                                                                                          self.gradient_checkpointing,
                                                                                                          self.training, 
                                                                                                          output_attentions,
                                                                                                          attention_mask,
                                                                                                          encoder_hidden_states,
                                                                                                          encoder_attention_mask,
                                                                                                          self.config.add_cross_attention
                                                                                                          )
        else:
            for i, layer_module in enumerate(self.layer):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = head_mask[i] if head_mask is not None else None
                past_key_value = past_key_values[i] if past_key_values is not None else None

                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, past_key_value, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer_module),
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                    )

                hidden_states = layer_outputs[0]
                if use_cache:
                    next_decoder_cache += (layer_outputs[-1],)
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
                    if self.config.add_cross_attention:
                        all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class ESMPipline(EsmModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.encoder = EsmEncoderPipline(config)

        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )