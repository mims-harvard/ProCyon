import os
import math
from typing import Optional, Tuple, Union, Dict, List

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import LlamaForCausalLM
from procyon.model.encoder import EasyProtSt, EasyESM2

import esm
from peft import get_peft_model, LoraConfig

from transformers.trainer import get_parameter_names, is_sagemaker_mp_enabled, ALL_LAYERNORM_LAYERS
from transformers import Trainer

class Trainer4ProtLlm(transformers.Trainer):

  def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
    super()._load_from_checkpoint(resume_from_checkpoint, model=model)
    if model is None:
      model = self.model
    
    prot2llm_linear_file = os.path.join(resume_from_checkpoint, "prot2llm_linear.bin")
    state_dict = torch.load(prot2llm_linear_file, map_location=torch.device("cpu")).state_dict()
    load_result = model.prot2llm_linear.load_state_dict(state_dict, strict=True)
    self._issue_warnings_after_load(load_result)

    if os.path.exists(os.path.join(resume_from_checkpoint, "llm2prot_linear.bin")):
      llm2prot_linear_file = os.path.join(resume_from_checkpoint, "llm2prot_linear.bin")
      state_dict = torch.load(llm2prot_linear_file, map_location=torch.device("cpu")).state_dict()
      load_result = model.llm2prot_linear.load_state_dict(state_dict, strict=True)
      self._issue_warnings_after_load(load_result)
    
    # load protst if the ckpt exists in the checkpoint folder
    if os.path.exists(os.path.join(resume_from_checkpoint, "protst.bin")):
      protst_file = os.path.join(resume_from_checkpoint, "protst.bin")
      state_dict = torch.load(protst_file, map_location=torch.device("cpu")).state_dict()
      load_result = model.prot_encoder.load_state_dict(state_dict, strict=True)
      self._issue_warnings_after_load(load_result)
  
  def create_optimizer(self):
    
    if is_sagemaker_mp_enabled():
        return super().create_optimizer()
    
    opt_model = self.model

    prot_params = []
    llm_params = []
    prot_params_wo_decay = []
    llm_params_wo_decay = []
    decay_params = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
    decay_params = [n for n in decay_params if "bias" not in n]
    for k, v in opt_model.named_parameters():
      if v.requires_grad:
        if "prot_encoder" in k or "prot2llm_linear" in k or "llm2prot_linear" in k:
          if k in decay_params:
            prot_params.append(v)
          else:
            prot_params_wo_decay.append(v)
        else:
          if k in decay_params:
            llm_params.append(v)
          else:
            llm_params_wo_decay.append(v)
    
    if self.args.lr_ratio != 1.0:
      optimizer_grouped_parameters = [
          {"params": prot_params, "lr": self.args.learning_rate * self.args.lr_ratio, "weight_decay": self.args.weight_decay},
          {"params": prot_params_wo_decay, "lr": self.args.learning_rate * self.args.lr_ratio, "weight_decay": 0.0},
          {"params": llm_params, "lr": self.args.learning_rate, "weight_decay": self.args.weight_decay},
          {"params": llm_params_wo_decay, "lr": self.args.learning_rate, "weight_decay": 0.0},
      ]
    else:
      optimizer_grouped_parameters = [
          {"params": prot_params + llm_params, "lr": self.args.learning_rate, "weight_decay": self.args.weight_decay},
          {"params": prot_params_wo_decay + llm_params_wo_decay, "lr": self.args.learning_rate, "weight_decay": 0.0},
      ]
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

    self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return self.optimizer

class ProtLlm(nn.Module):

  _keys_to_ignore_on_save = None

  def __init__(self, model_args=None, device="cpu"):
    super().__init__()
    self.model_args = model_args
    self.llm = LlamaForCausalLM.from_pretrained(model_args.llm_name_or_path, torch_dtype=torch.bfloat16, device_map=device, use_auth_token=True, low_cpu_mem_usage=True)
    self._llm_config = self.llm.config
    self.llm_tok = AutoTokenizer.from_pretrained(model_args.llm_name_or_path)
    self.llm_tok.pad_token_id = self.llm_tok.eos_token_id

  def init_prot_model(self, data_args=None, device="cpu"):
    loaded = torch.load(os.path.join(data_args.train_data_dir, "protein_cache.pt"))
    self.prot_vectors = loaded["all_outputs"].to(device=device, dtype=torch.bfloat16)
    self.prot_output_size = self.prot_vectors.size(-1)
    self.learn_protst = False
    try:
      self.prot_tok = {nid: i for i, nid in enumerate(loaded["all_nids"])}
    except KeyError:
      self.prot_tok = loaded["nid2index"]

    self.prot2llm_linear = nn.Linear(self.prot_output_size, self.llm.config.hidden_size, dtype=torch.bfloat16)
    self.llm2prot_linear = nn.Linear(self.llm.config.hidden_size, self.prot_output_size, dtype=torch.bfloat16, bias=False)

  def forward(
      self,
      input_ids: Optional[torch.LongTensor] = None,
      past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
      attention_mask: Optional[torch.Tensor] = None,
      inputs_embeds: Optional[torch.Tensor] = None,
      labels: Optional[torch.Tensor] = None,
      output_attentions: Optional[bool] = None,
      output_hidden_states: Optional[bool] = None,
      return_dict: Optional[bool] = None,
      prot_ids = None,
      prot_emb_mask = None,
      prot_labels = None, 
      **unused,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
    
    # if the first tok is a prot, then it wont be in the prot_labels
    prot_vocab_size = self.prot_vectors.size(0)
    prot_embeds = self.prot_vectors[prot_ids]
    prot_embeds = self.prot2llm_linear(prot_embeds)

    # replace some places with protein embeddings
    inputs_embeds = self.llm.model.embed_tokens(input_ids)
    inputs_embeds[prot_emb_mask] = prot_embeds
  
    transformer_outputs = self.llm.model(
      input_ids=None,
      past_key_values=past_key_values,
      attention_mask=attention_mask,
      inputs_embeds=inputs_embeds,
      use_cache=False,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )
    hidden_states = transformer_outputs[0]

    lm_logits = self.llm.lm_head(hidden_states).float()

    # NOTE make some of the labels -100 to avoid predict proteins
    loss = None
    if labels is not None:
      # Shift so that tokens < n predict n
      shift_logits = lm_logits[..., :-1, :].contiguous()
      shift_labels = labels[..., 1:].contiguous()
      batch_size, seq_length, vocab_size = shift_logits.shape
      # Flatten the tokens
      loss_fct = CrossEntropyLoss()
      loss = loss_fct(
          shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
      )
      # protein lm loss:
      #         all      : [prot0] tok1 tok2 [prot1] tok3 [prot2]
      #         prot mask: 1       0    0    1       0    1
      # e.g. shift logits: [prot0] tok1 tok2 [prot1] tok3
      #           targets: tok1    tok2 [prot1] tok3 [prot2]
      #    shit_prot_mask: 0       0    1      0     1
      if prot_labels is None:
        loss_protlm = 0
      else:
        shift_prot_emb_mask = prot_emb_mask[..., 1:]
        shift_hidden_states = hidden_states[:, :-1, :]
        hidden_states4protlm = shift_hidden_states[shift_prot_emb_mask]
        hidden_states4protlm = self.llm2prot_linear(hidden_states4protlm)
        logits4protlm = torch.matmul(hidden_states4protlm, self.prot_vectors.transpose(0, 1))
        loss_protlm = loss_fct(
            logits4protlm.view(-1, prot_vocab_size), prot_labels.view(-1)
        )
      loss = loss + loss_protlm

    if not return_dict:
      output = (lm_logits,) + transformer_outputs[1:]
      return ((loss,) + output) if loss is not None else output

    return CausalLMOutputWithCrossAttentions(
      loss=loss,
      logits=lm_logits,
      past_key_values=transformer_outputs.past_key_values,
      hidden_states=transformer_outputs.hidden_states,
      attentions=transformer_outputs.attentions,
    )
  
  
class ProtLlmForDownstreamTask(ProtLlm):

  def init_prot_model(self, model_args=None, device="cpu"):
    self.prot_output_size = model_args.prot_output_size
    self.prot_tok = esm.Alphabet.from_architecture(model_args.esm_tok_arch_name)
    self.prot2llm_linear = nn.Linear(self.prot_output_size, self.llm.config.hidden_size, dtype=torch.bfloat16)
    self.llm2prot_linear = nn.Linear(self.llm.config.hidden_size, self.prot_output_size, dtype=torch.bfloat16, bias=False)

  def post_init_prot_model(self, model_args=None, device="cpu", learn_protst=False, learn_projector=True):
    if model_args.protein_model_name == "esm":
      protst = EasyESM2(model_args)
    else:
      protst = EasyProtSt(model_args)
      checkpoint = model_args.protein_model_checkpoint
      model_state = torch.load(checkpoint, map_location=torch.device("cpu"))["model"]
      protein_model_state = {}
      for k in model_state.keys():
        if k.startswith("protein_model."):
            protein_model_state[k[14:]] = model_state[k]
      protst.load_state_dict(protein_model_state, strict=False)

    for name, p in self.prot2llm_linear.named_parameters():
      p.requires_grad = learn_projector
    for name, p in protst.named_parameters():
      p.requires_grad = learn_protst
    
    self.learn_projector = learn_projector
    self.learn_protst = learn_protst
    self.prot_output_size = protst.output_dim
    self.prot_encoder = protst
    self.prot_encoder.to(device)
  
  def get_prot_embeds(self, input_ids=None, attention_mask=None, residue_mask=None, **unused):
    if self.learn_protst:
      protst_outputs = self.prot_encoder(input_ids, residue_mask)
    else:
      self.prot_encoder.eval()
      with torch.no_grad():
        protst_outputs = self.prot_encoder(input_ids, residue_mask).detach()

    protst_outputs = protst_outputs.to(dtype=torch.bfloat16)
    prot_embeds = self.prot2llm_linear(protst_outputs)
    return prot_embeds


class ProtLlmForBinaryCls(ProtLlmForDownstreamTask):

  _keys_to_ignore_on_save = None

  def init_cls_head(self, label2tok_ids: List):
    tok_ids4cls = []
    for label, tok_ids in enumerate(label2tok_ids):
      tok_ids4cls.append(tok_ids[0])
    if not hasattr(self.llm.model, 'embed_tokens'):
      cls_emb = self.llm.model.model.embed_tokens.weight.data[tok_ids4cls].clone()
    else:
      cls_emb = self.llm.model.embed_tokens.weight.data[tok_ids4cls].clone()
    self.cls_emb = cls_emb
  
  def forward(
      self,
      input_ids: Optional[torch.LongTensor] = None,
      past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
      attention_mask: Optional[torch.Tensor] = None,
      inputs_embeds: Optional[torch.Tensor] = None,
      labels: Optional[torch.Tensor] = None,
      output_attentions: Optional[bool] = None,
      output_hidden_states: Optional[bool] = None,
      return_dict: Optional[bool] = None,
      prot_input_ids = None,
      prot_attention_mask = None,
      prot_emb_mask = None,
      **unused,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:

    prot_embeds = self.get_prot_embeds(input_ids=prot_input_ids, attention_mask=prot_attention_mask, **unused)
    if not hasattr(self.llm.model, 'embed_tokens'):
      inputs_embeds = self.llm.model.model.embed_tokens(input_ids)
    else:
      inputs_embeds = self.llm.model.embed_tokens(input_ids)
    inputs_embeds[prot_emb_mask] = prot_embeds.to(inputs_embeds.dtype)

    transformer_outputs = self.llm.model(
      input_ids=None,
      past_key_values=past_key_values,
      attention_mask=attention_mask,
      inputs_embeds=inputs_embeds,
      use_cache=False,
      output_attentions=output_attentions,
      output_hidden_states=True,
      return_dict=return_dict,
    )
    
    hidden_states = transformer_outputs.hidden_states[-1]
    hidden_states4cls = hidden_states[:, -1, :]
    if self.cls_emb.device != hidden_states4cls.device:
      self.cls_emb = self.cls_emb.to(device=hidden_states4cls.device)
    logits = torch.matmul(hidden_states4cls, self.cls_emb.transpose(0, 1))
    logits = logits.float()
    
    loss = None
    if labels is not None:
      # Flatten the tokens
      loss_fct = torch.nn.CrossEntropyLoss()
      flatten_logits = logits.view(-1, self.cls_emb.shape[0])
      flatten_labels = labels.view(-1)
      # Enable model parallelism
      flatten_labels = flatten_labels.to(hidden_states4cls.device)
      loss = loss_fct(flatten_logits, flatten_labels)

    if not return_dict:
      output = (logits,) + transformer_outputs[1:]
      return ((loss,) + output) if loss is not None else output

    return CausalLMOutputWithCrossAttentions(
      loss=loss,
      logits=logits,
      past_key_values=transformer_outputs.past_key_values,
      hidden_states=transformer_outputs.hidden_states,
      attentions=transformer_outputs.attentions,
    )
  

class ProtLlmForText2Prot(ProtLlmForDownstreamTask):

  _keys_to_ignore_on_save = None

  def get_original_prot_embeds(self, input_ids=None, attention_mask=None, residue_mask=None, **unused):

    self.prot_encoder.eval()
    with torch.no_grad():
      protst_outputs = self.prot_encoder(input_ids, residue_mask).detach()

    protst_outputs = protst_outputs.to(dtype=torch.bfloat16)
    return protst_outputs

  def get_prot_embeds(self, input_ids=None, attention_mask=None, residue_mask=None, **unused):
    if self.learn_protst:
      protst_outputs = self.prot_encoder(input_ids, residue_mask)
    else:
      self.prot_encoder.eval()
      with torch.no_grad():
        protst_outputs = self.prot_encoder(input_ids, residue_mask).detach()

    prot_embeds = self.prot2llm_linear(protst_outputs)
    return prot_embeds

  def forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    prot_input_ids = None,
    prot_attention_mask = None,
    prot_emb_mask = None,
    has_prot_input=False,
    residue_mask = None,
    **unused, ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:

    if has_prot_input:
      if residue_mask is None:
        residue_mask = prot_attention_mask
      prot_embeds = self.get_prot_embeds(input_ids=prot_input_ids, residue_mask=residue_mask, **unused)

      # replace some places with protein embeddings
      inputs_embeds = self.llm.model.embed_tokens(input_ids)
      inputs_embeds[prot_emb_mask] = prot_embeds.to(inputs_embeds.dtype)

      transformer_outputs = self.llm.model(
        input_ids=None,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        use_cache=False,
        output_attentions=output_attentions,
        output_hidden_states=True,
        return_dict=return_dict,
      )
    else:
      transformer_outputs = self.llm.model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        use_cache=False,
        output_attentions=output_attentions,
        output_hidden_states=True,
        return_dict=return_dict,
      )
    
    hidden_states = transformer_outputs.hidden_states[-1]
    hidden_states4cls = hidden_states[:, -1, :]
    hidden_states4cls = self.llm2prot_linear(hidden_states4cls)

    loss = None
    logits = None
    if labels is not None:
      logits = None
      loss = 0
    
    if not return_dict:
      raise NotImplementedError

    return CausalLMOutputWithCrossAttentions(
      loss=loss,
      logits=logits,
      past_key_values=transformer_outputs.past_key_values,
      hidden_states=hidden_states4cls,
      attentions=transformer_outputs.attentions,
    )