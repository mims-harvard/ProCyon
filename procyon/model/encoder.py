import os
import torch
import esm

from torch import nn
from collections.abc import Sequence
from esm.model.esm2 import ESM2


class TorchDrugMLP(nn.Module):

  def __init__(self, input_dim, hidden_dims, short_cut=False, batch_norm=False, activation="relu", dropout=0):
    super(TorchDrugMLP, self).__init__()

    if not isinstance(hidden_dims, Sequence):
        hidden_dims = [hidden_dims]
    self.dims = [input_dim] + hidden_dims
    self.short_cut = short_cut

    if isinstance(activation, str):
        self.activation = getattr(torch.nn.functional, activation)
    else:
        self.activation = activation
    if dropout:
        self.dropout = nn.Dropout(dropout)
    else:
        self.dropout = None

    self.layers = nn.ModuleList()
    for i in range(len(self.dims) - 1):
        self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
    if batch_norm:
        self.batch_norms = nn.ModuleList()
        for i in range(len(self.dims) - 2):
            self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))
    else:
        self.batch_norms = None

  def forward(self, input):
    layer_input = input

    for i, layer in enumerate(self.layers):
        hidden = layer(layer_input)
        if i < len(self.layers) - 1:
            if self.batch_norms:
                x = hidden.flatten(0, -2)
                hidden = self.batch_norms[i](x).view_as(hidden)
            hidden = self.activation(hidden)
            if self.dropout:
                hidden = self.dropout(hidden)
        if self.short_cut and hidden.shape == layer_input.shape:
            hidden = hidden + layer_input
        layer_input = hidden

    return hidden


class EasyProtSt(torch.nn.Module):

  def __init__(self, model_args):
    super().__init__()
    self.load_esm(model_args.esm_model_name, model_args.esm_model_file_path, alpabet_arch=model_args.esm_tok_arch_name)
    self.last_hidden_dim = 1280
    self.output_dim = 512
    self.activation = 'relu'
    num_mlp_layer = 2
    self.graph_mlp = TorchDrugMLP(
      self.last_hidden_dim,
      [self.last_hidden_dim] * (num_mlp_layer - 1) + [self.output_dim],
      activation=self.activation)
    self.residue_mlp = TorchDrugMLP(
      self.last_hidden_dim,
      [self.last_hidden_dim] * (num_mlp_layer - 1) + [self.output_dim],
      activation=self.activation)
  
  def load_esm(self, model_name, model_file_path, alpabet_arch=None):
    if model_file_path == "":
        self.model = ESM2(alphabet=alpabet_arch)
        del self.model.contact_head
    else:
        raise NotImplementedError
  
  def readout(self, residue_feature: torch.FloatTensor, residue_masks: torch.FloatTensor):
    # return residue_feature[1:].mean(dim=1)
    lens = residue_masks.sum(dim=1)
    pooled = (residue_feature * residue_masks[:,:,None]).sum(dim=1) / lens[:, None]
    return pooled
 
  def forward(self, batch_tokens, residue_masks):
    esm_output = self.model(batch_tokens, repr_layers=[33])
    residue_feature = esm_output["representations"][33]
    if self.training:
       residue_feature += (esm_output["logits"] * 0.0).sum()
    pooled_feature = self.readout(residue_feature, residue_masks)
    pooled_feature = self.graph_mlp(pooled_feature)
    return pooled_feature


class EasyESM2(torch.nn.Module):
   
  def __init__(self, model_args):
    super().__init__()
    self.load_esm(model_args.esm_model_name, model_args.esm_model_file_path, alpabet_arch=model_args.esm_tok_arch_name)
    self.output_dim = 1280
  
  def load_esm(self, model_name, model_file_path, alpabet_arch=None):
    self.model = ESM2(alphabet=alpabet_arch)
    del self.model.contact_head
    loaded = torch.load(model_file_path)
    new_ds = {}
    for k, v in loaded["model"].items():
      if k.startswith("encoder.sentence_encoder."):
        new_ds[k[len("encoder.sentence_encoder."):]] = v
      elif k.startswith("encoder."):
        new_ds[k[len("encoder."):]] = v
      else:
        #  raise ValueError(f"Unexpected key {k}")
         print("[WARNING] Unexpected key %s" % k)
    self.model.load_state_dict(new_ds, strict=True)
  
  def readout(self, residue_feature: torch.FloatTensor, residue_masks: torch.FloatTensor):
    # return residue_feature[1:].mean(dim=1)
    lens = residue_masks.sum(dim=1)
    pooled = (residue_feature * residue_masks[:,:,None]).sum(dim=1) / lens[:, None]
    return pooled

  def forward(self, input_ids=None, residue_mask=None, **unused):
    esm_output = self.model(input_ids, repr_layers=[33])
    residue_feature = esm_output["representations"][33]
    if self.training:
       residue_feature += (esm_output["logits"] * 0.0).sum()
    pooled_feature = self.readout(residue_feature, residue_mask)
    return pooled_feature