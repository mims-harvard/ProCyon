import warnings

import torch

from torch import nn

from torchdrug import (
    core,
    layers,
    models,
)
from torchdrug.core import Registry as R
from torchdrug.layers import functional

from transformers import BertForMaskedLM, BertTokenizer

############################################
#### Protein Model
############################################
@R.register("models.PretrainESM")
class PretrainESM(models.ESM):

    def __init__(self, path, model="ESM-1b",
        output_dim=512, num_mlp_layer=2, activation='relu',
        readout="mean", mask_modeling=False, use_proj=True):
        super(PretrainESM, self).__init__(path, model, readout)
        self.mask_modeling = mask_modeling

        self.last_hidden_dim = self.output_dim
        self.output_dim = output_dim if use_proj else self.last_hidden_dim
        self.num_mlp_layer = num_mlp_layer
        self.activation = activation
        self.use_proj = use_proj

        self.graph_mlp = layers.MLP(self.last_hidden_dim,
                                    [self.last_hidden_dim] * (num_mlp_layer - 1) + [output_dim],
                                    activation=self.activation)
        self.residue_mlp = layers.MLP(self.last_hidden_dim,
                                      [self.last_hidden_dim] * (num_mlp_layer - 1) + [output_dim],
                                      activation=self.activation)

    def forward(self, graph, input, all_loss=None, metric=None):
        input = graph.residue_type
        if self.mask_modeling:
            non_mask = ~(input == self.alphabet.mask_idx)
            input[non_mask] = self.mapping[input[non_mask]]
        else:
            input = self.mapping[input]
        size = graph.num_residues
        if (size > self.max_input_length).any():
            warnings.warn("ESM can only encode proteins within %d residues. Truncate the input to fit into ESM."
                          % self.max_input_length)
            starts = size.cumsum(0) - size
            size = size.clamp(max=self.max_input_length)
            ends = starts + size
            mask = functional.multi_slice_mask(starts, ends, graph.num_residue)
            input = input[mask]
            graph = graph.subresidue(mask)
        size_ext = size
        if self.alphabet.prepend_bos:
            bos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * self.alphabet.cls_idx
            input, size_ext = functional._extend(bos, torch.ones_like(size_ext), input, size_ext)
        if self.alphabet.append_eos:
            eos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * self.alphabet.eos_idx
            input, size_ext = functional._extend(input, size_ext, eos, torch.ones_like(size_ext))
        input = functional.variadic_to_padded(input, size_ext, value=self.alphabet.padding_idx)[0]

        output = self.model(input, repr_layers=[33])
        residue_feature = output["representations"][33]

        residue_feature = functional.padded_to_variadic(residue_feature, size_ext)
        starts = size_ext.cumsum(0) - size_ext
        if self.alphabet.prepend_bos:
            starts = starts + 1
        ends = starts + size
        mask = functional.multi_slice_mask(starts, ends, len(residue_feature))
        residue_feature = residue_feature[mask]
        graph_feature = self.readout(graph, residue_feature)

        if self.use_proj:
            graph_feature = self.graph_mlp(graph_feature)
            residue_feature = self.residue_mlp(residue_feature)

        return {
            "graph_feature": graph_feature,
            "residue_feature": residue_feature,
        }



############################################
#### Text Model
############################################


class HuggingFaceModel(nn.Module, core.Configurable):
    """
    Pretrained models from HuggingFace.
    """

    huggingface_card = {
        "PubMedBERT-abs": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        "PubMedBERT-full": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    }

    huggingface_model_type = {
        "PubMedBERT-abs": "BertForMaskedLM",
        "PubMedBERT-full": "BertForMaskedLM"
    }
    huggingface_tokenizer_type = {
        "PubMedBERT-abs": "BertTokenizer",
        "PubMedBERT-full": "BertTokenizer"
    }

    huggingface_tokenizer_arguments = {
        "BertTokenizer": "https://github.com/huggingface/transformers/blob/5987c637ee68aacd78945697c636abcd204b1997/src/transformers/models/bert/tokenization_bert.py#L137",
    }

    def __init__(self):
        super(HuggingFaceModel, self).__init__()

    def _build_from_huggingface(self, name, path="", local_files_only=True):

        if name not in self.huggingface_card:
            raise NotImplementedError
        card = self.huggingface_card[name]
        model_type = self.huggingface_model_type[name]
        tokenizer_type = self.huggingface_tokenizer_type[name]

        if path != "":
            model = eval(model_type).from_pretrained(path, local_files_only=local_files_only)
            tokenizer = eval(tokenizer_type).from_pretrained(path, local_files_only=local_files_only)
        else:
            model = eval(model_type).from_pretrained(card, local_files_only=local_files_only)
            tokenizer = eval(tokenizer_type).from_pretrained(card, local_files_only=local_files_only)

        return model, tokenizer

@R.register("models.PubMedBERT")
class PubMedBERT(HuggingFaceModel):
    """
    PubMedBERT encodes text description for proteins, starting
    from the pretrained weights provided in `https://microsoft.github.io/BLURB/models.html`

    Parameters:
        model (string, optional): model name. Available model names are ``PubMedBERT-abs``
            and ``PubMedBERT-full``. They differentiate from each other by training corpus,
            i.e., abstract only or abstract + full text.

    """

    last_hidden_dim = {
        "PubMedBERT-abs": 768,
        "PubMedBERT-full": 768,
    }

    def __init__(
        self,
        model,
        path="",
        output_dim=512,
        num_mlp_layer=2,
        activation='relu',
        readout='mean',
        attribute=["prot_name", "function", "subloc", "similarity"],
    ):
        super(PubMedBERT, self).__init__()

        _model, _tokenizer = self._build_from_huggingface(model, path=path)

        self.last_hidden_dim = self.last_hidden_dim[model]
        self.output_dim = output_dim
        self.num_mlp_layer = num_mlp_layer

        self.model = _model
        self.tokenizer = _tokenizer
        self.pad_idx = self.tokenizer.pad_token_id
        self.sep_idx = self.tokenizer.sep_token_id
        self.cls_idx = self.tokenizer.cls_token_id
        self.mask_idx = self.tokenizer.mask_token_id

        self.activation = activation
        self.readout = readout
        self.attribute = attribute

        self.text_mlp = layers.MLP(self.last_hidden_dim,
                                   [self.last_hidden_dim] * (self.num_mlp_layer - 1) + [self.output_dim],
                                   activation=self.activation)
        self.word_mlp = layers.MLP(self.last_hidden_dim,
                                   [self.last_hidden_dim] * (self.num_mlp_layer - 1) + [self.output_dim],
                                   activation=self.activation)

    def _combine_attributes(self, graph, version=0):

        num_sample = len(graph)
        cls_ids = torch.ones(num_sample, dtype=torch.long, device=self.device).unsqueeze(1) * self.cls_idx
        sep_ids = torch.ones(num_sample, dtype=torch.long, device=self.device).unsqueeze(1) * self.sep_idx

        if version == 0:
            # [CLS] attr1 [PAD] ... [PAD] [SEP] attr2 [PAD] ... [PAD] [SEP] attrn [PAD] ... [PAD]
            input_ids = [cls_ids]
            for k in self.attribute:
                input_ids.append(graph.data_dict[k].long())
                input_ids.append(sep_ids)
            input_ids = torch.cat(input_ids[:-1], dim=-1)

        else:
            raise NotImplementedError

        return input_ids, input_ids != self.pad_idx

    def forward(self, graph, all_loss=None, metric=None, input_ids=None, attention_mask=None):
        if input_ids is None or attention_mask is None:
            input_ids, attention_mask = self._combine_attributes(graph)
        model_inputs = {"input_ids": input_ids,
                        "token_type_ids": torch.zeros_like(input_ids),
                        "attention_mask": attention_mask
                        }
        model_outputs = self.model.bert(**model_inputs)
        if self.readout == "mean":
            is_special = (input_ids == self.cls_idx) | (input_ids == self.sep_idx) | (input_ids == self.pad_idx)
            text_mask = (~is_special).to(torch.float32).unsqueeze(-1)
            output = (model_outputs.last_hidden_state * text_mask).sum(1) / (text_mask.sum(1) + 1.0e-6)
        elif self.readout == "cls":
            output = model_outputs.last_hidden_state[:, 0, :]
        else:
            raise NotImplementedError

        output = self.text_mlp(output)
        word_output = self.word_mlp(model_outputs.last_hidden_state)

        return {
            "text_feature": output,
            "word_feature": word_output
        }
