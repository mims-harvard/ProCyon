import torch
from torch import nn
import numpy as np

from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.modeling_outputs import BaseModelOutput
from procyon.data.data_utils import DATA_DIR
from procyon.model import simcse
from procyon.model.simcse import Pooler, MLPLayer, Similarity


import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPooling


MODEL_DIR = 'pubmedbert'


class PubMedBERT(nn.Module):
    def __init__(self,
            pooler_type: str,
            model_path = f'{DATA_DIR}/model_weights/{MODEL_DIR}',
        ):
        super(PubMedBERT, self).__init__()
        
        self.model_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.text_tokenizer_name = 'pubmedbert'
        assert self.tokenizer.padding_side == "right"
        assert self.tokenizer.truncation_side == "right"

        self.model = AutoModelForMaskedLM.from_pretrained(model_path)

        self.pooler_type = pooler_type
        self.pooler = Pooler(self.pooler_type)
        if self.pooler_type == "cls":
            self.mlp = MLPLayer(self.model.config.hidden_size)
        self.sim = Similarity()

    def forward(self, input_ids = None, *, attn_masks = None, text_list = None, max_text_len=None, do_pooling=True):
        '''
        Args:
            input_ids: provide only if already_tokenized=False
            attn_masks: provide only if already_tokenized=False
            text_list: provide only if already_tokenized=True
        '''
        # Check preconditions
        if text_list is None:
            assert input_ids is not None and attn_masks is not None 
        else:
            assert input_ids is None and attn_masks is None 
            # Prepare input by prepending cls token and tokenizing
            inputs = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt", max_length=max_text_len)
            input_ids, attn_masks = inputs['input_ids'], inputs['attn_masks']
        
        input_ids, attn_masks = input_ids.to(self.model.device), attn_masks.to(self.model.device)

        # Verify correct tokenization (starts with CLS token)
        assert all(input_ids[:, 0] == 2) # TODO: fix magic number

        outputs = self.model(input_ids = input_ids, attention_mask = attn_masks, output_hidden_states = True)
        
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
    