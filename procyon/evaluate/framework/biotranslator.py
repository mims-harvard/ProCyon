import collections
import os
from typing import (
    Dict,
    List,
)

import torch

import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from esm.data import Alphabet
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig

from procyon.data.dataset import AASeqTextUnifiedDataset
from procyon.data.data_utils import DATA_DIR
from procyon.evaluate.framework.args import EvalArgs
from procyon.evaluate.framework.retrieval import AbstractRetrievalModel
from procyon.model.biotranslator_tencoder import HFTextEncoder
from procyon.training.training_args_IT import ModelArgs


def one_hot(seq, start=0, max_len=2000):
    """
    One-Hot encodings of protein sequences,
    this function was copied from DeepGOPlus paper
    :param seq:
    :param start:
    :param max_len:
    :return:
    """
    AALETTER = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
    ]
    AAINDEX = dict()
    for i in range(len(AALETTER)):
        AAINDEX[AALETTER[i]] = i + 1
    onehot = np.zeros((21, max_len), dtype=np.int32)
    l = min(max_len, len(seq))
    for i in range(start, start + l):
        onehot[AAINDEX.get(seq[i - start], 0), i] = 1
    onehot[0, 0:start] = 1
    onehot[0, start + l :] = 1
    return onehot


class NeuralNetwork(nn.Module):
    def __init__(self, bert_name, output_way):
        super(NeuralNetwork, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name)
        Config = AutoConfig.from_pretrained(bert_name)
        Config.attention_probs_dropout_prob = 0.3
        Config.hidden_dropout_prob = 0.3

        self.bert = AutoModel.from_pretrained(bert_name, config=Config)
        self.output_way = output_way

    def forward(self, input_ids, attention_mask, token_type_ids):
        x1 = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        if self.output_way == "cls":
            output = x1.last_hidden_state[:, 0]
        elif self.output_way == "pooler":
            output = x1.pooler_output
        return output


class BioDataEncoder(nn.Module):

    def __init__(
        self,
        feature=["seqs", "network", "description", "expression"],
        hidden_dim=1000,
        seq_input_nc=4,
        seq_in_nc=512,
        seq_max_kernels=129,
        seq_length=2000,
        network_dim=800,
        description_dim=768,
        text_dim=768,
    ):
        """

        :param seq_input_nc:
        :param seq_in_nc:
        :param seq_max_kernels:
        :param dense_num:
        :param seq_length:
        """
        super(BioDataEncoder, self).__init__()
        self.feature = feature
        self.text_dim = text_dim
        if "seqs" in self.feature:
            self.para_conv, self.para_pooling = [], []
            kernels = range(8, seq_max_kernels, 8)
            self.kernel_num = len(kernels)
            for i in range(len(kernels)):
                exec(
                    "self.conv1d_{} = nn.Conv1d(in_channels=seq_input_nc, out_channels=seq_in_nc, kernel_size=kernels[i], padding=0, stride=1)".format(
                        i
                    )
                )
                exec(
                    "self.pool1d_{} = nn.MaxPool1d(kernel_size=seq_length - kernels[i] + 1, stride=1)".format(
                        i
                    )
                )
            self.fc_seq = [
                nn.Linear(len(kernels) * seq_in_nc, hidden_dim),
                nn.LeakyReLU(inplace=True),
            ]
            self.fc_seq = nn.Sequential(*self.fc_seq)
        if "description" in self.feature:
            self.fc_description = [
                nn.Linear(description_dim, hidden_dim),
                nn.LeakyReLU(inplace=True),
            ]
            self.fc_description = nn.Sequential(*self.fc_description)
        if "network" in self.feature:
            self.fc_network = [
                nn.Linear(network_dim, hidden_dim),
                nn.LeakyReLU(inplace=True),
            ]
            self.fc_network = nn.Sequential(*self.fc_network)

    def forward(self, x=None, x_description=None, x_vector=None):
        x_list = []
        features = collections.OrderedDict()
        if "seqs" in self.feature:
            for i in range(self.kernel_num):
                exec("x_i = self.conv1d_{}(x)".format(i))
                exec("x_i = self.pool1d_{}(x_i)".format(i))
                exec("x_list.append(torch.squeeze(x_i).reshape([x.size(0), -1]))")
            features["seqs"] = self.fc_seq(torch.cat(tuple(x_list), dim=1))
        if "description" in self.feature:
            features["description"] = self.fc_description(x_description)
        if "network" in self.feature:
            features["network"] = self.fc_network(x_vector)
        for i in range(len(self.feature)):
            if i == 0:
                x_enc = features[self.feature[0]]
            else:
                x_enc = torch.cat((x_enc, features[self.feature[i]]), dim=1)
        return x_enc


class TextEncoder(nn.Module):
    def __init__(
        self,
        bert_name,
        output_way,
        maxlen=256,
        bert_ckpt=None,
        embed_dim=512,
        proj="mlp",
        pooler_type="cls_pooler",
    ):
        super(TextEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name)

        self.text = HFTextEncoder(
            bert_name,
            output_dim=embed_dim,
            proj=proj,
            pooler_type=pooler_type,
            pretrained=True,
        )
        self.text.device = "cuda"

        self.output_way = output_way
        self.maxlen = maxlen
        if bert_ckpt:
            self.load_state_dict(torch.load(bert_ckpt), strict=False)

        print(torch.load(bert_ckpt).keys())
        for pn, p in self.text.named_parameters():
            print(pn)

    def forward(self, text):
        input_ids = (
            self.tokenizer(
                text,
                max_length=self.maxlen,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            .to(self.text.device)
            .input_ids
        )

        output = self.text(input_ids)
        return output


class BioTranslator(nn.Module):

    def __init__(
        self,
        features,
        hidden_dim,
        seq_input_nc,
        seq_in_nc,
        seq_max_kernels,
        network_dim,
        max_length,
        term_enc_dim,
        bert_name,
        text_output_way,
        text_maxlen,
        bert_ckpt,
        data_ckpt,
        **kwargs,
    ):
        super(BioTranslator, self).__init__()
        self.loss_func = torch.nn.BCELoss()
        self.data_encoder = BioDataEncoder(
            feature=features,
            hidden_dim=hidden_dim,
            seq_input_nc=seq_input_nc,
            seq_in_nc=seq_in_nc,
            seq_max_kernels=seq_max_kernels,
            network_dim=network_dim,
            seq_length=max_length,
            text_dim=term_enc_dim,
        )
        self.text_encoder = TextEncoder(
            bert_name,
            output_way=text_output_way,
            maxlen=text_maxlen,
            bert_ckpt=bert_ckpt,
        )
        self.activation = torch.nn.Sigmoid()
        self.temperature = torch.tensor(0.07, requires_grad=True)
        self.text_dim = term_enc_dim

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.data_encoder = self.data_encoder.to(self.device)
        self.temperature = self.temperature.to(self.device)
        self.activation = self.activation.to(self.device)
        self.text_encoder = self.text_encoder.to(self.device)

        self.data_encoder.load_state_dict(torch.load(data_ckpt), strict=False)
        for pn, p in self.data_encoder.named_parameters():
            print(pn)

    def query_embedding(self, texts):
        texts = self.text_encoder(texts)
        return texts

    def target_embedding(self, input_seq, input_description, input_vector):
        input_seq = np.stack([one_hot(ss) for ss in input_seq])
        input_seq = torch.from_numpy(input_seq).float().to(self.device)
        data_encodings = self.data_encoder(input_seq, input_description, input_vector)

        return data_encodings

    def forward(self, input_seq, input_description, input_vector, texts):
        texts = self.text_encoder(texts)
        input_seq = np.stack([one_hot(ss) for ss in input_seq])

        input_seq = torch.from_numpy(input_seq).float().to(self.device)
        # get textual description encodings
        text_encodings = texts
        # get biology instance encodings
        data_encodings = self.data_encoder(input_seq, input_description, input_vector)
        # compute logits
        return data_encodings


class BioTranslatorRetrievalEval(AbstractRetrievalModel):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device,
    ):
        super().__init__(model_config, eval_args, model_args, device)
        # Model checkpoints can be downloaded from Google Drive here:
        #  https://drive.google.com/drive/folders/1evfcXOVEdaOBoF_ltuijh_2F4hWi8dWe
        model_config["bert_ckpt"] = os.path.join(
            DATA_DIR,
            "model_weights",
            "biotranslator",
            "biotranslator_text_encoder.pth",
        )
        model_config["data_ckpt"] = os.path.join(
            DATA_DIR,
            "model_weights",
            "biotranslator",
            "biotranslator_data_encoder.pth",
        )

        self.model = BioTranslator(**model_config)
        self.protein_tokenizer = Alphabet.from_architecture(
            model_args.protein_tokenizer_name
        )
        self.decode_mapping = {}
        for k in self.protein_tokenizer.tok_to_idx:
            self.decode_mapping[self.protein_tokenizer.tok_to_idx[k]] = k

        self.model.eval()

    def decode(self, indexes):
        return "".join([self.decode_mapping[item.item()] for item in indexes[1:]])

    @torch.no_grad()
    def get_predictions(
        self,
        query_loader: DataLoader,
        target_loader: DataLoader,
        query_order: List,
        target_order: List,
    ) -> torch.Tensor:
        if not isinstance(query_loader.dataset, AASeqTextUnifiedDataset):
            raise ValueError(
                f"biotranslator expected query dataset to be AASeqTextUnifiedDataset, "
                f"got {type(query_loader.dataset)}"
            )

        all_query_embedding = []
        query_ids = []
        all_target_embedding = []
        target_ids = []

        for batch in tqdm(query_loader):
            query_ids += [x[-1] for x in batch["reference_indices"]["input"]["text"]]

            batch_query_ids = [x[-1] for x in batch["input"]["text"]]
            text_batch = [batch["data"]["text"][i] for i in batch_query_ids]

            all_query_embedding.append(self.model.query_embedding(text_batch))

        query_idxs = {query_id: idx for idx, query_id in enumerate(query_ids)}
        rearrange_idxs = [query_idxs[query_id] for query_id in query_order]
        all_query_embedding = torch.cat(all_query_embedding, dim=0)[rearrange_idxs]

        for protein_ids in tqdm(target_loader):
            batch_target_ids = protein_ids.tolist()
            target_ids += batch_target_ids

            seqs = [
                query_loader.collate_fn.aaseq_sequences[x] for x in batch_target_ids
            ]
            all_target_embedding.append(self.model.target_embedding(seqs, None, None))

        target_idxs = {target_id: idx for idx, target_id in enumerate(target_ids)}
        rearrange_idxs = [target_idxs[target_id] for target_id in target_order]
        all_target_embedding = torch.cat(all_target_embedding, dim=0)[rearrange_idxs]

        query_embs_normalized = F.normalize(all_query_embedding)
        target_embs_normalized = F.normalize(all_target_embedding)
        # # Dims like: num_queries X num_targets
        sims = query_embs_normalized @ target_embs_normalized.T

        return sims.detach().cpu().to(torch.float64)
