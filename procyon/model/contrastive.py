import math
from collections import defaultdict
import os
import csv
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist

from torch.distributed.nn.functional import all_gather as all_gather_with_backprop
# Find doc here, not on PyTorch doc site: https://github.com/pytorch/pytorch/blob/472500e32a94c15a630361414004398feaeadbd9/torch/distributed/nn/functional.py#L104

import numpy as np
import copy

'''
A stripped-down version of contrastive learning heads considered for retrieval learning in TxPLM
'''

class InfoNCE(nn.Module):
    def __init__(self,
            input_embed_dim,
            use_projection = True,
        ):
        super().__init__()
        
        self.input_embed_dim = input_embed_dim
        self.use_projection = use_projection

        self.temperature = nn.Parameter(0.07 * torch.ones([]))
        if self.use_projection: 
            self.ret_projection = ProjectionHead(self.input_embed_dim, self.input_embed_dim)
            self.protein_z_projection = ProjectionHead(self.input_embed_dim, self.input_embed_dim)
        else:
            self.ret_projection, self.protein_z_projection = None, None

    def forward(self, c_input):

        # https://github.com/salesforce/BLIP/blob/main/models/blip_retrieval.py#L73
        with torch.no_grad(): 
            self.temperature.clamp_(0.001, 0.5)

        if self.use_projection:
            pos_s_z, neg_s_z = self.protein_z_projection(c_input["positive"]["sequence"]), self.protein_z_projection(c_input["negative"]["sequence"])
            pos_t_z, neg_t_z = self.ret_projection(c_input["positive"]["text"]), self.ret_projection(c_input["negative"]["text"])
        else:
            pos_s_z, pos_t_z = c_input["positive"]["sequence"], c_input["positive"]["text"]
            neg_s_z, neg_t_z = c_input["negative"]["sequence"], c_input["negative"]["text"]

        # Extract indices:
        pos_inds_t, neg_inds_t = c_input["positive"]["indices_text"], c_input["negative"]["indices_text"]
        pos_inds_s, neg_inds_s = c_input["positive"]["indices_sequence"], c_input["negative"]["indices_sequence"]

        # Compute (sequence -> text)
        sim_pos_st = F.cosine_similarity(pos_s_z, pos_t_z) / self.temperature
        sim_neg_st = F.cosine_similarity(neg_s_z, neg_t_z) / self.temperature
        
        # Construct matrix:
        neg_sim_mat_st = torch.cat([
            torch.cat([sim_pos_st[i].unsqueeze(0), sim_neg_st[neg_inds_s == si], sim_neg_st[neg_inds_t == ti]]) 
                for i, (si, ti) in enumerate(zip(pos_inds_s, pos_inds_t))], 
            dim = 0)
        # Idea in above line: need a matrix where each row is [p_i, n_i[0], n_i[1], ..., n_i[n_n]], where p_i's are positive scores, n_i's are negative scores
        #   - Can then compute similarity via cross entropy against "labels" which are the 0th index in each row
        #   - Each row contains the positive score for a pair and then it's corresponding negatives, but we have to check for negatives on both sides of the tail
        # This line is in roughly O(n_p * n_n) time, where n_n == the number of negatives since the boolean index must search for the index within the negative similiarty scores
        # Should be fairly fast, but we could check it later

        # Use logsumexp, log transform to efficiently compute
        loss_st = -1.0 * (sim_pos_st - neg_sim_mat_st.logsumexp(dim=-1)).mean() # Equivalent to 1/N \sum log[ {e^(sim / \tau)} / { \sum{all sims} }]

        # Compute the other way (text -> sequence)
        sim_pos_ts = F.cosine_similarity(pos_t_z, pos_s_z) / self.temperature
        sim_neg_ts = F.cosine_similarity(neg_t_z, neg_s_z) / self.temperature
        
        # Construct matrix:
        neg_sim_mat_ts = torch.cat([
            torch.cat([sim_pos_ts[i].unsqueeze(0), sim_neg_ts[neg_inds_s == si], sim_neg_ts[neg_inds_t == ti]]) 
                for i, (si, ti) in enumerate(zip(pos_inds_s, pos_inds_t))], 
            dim = 0)

        # Use logsumexp, log transform to efficiently compute
        loss_ts = -1.0 * (sim_pos_ts - neg_sim_mat_ts.logsumexp(dim=-1)).mean() # Equivalent to 1/N \sum log[ {e^(sim / \tau)} / { \sum{all sims} }]

        # sim_mat = sim_mat.softmax(dim = -1)
        # loss = F.cross_entropy(sim_mat, torch.zeros(sim_mat.shape[0]).to(sim_mat.device))

        # Average losses together:
        loss = (loss_st + loss_ts) / 2.0

        return loss

class InfoNCEInBatch(nn.Module):
    '''
    02/12/2024: WE USE THIS ONE

    InfoNCE loss with in-batch negatives automatically built in
    '''
    def __init__(self,
            input_embed_dim,
            use_projection = True,
            all_gather_version = False,
        ):
        super().__init__()
        
        self.input_embed_dim = input_embed_dim
        self.use_projection = use_projection
        self.all_gather_version = all_gather_version

        self.temperature = nn.Parameter(0.07 * torch.ones([]))
        #self.temperature = 0.25
        if self.use_projection: 
            self.ret_projection = ProjectionHead(self.input_embed_dim, self.input_embed_dim)
            self.protein_z_projection = ProjectionHead(self.input_embed_dim, self.input_embed_dim)
        else:
            self.ret_projection, self.protein_z_projection = None, None

    def forward(self, c_input, negatives_mask = None):

        # https://github.com/salesforce/BLIP/blob/main/models/blip_retrieval.py#L73
        with torch.no_grad(): 
            self.temperature.clamp_(0.001, 0.5)

        if self.use_projection:
            pos_s_z = self.protein_z_projection(c_input["positive"]["sequence"])
            pos_t_z = self.ret_projection(c_input["positive"]["text"])
        else:
            pos_s_z, pos_t_z = c_input["positive"]["sequence"], c_input["positive"]["text"]

        # Compute both ways: See: https://github.com/kohjingyu/fromage/blob/51fb06acf72f7abacd6da49cbc8c09a56826fbd0/main.py#L487

        # Normalize:
        #ret_copy_pos_s_z = pos_s_z.detach().clone()
        #ret_copy_pos_t_z = pos_t_z.detach().clone()

        pos_s_z = F.normalize(pos_s_z, dim = -1)
        pos_t_z = F.normalize(pos_t_z, dim = -1)

        if self.all_gather_version and torch.distributed.is_initialized():
            #torch.distributed.monitored_barrier()
            # Perform all_gather across GPUs like https://github.com/DeepGraphLearning/ProtST/blob/db53a76ed2430eb66dd9c8134ace99fd60980fb3/protst/task.py#L69
            all_s_z = all_gather_with_backprop(pos_s_z)
            all_s_z = torch.cat(all_s_z, dim=0)

            #torch.distributed.monitored_barrier()

            all_t_z = all_gather_with_backprop(pos_t_z)
            all_t_z = torch.cat(all_t_z, dim=0)

            #torch.distributed.monitored_barrier()

            # Local batch sizes, i.e., micro-batch size on each GPU
            local_batch_size_s = pos_s_z.shape[0]
            local_batch_size_t = pos_t_z.shape[0]

            # print("Global batch size: {}".format(all_s_z.shape[0]))
            # print("My rank: {}".format(dist.get_rank()))

            # Compute similarities against all negatives:
            sim_st = torch.matmul(pos_s_z, all_t_z.t()) / self.temperature
            sim_ts = torch.matmul(pos_t_z, all_s_z.t()) / self.temperature

            #Set rank-aware targets:
            # dist.get_rank() should get global rank
            target_st = dist.get_rank() * local_batch_size_s + torch.arange(pos_s_z.shape[0]).to(pos_s_z.device)
            target_ts = dist.get_rank() * local_batch_size_t + torch.arange(pos_t_z.shape[0]).to(pos_t_z.device)
        
        else:
            # If not gathering, computing similarities is simpler
            # Similarity matrix:
            sim_st = torch.matmul(pos_s_z, pos_t_z.t()) / self.temperature
            sim_ts = sim_st.t()
            # However, do need to set the targets:
            target_st = torch.arange(sim_st.shape[0]).to(sim_st.device)
            target_ts = torch.arange(sim_ts.shape[0]).to(sim_ts.device)

        if negatives_mask is not None:
            #print("Negatives_mask", negatives_mask.shape)
            #print("SimST", sim_st.shape)
            #print("SimTS", sim_ts.shape)
            #assert (negatives_mask.shape[0] == sim_st.shape[0]) and (negatives_mask.shape[1] == sim_st.shape[1])

            # dist.get_rank() gets global rank
            neg_mask_index = dist.get_rank() * local_batch_size_s + torch.arange(sim_st.shape[0]).to(negatives_mask.device)

            # if dist.get_rank() == 0:
            #     print("NEG MASK")
            #     print(negatives_mask)

            # Mask out negatives
            # sim_st = sim_st[negatives_mask[neg_mask_index]]
            # sim_ts = sim_ts[negatives_mask[neg_mask_index]]
            sim_st = sim_st * negatives_mask[neg_mask_index].float()
            sim_ts = sim_ts * negatives_mask[neg_mask_index].float()

        # L_st = F.cross_entropy(sim_st, torch.arange(sim_st.shape[0]).to(sim_st.device)) # Computes similarities along the diagonal
        # L_ts = F.cross_entropy(sim_ts, torch.arange(sim_ts.shape[0]).to(sim_ts.device))
        L_st = F.cross_entropy(sim_st, target_st) # Computes similarities along the diagonal
        L_ts = F.cross_entropy(sim_ts, target_ts)

        
        return (L_st + L_ts) / 2.0 #ret_copy_pos_s_z, ret_copy_pos_t_z

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=100,
        dropout=0.5
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        # x = self.layer_norm(x)
        return x

class MaxMarginContrastiveLoss(nn.Module):
    # TODO: Needs some work
    def __init__(self, protein_embed_dim, text_embed_dim, 
            margin = 0.0, use_projection = False):
        super().__init__()
        self.margin = margin
        self.use_projection = use_projection
        self.protein_embed_dim = protein_embed_dim
        self.text_embed_dim = text_embed_dim

        self.scorer = nn.Linear(protein_embed_dim * 2, 1)

    def forward(self, c_input):
        if self.use_projection:
            pos_s_z, neg_s_z = self.protein_z_projection(c_input["positive"]["sequence"]), self.protein_z_projection(c_input["negative"]["sequence"])
            pos_t_z, neg_t_z = self.ret_projection(c_input["positive"]["text"]), self.ret_projection(c_input["negative"]["text"])
        else:
            pos_s_z, pos_t_z = c_input["positive"]["sequence"], c_input["positive"]["text"]
            neg_s_z, neg_t_z = c_input["negative"]["sequence"], c_input["negative"]["text"]

        # pos_s_z = F.normalize(pos_s_z, dim = -1)
        # pos_t_z = F.normalize(pos_t_z, dim = -1)
        
        # sim_st = torch.matmul(pos_s_z, pos_t_z.t()) + self.margin

        pos_scores = self.scorer(torch.cat([pos_s_z, pos_t_z], dim = -1))
        neg_scores = self.scorer(torch.cat([neg_s_z, neg_t_z], dim = -1))

        # From get_kepler_loss:
        positive_loss = F.logsigmoid(pos_scores + self.margin).mean()
        negative_loss = F.logsigmoid(neg_scores + self.margin).mean()

        loss = (-1.0 * positive_loss + negative_loss) # Loss is separable, so fine to reduce final losses like this

        # in-place operations - positives are on the diagonal
        # sim_st[torch.arange(sim_st.shape[0]),torch.arange(sim_st.shape[0])] *= -1.0

        # loss = sim_st.mean() # Reduce via mean

        return loss

        