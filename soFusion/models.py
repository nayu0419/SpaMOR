import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.cluster import KMeans
import torch.optim as optim
from random import shuffle
import pandas as pd
import numpy as np
import scanpy as sc
from soFusion.utils import ZINB
from soFusion.layers import GraphConvolution,SelfAttention,MLP,MGCN,ZINBDecoder,Omics_label_Predictor,MixtureNBLogNormal,BernoulliDecoder,Decoder
import sys

import torch
import torch.nn as nn

class soFusion(nn.Module):
    def __init__(self, nfeatX, nfeatI, hidden_dims, sequencing, weights):
        super(soFusion, self).__init__()
        self.sequencing = sequencing
        self.mgcn = MGCN(nfeatX, nfeatI, hidden_dims)
        self.attlayer1 = SelfAttention(dropout=0.1)
        self.attlayer2 = SelfAttention(dropout=0.1)
        self.fc = nn.Linear(hidden_dims[1] * 2, hidden_dims[1])
        self.mlp = MLP(hidden_dims[1], dropout_rate=0.1)
        self.olp = Omics_label_Predictor(hidden_dims[1])

        # Initialize loss functions once
        self.loss_bec = nn.BCELoss()
        self.loss_ce = nn.CrossEntropyLoss()

        # Initialize decoders based on sequencing types
        self.decoders = nn.ModuleList()
        for idx, seq_type in enumerate(sequencing):
            nfeat = nfeatX if idx == 0 else nfeatI
            if seq_type == "RNA":
                decoder = ZINBDecoder(hidden_dims[1], nfeat)
            elif seq_type == "ADT":
                decoder = MixtureNBLogNormal(hidden_dims[1], nfeat)
            elif seq_type in ["DNA", "ATAC"]:
                decoder = BernoulliDecoder(hidden_dims[1], nfeat)
            else:
                decoder = Decoder(hidden_dims[1], nfeat)
            self.decoders.append(decoder)

        self.weights = weights

    def forward(self, x, i, adj):
        emb_x, emb_i = self.mgcn(x, i, adj)

        # Attention layers
        _, att_emb_x = self.attlayer1(emb_x, emb_x, emb_x)
        _, att_emb_i = self.attlayer2(emb_i, emb_i, emb_i)

        # Omics label prediction
        z_conxy = torch.cat([att_emb_x, att_emb_i], dim=0)
        y_pre = self.olp(z_conxy)

        # MLP outputs
        q_x, q_i = self.mlp(emb_x, emb_i)

        # Consistency information
        emb_con = torch.cat([q_x, q_i], dim=1)
        z_xi = self.fc(emb_con)
        z_I = self.weights[0] * att_emb_x + self.weights[1] * att_emb_i + self.weights[2] * z_xi

        # Initialize losses list
        losses = []
        inputs = [x, i]

        # Calculate losses for each sequencing type
        for idx, (seq_type, decoder, input_data) in enumerate(zip(self.sequencing, self.decoders, inputs)):
            if seq_type == "RNA":
                pi, disp, mean = decoder(z_I)
                loss = ZINB(pi, theta=disp, ridge_lambda=1).loss(input_data, mean, mean=True)
            elif seq_type == "ADT":
                log_likelihood = decoder(z_I, input_data)
                loss = -log_likelihood
            elif seq_type in ["DNA", "ATAC"]:
                pre = decoder(z_I)
                loss = self.loss_bec(pre, input_data)
            else:
                pre = decoder(z_I)
                loss = self.loss_ce(pre, input_data)
            losses.append(loss)

        total_loss = sum(losses)
        print(*losses)

        return z_I, q_x, q_i, total_loss, y_pre


