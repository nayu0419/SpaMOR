# -*- coding: utf-8 -*-
import scipy.sparse as sp
import torch
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from SpaMOR.process import *
from SpaMOR import models
import numpy as np
import torch.optim as optim
from SpaMOR.utils import *
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
import torch.nn as nn
def train(adatalist, adj, sequencing=["unknown", "unknown"], k=10, h=[1000, 1000], n_epochs=200, lr=0.0001,
          key_added='SpaMOR', random_seed=0, l=1, weight_decay=0.0001, a=10, b=100, c=10, d=1, embed=True,
          radius=50, edge_subset_sz=1000000, weight=[1,1,1],device=None):

    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    set_seed(random_seed)
    adata1, adata2 = adatalist
    seq1, seq2 = sequencing

    # def get_features(adata):
    #     if 'highly_variable' in adata.var.columns:
    #         return torch.FloatTensor(adata[:, adata.var['highly_variable']].X.toarray()).to(device)
    #     else:
    #         return torch.FloatTensor(adata.X.toarray()).to(device)

    # def get_features(adata):
    #     if 'highly_variable' in adata.var.columns:
    #         if hasattr(adata[:, adata.var['highly_variable']].X, 'toarray'):
    #             return torch.FloatTensor(adata[:, adata.var['highly_variable']].X.toarray()).to(device)
    #         else:
    #             return torch.FloatTensor(adata[:, adata.var['highly_variable']].X).to(device)
    #     else:
    #         if hasattr(adata.X, 'toarray'):
    #             return torch.FloatTensor(adata.X.toarray()).to(device)
    #         else:
    #             return torch.FloatTensor(adata.X).to(device)

    def get_features(adata):

        if 'highly_variable' in adata.var.columns:

            if hasattr(adata[:, adata.var['highly_variable']].X, 'toarray'):
                return torch.FloatTensor(adata[:, adata.var['highly_variable']].X.toarray()).to(device)
            else:
                return torch.FloatTensor(adata[:, adata.var['highly_variable']].X).to(device)
        else:

            if hasattr(adata.X, 'toarray'):
                return torch.FloatTensor(adata.X.toarray()).to(device)
            else:
                return torch.FloatTensor(adata.X).to(device)

    features1 = get_features(adata1)
    features2 = get_features(adata2)

    N1 = features1.size(0)
    N2 = features2.size(0)

    # Omics label preparation
    ol_x = torch.cat([torch.ones(N1, 1), torch.zeros(N1, 1)], dim=1).to(device)
    ol_y = torch.cat([torch.zeros(N2, 1), torch.ones(N2, 1)], dim=1).to(device)
    ol = torch.cat([ol_x, ol_y], dim=0).to(device)

    # Adjacency matrix normalization
    adj = np.exp(-1 * (adj ** 2) / (2 * (l ** 2)))
    adj = sp.coo_matrix(adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)

    model = models.SpaMOR(nfeatX=features1.shape[1],
                   nfeatI=features2.shape[1],
                   hidden_dims=h,
                   sequencing=sequencing,
                          weights=weight).to(device)

    coords = torch.tensor(adata2.obsm['spatial']).float().to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        z_I, q_x, q_i, loss_distribution, y_pre = model(features1, features2, adj)

        cl_loss = consistency_loss(q_x, q_i)

        # Spatial regularization
        z_dists = torch.cdist(z_I, z_I, p=2)
        z_dists = z_dists / z_dists.max()
        sp_dists = torch.cdist(coords, coords, p=2)
        sp_dists = sp_dists / sp_dists.max()
        n_items = z_I.size(0) ** 2
        reg_loss = ((1.0 - z_dists) * sp_dists).sum() / n_items

        ce_loss = nn.CrossEntropyLoss()(y_pre, ol)

        total_loss = a * loss_distribution + b * cl_loss + c * reg_loss + d * ce_loss
        total_loss.backward()
        optimizer.step()

        print(f"Epoch {epoch:3d}: Distribution Loss={loss_distribution.item():.5f}, "
              f"CL Loss={cl_loss.item():.5f}, Reg Loss={reg_loss.item():.5f}, "
              f"CE Loss={ce_loss.item():.5f}",f"Total Loss={total_loss.item():.5f}")

    model.eval()
    with torch.no_grad():
        z_I, q_x, q_i, _, _ = model(features1, features2, adj)
        emb = z_I.cpu().numpy()
        emb = np.nan_to_num(emb)
        q_x = q_x.cpu().numpy()
        q_i = q_i.cpu().numpy()



    if embed:
        pca = PCA(n_components=30, random_state=42)
        emb = pca.fit_transform(emb)
        q_x = pca.fit_transform(q_x)
        q_i = pca.fit_transform(q_i)

    # Assign embeddings and cluster labels to adata objects
    for adata, q, key in zip([adata1, adata2], [q_x, q_i], sequencing):
        adata.obsm['emb_pca'] = emb
        adata.obsm['emb_q'] = q
        kmeans = KMeans(n_clusters=k, random_state=random_seed).fit(emb)
        kmeans1 = KMeans(n_clusters=k, random_state=random_seed).fit(q)
        adata.obs[key_added] = kmeans.labels_.astype(str)
        adata.obs[key] = kmeans1.labels_.astype(str)

    return adata1, adata2