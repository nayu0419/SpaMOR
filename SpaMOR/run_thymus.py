#-*- coding : utf-8 -*-
import os,csv,re
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from SpaMOR.utils import *
from SpaMOR.process import *
from SpaMOR import train_model
from datetime import datetime
import anndata

adata1=anndata.read("Data/Mouse_Thymus/adata_ADT.h5ad")
adata2=anndata.read("Data/Mouse_Thymus/adata_RNA.h5ad")

prefilter_genes(adata1, min_cells=3)  # avoiding all genes are zeros
# prefilter_specialgenes(adata)
# sc.pp.highly_variable_genes(adata1, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_per_cell(adata1)
sc.pp.log1p(adata1)

prefilter_genes(adata2, min_cells=3)  # avoiding all genes are zeros
# prefilter_specialgenes(adata)
sc.pp.highly_variable_genes(adata2, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_per_cell(adata2)
sc.pp.log1p(adata2)



adj = calculate_adj_matrix(adata1,"x","y")
adatalist = [adata1,adata2]
sequencing = ["ADT","RNA"]
# sequencing = ["DNA","sss"]
adata1, adata2= train_model.train(adatalist,adj,sequencing, k=10,n_epochs=200,h=[3000,3000],a=1, b=1, c=1, d=1,weight=[1,1,1],device='cpu')
#a=1, b=5, c=1, d=5
sc.pl.spatial(adata1, spot_size=100,color='SpaMOR',save="SpaMOR")
sc.pp.neighbors(adata1, use_rep='emb_pca')
sc.tl.umap(adata1)
plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.umap(adata1, color="SpaMOR", title='SpaMOR',save="SpaMOR_umap")

sc.pl.spatial(adata1, spot_size=100,color='ADT',save="ADT")
sc.pp.neighbors(adata1, use_rep='emb_q')
sc.tl.umap(adata1)
plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.umap(adata1, color="ADT", title='ADT',save="ADT_umap")

sc.pl.spatial(adata2, spot_size=100,color='RNA',save="RNA")
sc.pp.neighbors(adata2, use_rep='emb_q')
sc.tl.umap(adata2)
plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.umap(adata2, color="RNA", title='RNA',save="RNA_umap")

adata1.write('adata_A1_ADT.h5ad')
adata2.write('adata_A1_RNA.h5ad')