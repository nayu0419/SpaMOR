#-*- coding : utf-8 -*-
import os,csv,re
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from soFusion.utils import *
from soFusion.process import *
from soFusion import train_model
from datetime import datetime
import anndata


# adata2=anndata.read("Data/3-Spatial_CITE_seq_human_tonsil/GSM6578062_human_tonsil_gene.h5ad")
# adata1=anndata.read("Data/3-Spatial_CITE_seq_human_tonsil/GSM6578062_human_tonsil_protein.h5ad")

adata2=anndata.read("Data/3-Spatial_CITE_seq_human_tonsil/tonsil_rna_name.h5ad")
adata1=anndata.read("Data/3-Spatial_CITE_seq_human_tonsil/tonsil_adt_name.h5ad")


prefilter_genes(adata1, min_cells=3)  # avoiding all genes are zeros
sc.pp.normalize_per_cell(adata1)
sc.pp.log1p(adata1)

prefilter_genes(adata2, min_cells=3)  # avoiding all genes are zeros
sc.pp.normalize_per_cell(adata2)
sc.pp.log1p(adata2)
sc.pp.highly_variable_genes(adata2, flavor="seurat_v3", n_top_genes=3000)

coor = pd.DataFrame(adata1.obsm['spatial'])
coor.index = adata1.obs.index
coor.columns = ['imagerow', 'imagecol']
adata1.obs["x"] = coor['imagerow']
adata1.obs["y"] = coor['imagecol']

adj = calculate_adj_matrix(adata1,"x","y")
adatalist = [adata1,adata2]
sequencing = ["ADT","RNA"]

# print(type(adata2.X))

adata1, adata2= train_model.train(adatalist,adj,sequencing, k=6, n_epochs=50,h=[3000,3000],l=0.55, a=1, b=1, c=1, d=1,
                                  weight=[5,1,1])


sc.pl.spatial(adata1, spot_size=1,color='soFusion',title='soFusion ',save="tonsil_soFusion")
sc.pp.neighbors(adata1, use_rep='emb_pca')
sc.tl.umap(adata1)
plt.rcParams["figure.figsize"] = (3, 4)
sc.pl.umap(adata1, color="soFusion", title='soFusion',save="tonsil_soFusion_umap")



sc.pl.spatial(adata1, spot_size=1,color='ADT',title='ADT',save="tonsil_ADT")
sc.pp.neighbors(adata1, use_rep='emb_q')
sc.tl.umap(adata1)
plt.rcParams["figure.figsize"] = (3, 4)
sc.pl.umap(adata1, color="ADT", title='ADT',save="tonsil_ADT_umap")


sc.pl.spatial(adata2, spot_size=1,color='RNA',title='RNA',save="tonsil_RNA")
sc.pp.neighbors(adata2, use_rep='emb_q')
sc.tl.umap(adata2)
plt.rcParams["figure.figsize"] = (3, 4)
sc.pl.umap(adata2, color="RNA", title='RNA',save="tonsil_RNA_umap")


adata1.write('tonsil-adt-name.h5ad')
adata2.write('tonsil-rna-name.h5ad')

