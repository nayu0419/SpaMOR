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

# adata1=anndata.read("Data/Mouse_Spleen/adata_Pro.h5ad")
# adata2=anndata.read("Data/Mouse_Spleen/adata_RNA.h5ad")

adata1=anndata.read("Data/Data_SpatialGlue/Dataset2_Mouse_Spleen2/adata_ADT.h5ad")
adata2=anndata.read("Data/Data_SpatialGlue/Dataset2_Mouse_Spleen2/adata_RNA.h5ad")

prefilter_genes(adata1, min_cells=3)  # avoiding all genes are zeros
sc.pp.normalize_per_cell(adata1)
sc.pp.log1p(adata1)

prefilter_genes(adata2, min_cells=3)  # avoiding all genes are zeros
sc.pp.highly_variable_genes(adata2, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_per_cell(adata2)
sc.pp.log1p(adata2)

coor = pd.DataFrame(adata1.obsm['spatial'])
coor.index = adata1.obs.index
coor.columns = ['imagerow', 'imagecol']
adata1.obs["x"] = coor['imagerow']
adata1.obs["y"] = coor['imagecol']

adj = calculate_adj_matrix(adata1,"x","y")
adatalist = [adata1,adata2]
sequencing = ["ADT","RNA"]
# sequencing = ["DNA","sss"]
adata1, adata2= train_model.train(adatalist,adj,sequencing, k=5, n_epochs=50,h=[3000,3000],a=1, b=1, c=1, d=1,weight=[1,1,1])

sc.pl.spatial(adata1, spot_size=1.5,color='soFusion',save="spleen2_soFusion")
sc.pp.neighbors(adata1, use_rep='emb_pca')
sc.tl.umap(adata1)
plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.umap(adata1, color="soFusion", title='soFusion',save="spleen2_soFusion_umap")

sc.pl.spatial(adata1, spot_size=1.5,color='ADT',save="spleen2_ADT")
sc.pp.neighbors(adata1, use_rep='emb_q')
sc.tl.umap(adata1)
plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.umap(adata1, color="ADT", title='ADT',save="spleen2_ADT_umap")

sc.pl.spatial(adata2, spot_size=1.5,color='RNA',save="spleen2_RNA")
sc.pp.neighbors(adata2, use_rep='emb_q')
sc.tl.umap(adata2)
plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.umap(adata2, color="RNA", title='RNA',save="spleen2_RNA_umap")

# adata1.write('spleen1_ADT.h5ad')
# adata2.write('spleen1_RNA.h5ad')