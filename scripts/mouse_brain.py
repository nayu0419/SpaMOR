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

#
# adata1=anndata.read("Data/4-Mouse_Brain/adata_peaks_normalized.h5ad")
# adata2=anndata.read("Data/4-Mouse_Brain/adata_RNA.h5ad")

# adata1=anndata.read("Data/Data_SpatialGlue/Dataset8_Mouse_Brain_H3K4me3/adata_peaks_normalized.h5ad")
# adata2=anndata.read("Data/Data_SpatialGlue/Dataset8_Mouse_Brain_H3K4me3/adata_RNA.h5ad")

# adata1=anndata.read("Data/Data_SpatialGlue/Dataset9_Mouse_Brain_H3K27ac/adata_peaks_normalized.h5ad")
# adata2=anndata.read("Data/Data_SpatialGlue/Dataset9_Mouse_Brain_H3K27ac/adata_RNA.h5ad")

adata1=anndata.read("Data/Data_SpatialGlue/Dataset10_Mouse_Brain_H3K27me3/adata_peaks_normalized.h5ad")
adata2=anndata.read("Data/Data_SpatialGlue/Dataset10_Mouse_Brain_H3K27me3/adata_RNA.h5ad")


prefilter_genes(adata1, min_cells=3)  # avoiding all genes are zeros
sc.pp.highly_variable_genes(adata1, flavor="seurat_v3", n_top_genes=8000)
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
sequencing = ["ATAC","RNA"]


adata1, adata2= train_model.train(adatalist,adj,sequencing, k=18,n_epochs=50,h=[3000,3000], l=0.8, a=1, b=1, c=1, d=1,weight=[1,10,5])



sc.pl.spatial(adata1, spot_size=1,color='soFusion',save="brain0_soFusion")
sc.pp.neighbors(adata1, use_rep='emb_pca')
sc.tl.umap(adata1)
plt.rcParams["figure.figsize"] = (3, 4)
sc.pl.umap(adata1, color="soFusion", title='soFusion',save="brain0_soFusion_umap")


adata1.write('adata_ATAC_brain.h5ad')
adata2.write('adata_RNA_brain.h5ad')

