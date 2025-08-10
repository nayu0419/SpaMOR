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



adata1=anndata.read("Data/Mouse_embryonic_brain/E15/E15_adata_atac.h5ad")
adata2=anndata.read("Data/Mouse_embryonic_brain/E15/E15_adata_rna.h5ad")

# adata1=anndata.read("Data/Mouse_embryonic_brain/E13/E13_adata_atac.h5ad")
# adata2=anndata.read("Data/Mouse_embryonic_brain/E13/E13_adata_rna.h5ad")


adata1.obs['Y'] = adata1.obs['Combined_Clusters_annotation'].astype(str)
adata2.obs['Y'] = adata1.obs['Combined_Clusters_annotation'].astype(str)

spatial_coords = np.column_stack((adata1.obs["array_col"], adata1.obs["array_row"]))
adata1.obsm['spatial'] = spatial_coords
adata2.obsm['spatial'] = spatial_coords
#
coor = pd.DataFrame(adata1.obsm['spatial'])
coor.index = adata1.obs.index
coor.columns = ['imagerow', 'imagecol']
adata1.obs["y_pixel"]=coor['imagerow']
adata1.obs["x_pixel"]=coor['imagecol']


adata1.X = adata1.X.astype(float)
prefilter_genes(adata1, min_cells=3)  # avoiding all genes are zeros
sc.pp.highly_variable_genes(adata1, flavor="seurat_v3", n_top_genes=8000)
sc.pp.normalize_per_cell(adata1)
sc.pp.log1p(adata1)


prefilter_genes(adata2, min_cells=3)  # avoiding all genes are zeros
sc.pp.normalize_per_cell(adata2)
sc.pp.log1p(adata2)
sc.pp.highly_variable_genes(adata2, flavor="seurat_v3", n_top_genes=3000)


adj = calculate_adj_matrix(adata1,"array_col","array_row")
adatalist = [adata1,adata2]
sequencing = ["ATAC","RNA"]

adata1, adata2= train_model.train(adatalist,adj,sequencing, k=12,n_epochs=20,h=[3000,3000], l=1, a=1, b=1, c=1, d=1,
                                  weight=[1,5,5])

obs_df = adata1.obs.dropna()
ARI0 = adjusted_rand_score(obs_df['soFusion'], obs_df['Y'])
print('Adjusted rand index = %.2f' % ARI0)

sc.pl.spatial(adata1, spot_size=1,color='soFusion',title = 'soFusion (ARI={:.2f})'.format(ARI0),save="E15")
sc.pp.neighbors(adata1, use_rep='emb_pca')
sc.tl.umap(adata1)
plt.rcParams["figure.figsize"] = (3, 4)
sc.pl.umap(adata1, color="soFusion", title='soFusion',save="E_soFusion_umap")



