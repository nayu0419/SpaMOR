# soFusion  
## soFusion: facilitating tissue compartmentalization via spatial multi-omics data fusion 

# Introduction  
To effectively integrate information from different omics for spatial domain identification, we propose a novel spatial multi-omics representation learning method, soFusion, which can be applied to the integration of any two omics.   

soFusion first utilizes Graph Convolutional Networks (GCN) to obtain low-dimensional embeddings of two spatial omics data. To generate the cross-omics joint representation that can fully retain both the commonality and specificity information of different omics, soFusion introduces an intra-omics and inter-omics joint feature learning strategy. Furthermore, to better capture the distribution patterns of different omics data, soFusion designs three specific decoders, which model the count features of transcriptomics, epigenomics, and proteomics data using zero-inflated negative binomial (ZINB) distribution, Bernoulli distribution, and a mixture of two negative binomial distributions, respectively. For the needs of other omics, soFusion also designs a universal decoder based on the fully connected network. Finally, soFusion uses the multi-omics representation, which fully integrates spatial multi-omics information, to identify spatial domains.  

The workflow of soFusion is shown in the following diagram.  

![image](./soFusion.png)

# Installation  
soFusion is implemented using Python 3.7.12 and Pytorch 1.11.0.  

## Requirements  
numpy==1.21.5  
torch==1.11.0  
pandas==1.3.5  
numba==0.55.1  
scanpy==1.9.1  
scikit-learn==1.0.2  
scipy==1.7.3  
anndata==0.8.0  
matplotlib==3.5.2    

## Install soFusion  
```python
git clone https://github.com/sunxue-yy/soFusion.git

cd soFusion

python setup.py build

python setup.py install --user
```


# Datasets    
All datasets used in this study are publicly available. Users can download them from the links below.

  The mouse thymus dataset and human Lymph Node dataset are obtained from https://zenodo.org/records/10362607.   

  The SPOTS mouse spleen dataset is available at GEO with accession code GSE198353.   

  The human tonsil dataset can be accessed at https://doi.org/10.6084/m9.figshare.21623148.v5.   

  The MISAR-seq mouse brain dataset is accessible at the National Genomics Data Center with accession number OEP003285.  

  The spatial ATAC-RNA-seq mouse brain dataset can be found at https://web.atlasxomics.com/visualization/Fan/.  


# Tutorial  
Here, we present two examples to illustrate the application of soFusion for spatial domain identification.   

## Mouse thymus Stereo-CITE-seq dataset  

```python
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from soFusion.utils import *
from soFusion.process import *
from soFusion import train_model
from datetime import datetime
import anndata

adata1=anndata.read("Data/Mouse_Thymus/adata_ADT.h5ad")
adata2=anndata.read("Data/Mouse_Thymus/adata_RNA.h5ad")

prefilter_genes(adata1, min_cells=3)  
sc.pp.normalize_per_cell(adata1)
sc.pp.log1p(adata1)

prefilter_genes(adata2, min_cells=3)  
sc.pp.highly_variable_genes(adata2, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_per_cell(adata2)
sc.pp.log1p(adata2)


adj = calculate_adj_matrix(adata1,"x","y")
adatalist = [adata1,adata2]
sequencing = ["ADT","RNA"]

adata1, adata2= train_model.train(adatalist,adj,sequencing, k=10,n_epochs=50,h=[3000,3000],device='cpu')


sc.pl.spatial(adata1, spot_size=100,color='soFusion',save="soFusion")
sc.pp.neighbors(adata1, use_rep='emb_pca')
sc.tl.umap(adata1)
plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.umap(adata1, color="soFusion", title='soFusion',save="soFusion_umap")
```
## MISAR-seq mouse brain dataset    
```python
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from soFusion.utils import *
from soFusion.process import *
from soFusion import train_model
from datetime import datetime
import anndata

adata1=anndata.read("Data/Mouse_embryonic_brain/E15/E15_adata_atac.h5ad")
adata2=anndata.read("Data/Mouse_embryonic_brain/E15/E15_adata_rna.h5ad")

adata1.obs['Y'] = adata1.obs['Combined_Clusters_annotation'].astype(str)
adata2.obs['Y'] = adata1.obs['Combined_Clusters_annotation'].astype(str)

spatial_coords = np.column_stack((adata1.obs["array_col"], adata1.obs["array_row"]))
adata1.obsm['spatial'] = spatial_coords
adata2.obsm['spatial'] = spatial_coords

coor = pd.DataFrame(adata1.obsm['spatial'])
coor.index = adata1.obs.index
coor.columns = ['imagerow', 'imagecol']
adata1.obs["y_pixel"]=coor['imagerow']
adata1.obs["x_pixel"]=coor['imagecol']


adata1.X = adata1.X.astype(float)
prefilter_genes(adata1, min_cells=3) 
sc.pp.highly_variable_genes(adata1, flavor="seurat_v3", n_top_genes=8000)
sc.pp.normalize_per_cell(adata1)
sc.pp.log1p(adata1)


prefilter_genes(adata2, min_cells=3)  
sc.pp.normalize_per_cell(adata2)
sc.pp.log1p(adata2)
sc.pp.highly_variable_genes(adata2, flavor="seurat_v3", n_top_genes=3000)


adj = calculate_adj_matrix(adata1,"array_col","array_row")
adatalist = [adata1,adata2]
sequencing = ["ATAC","RNA"]

adata1, adata2= train_model.train(adatalist,adj,sequencing, k=14,n_epochs=20,h=[3000,3000],l=0.5,device='cpu')

obs_df = adata1.obs.dropna()
ARI0 = adjusted_rand_score(obs_df['soFusion'], obs_df['Y'])
print('Adjusted rand index = %.2f' % ARI0)

sc.pl.spatial(adata1, spot_size=1,color='soFusion',title='soFusion {}'.format(ARI0),save="E_soFusion")
sc.pp.neighbors(adata1, use_rep='emb_pca')
sc.tl.umap(adata1)
plt.rcParams["figure.figsize"] = (3, 4)
sc.pl.umap(adata1, color="soFusion", title='soFusion',save="E_soFusion_umap")

```
