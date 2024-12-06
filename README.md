# SpaMOR  
## SpaMOR: Integrating spatial multi-omics representation for spatial domain identification 

# Introduction  
To effectively integrate information from different omics for spatial domain identification, we propose a novel spatial multi-omics representation learning method, SpaMOR, which can be applied to the integration of any two omics.   

SpaMOR first utilizes Graph Convolutional Networks (GCN) to obtain low-dimensional embeddings of two spatial omics data. To generate the cross-omics joint representation that can fully retain both the commonality and specificity information of different omics, SpaMOR introduces an intra-omics and inter-omics joint feature learning strategy. Furthermore, to better capture the distribution patterns of different omics data, SpaMOR designs three specific decoders, which model the count features of transcriptomics, epigenomics, and proteomics data using zero-inflated negative binomial (ZINB) distribution, Bernoulli distribution, and a mixture of two negative binomial distributions, respectively. For the needs of other omics, SpaMOR also designs a universal decoder based on the fully connected network. Finally, SpaMOR uses the multi-omics representation, which fully integrates spatial multi-omics information, to identify spatial domains.  

The workflow of SpaMOR is shown in the following diagram.  

# Installation  
SpaGRA is implemented using Python 3.7.12 and Pytorch 1.11.0.  

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

# Datasets    
All datasets used in this study are publicly available. Users can download them from the links below.

The mouse thymus dataset and human Lymph Node dataset are obtained from https://zenodo.org/records/10362607.   
The SPOTS mouse spleen dataset is available at GEO with accession code GSE198353.   
The human tonsil dataset can be accessed at https://doi.org/10.6084/m9.figshare.21623148.v5.   
The MISAR-seq mouse brain dataset is accessible at the National Genomics Data Center with accession number OEP003285.  
The spatial ATAC-RNA-seq mouse brain dataset can be found at https://web.atlasxomics.com/visualization/Fan/.
