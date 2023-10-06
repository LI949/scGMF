# scGMF
scGMF: imputation of scRNA-seq data based on generalized matrix factorization

## demo
adata = sc.AnnData(X)     #X: expression matrix \
adata.obs['cl_type'] = Y  #Y: cell type label.If clustering is required, Y is necessary.\
adata.obs['cl_type'] = adata.obs['cl_type'].astype('category')\
n_clusters = len(np.unique(Y))\
adata = preprocess(adata)  # pre-process\
adata, record, epochloss_list = scNCF(adata, n_clusters=n_clusters, cl_type='cl_type',encoder='GMF', decoder='ZINB',mode ='denoise')\

imputed_matrix = np.array(adata.obsm['X_imputed']) # imputed matrix \
embedding_matrix = np.array(adata.obsm['feat'])    # embedding matrix\
