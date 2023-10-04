import numpy as np
import scanpy as sc
import h5py
import random
import torch
import os
import sys
from train.scNCF import scNCF
from data.data_utils import preprocess,sample
from train.utils import calculate_metric

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 2021
def setup_seed(seed):
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

setup_seed(seed)

######################## Load and process data ##########################

Final_ari_k, Final_nmi_k = [],[]
Final_ari_l, Final_nmi_l = [],[]
Final_cluster_l=[]
dataset = 'PBMC'
data_mat = h5py.File("datasets/10X_PBMC.h5", "r")
if os.path.exists(dataset + "_output") == False:
    os.mkdir(dataset + "_output")

X = np.array(data_mat['X'])
Y = np.array(data_mat['Y'])
X = np.ceil(X).astype(np.int_)
n_clusters = len(np.unique(Y))

t=10
for i in range(t):
    print('----------------times: %d ----------------- '% int(i+1))
    seed_sample = seed + 10*i
    x_sample, y_sample = sample(X,Y,seed_sample)

    adata = sc.AnnData(x_sample)
    adata.obs['cl_type'] = y_sample
    adata.obs['cl_type'] = adata.obs['cl_type'].astype('category')
    adata = preprocess(adata)
    print(adata)
    print("Sparsity: ", np.where(adata.X != 0)[0].shape[0] / (adata.X.shape[0] * adata.X.shape[1]))

    ######################## Perform scNCF ##########################

    Encoder='GMF'   #
    log_interval = 50
    adata, record, epochloss_list = scNCF(adata, n_clusters=n_clusters, cl_type='cl_type',
                          encoder=Encoder, decoder='ZINB', n_epoch=200,log_interval=log_interval)
    print(adata)

    final_ari_k, final_ari_l = record['ari_k'][-1], record['ari_l'][-1]
    final_nmi_k, final_nmi_l = record['nmi_k'][-1], record['nmi_l'][-1]
    final_sil_k, final_sil_l = record['sil_k'][-1], record['sil_l'][-1]
    pred_cluster_l = record['pred_cluster_l'][-1]
    Final_ari_k.append(final_ari_k), Final_nmi_k.append(final_nmi_k)
    Final_ari_l.append(final_ari_l), Final_nmi_l.append(final_nmi_l)
    Final_cluster_l.append(pred_cluster_l)

print(f'Final_nmi_k: {Final_nmi_k}')
print(f'Final_ari_k: {Final_ari_k}')
print(f'Final_nmi_l: {Final_nmi_l}')
print(f'Final_ari_l: {Final_ari_l}')
print(f'Final_cluster_l: {Final_cluster_l}')

######################## Plot results ##########################
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

adata.obs['cl_type'] = adata.obs['cl_type'].astype(str).astype('category')
sc.pp.neighbors(adata, use_rep="feat")
sc.tl.umap(adata)

plt.close('all')
epochs = range(len(epochloss_list))
plt.plot(epochs, epochloss_list, 'b.-', label='Training loss')
plt.plot(epochs, record['all_loss_val'],'r.-', label='Validating loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validating loss')
plt.legend()  
plt.savefig(dataset + '_output/'+ dataset + '_loss.png', dpi=300)

np.savez(dataset + '_output/'+ dataset + "_scGMF.npz",ari_k=Final_ari_k,nmi_k=Final_nmi_k,
         ari_l=Final_ari_l,nmi_l=Final_nmi_l,pred_cluster=Final_cluster_l,
         louvain_cluster=adata.obs['louvain'],kmeans_cluster=adata.obs['kmeans'],
         umap=adata.obsm['X_umap'],true_type=adata.obs['cl_type'])


