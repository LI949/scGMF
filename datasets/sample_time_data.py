import random
import numpy as np
import pandas as pd
import h5py
import torch
import os
import sys
import scipy as sp
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

filename = "datasets/20k_PBMC_3p_HT_nextgem_Chromium_X_filtered_feature_bc_matrix.h5"
with h5py.File(filename, "r") as f:
    exprs_handle = f["matrix"]
    if isinstance(exprs_handle, h5py.Group):
        mat = sp.sparse.csc_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                    exprs_handle["indptr"][...]), shape=exprs_handle["shape"][...] )

X = np.array(mat.toarray()).T
X = np.delete(X,np.where(~X.any(axis=0))[0], axis=1) #delete genes which not express
#
seed = 2021
setup_seed(seed)
filepath = 'datasets/time_data/'
if not os.path.exists(filepath):
    os.makedirs(filepath)

for cell in [2000, 4000, 8000, 12000, 16000]:
    setup_seed(seed)

    path = filepath + str(cell) + '_timedata.txt'
    x_sample = X[np.random.choice(X.shape[0], size=cell, replace=False), :]
    df = pd.DataFrame(x_sample.T)
    df.to_csv(path,sep='\t')


