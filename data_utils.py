import scanpy as sc
import numpy as np
import pandas as pd
import random
import os
import torch
from data.datasets import ExpressDataset, UnExpressDataset, TrainDataset
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

def sample(x, label, seed):
    x_sample = pd.DataFrame()
    if np.unique(label)[0]==0:
        label = label + 1
    for i in range(len(np.unique(label))):
        j = i + 1
        data = x[label == j, ]
        data = pd.DataFrame(data)
        data = data.sample(frac=0.95, replace=False, weights=None, random_state=seed, axis=0)
        data['y'] = j
        x_sample = x_sample.append(data, ignore_index=True)

    y_sample = np.asarray(x_sample['y'], dtype='int')
    x_sample = np.asarray(x_sample.iloc[:, :-1])
    return x_sample, y_sample


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

def preprocess(adata, filter_min_counts=True, size_factors=True, normalize_input=False, logtrans_input=True):
    #log转换预处理
    if filter_min_counts:
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factor'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factor'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata

def prepare_data(adata, decoder, raw_exp=False,val_ratio=0.02):
    X = adata.X   #
    #再处理数据
    size_factor = adata.obs['size_factor'].values
    gene_factor = np.max(X, axis=0, keepdims=True).reshape(-1)#求每一列的最大值，6557*1，即基因的最大表达值
    adata.var['gene_factor'] = gene_factor
    #直接取得矩阵中所有细胞、基因的序列
    exp_cell, exp_gene = np.where(X > 0)
    train_idx, val_idx = train_test_split(np.array(range(len(exp_cell))), test_size=val_ratio)
    exp_train_cell, exp_train_gene = exp_cell[train_idx], exp_gene[train_idx]
    exp_val_cell, exp_val_gene = exp_cell[val_idx], exp_gene[val_idx]

    unexp_cell, unexp_gene = np.where(X == 0)

    if raw_exp:
        exp_value = adata.raw.X[exp_train_cell, exp_train_gene].reshape(-1,1)
        exp_val_value = adata.raw.X[exp_val_cell, exp_val_gene].reshape(-1, 1)
    else:
        X = X / gene_factor  #映射到[0,1]
        exp_value = X[exp_train_cell, exp_train_gene].reshape(-1, 1)
        exp_val_value = X[exp_val_cell, exp_val_gene].reshape(-1, 1)

    #区分0和非0表达，划分数据集
    #decoder 分 MLP  ZINB 表达一个是映射到[0,1]后，一个是原始表达；
    exp_datasets = ExpressDataset(exp_train_cell, exp_train_gene, exp_value,
                                  decoder=decoder, size_factor=size_factor, gene_factor=gene_factor)
    exp_val_datasets = ExpressDataset(exp_val_cell, exp_val_gene, exp_val_value,
                                      decoder=decoder, size_factor=size_factor, gene_factor=gene_factor)
    unexp_datasets = UnExpressDataset(unexp_cell, unexp_gene,
                                      decoder=decoder, size_factor=size_factor, gene_factor=gene_factor)
    # unexp_val_datasets = UnExpressDataset(unexp_valid_cell, unexp_valid_gene,
    #                                   decoder=decoder, size_factor=size_factor,gene_factor=gene_factor)

    return adata,exp_datasets,exp_val_datasets, unexp_datasets

def prepare_data2(adata, decoder, raw_exp=False, sample_size=100, power=0., val_ratio=0.02):
    X = adata.X

    size_factor = adata.obs['size_factor'].values
    gene_factor = np.max(X, axis=0, keepdims=True).reshape(-1)
    adata.var['gene_factor'] = gene_factor

    exp_cell, exp_gene = np.where(X > 0)
    #replace:False表示不可以取相同数字，在[0,len(exp_cell))随机选 len(exp_cell)*val_ratio个数(entry)作为验证集候补
    val_idx = np.random.choice(range(len(exp_cell)), round(len(exp_cell)*val_ratio), replace=False)
    val_cell, val_gene = exp_cell[val_idx], exp_gene[val_idx] #取非0表达的细胞和基因的idx 作为验证集

    #逐行压缩的稀疏矩阵,在val_cell, val_gene位置即非0表达位置此矩阵元素为1，其余为0, shape为cell * gene
    val_mask = csr_matrix((np.ones(len(val_idx)), (val_cell, val_gene)), shape=X.shape)

    if raw_exp:
        val_value = adata.raw.X[val_cell, val_gene]
        train_datasets = TrainDataset(exp_mat=adata.raw.X, val_mask=val_mask, sample_size=sample_size, power=power,
                                      decoder=decoder, size_factor=size_factor,gene_factor=gene_factor)
        val_datasets = ExpressDataset(val_cell, val_gene, val_value, decoder=decoder,
                                      size_factor=size_factor, gene_factor=gene_factor)

    else:
        X = X / gene_factor
        val_value = X[val_cell, val_gene]
        train_datasets = TrainDataset(exp_mat=X, val_mask=val_mask, sample_size=sample_size,
                                      decoder=decoder, size_factor=None,gene_factor=None)
        val_datasets = ExpressDataset(val_cell, val_gene, val_value, decoder=decoder,
                                      size_factor=None, gene_factor=None)

    return adata, train_datasets, val_datasets

def prepare_data3(adata, decoder, raw_exp=False):
    X = adata.X
    size_factor = adata.obs['size_factor'].values
    gene_factor = np.max(X, axis=0, keepdims=True).reshape(-1)
    adata.var['gene_factor'] = gene_factor
    exp_cell, exp_gene = np.where(X > 0)
    cell, gene = np.where(X >= 0)

    if raw_exp:
        value = adata.raw.X[cell, gene].reshape(-1,1)
    else:
        X = X / gene_factor
        value = X[cell, gene].reshape(-1, 1)

    datasets = ExpressDataset(cell, gene, value,decoder=decoder, size_factor=size_factor, gene_factor=gene_factor)

    return adata,datasets

def prepare_data4(adata, decoder, raw_exp=False):
    X = adata.X
    size_factor = adata.obs['size_factor'].values
    gene_factor = np.max(X, axis=0, keepdims=True).reshape(-1)
    adata.var['gene_factor'] = gene_factor

    if raw_exp:
        Datasets = adata.raw.X
    else:
        Datasets = X / gene_factor

    datasets = CustomDataset(Datasets)

    return adata,datasets

def collate_train(batch):  #采样器的处理函数,batch_list送给collate_fn组织成batch最后的形式
    sample_number = len(batch[0][1])
    #处理变长数据，信息存储在res,分别提取批样本中的cell_idx等信息;MLP存4个，ZINB存7个
    #每个cell里采样了sample_number个，故'cell_idx'和'sz_factor'要* sample_number
    res = {
        'cell_idx' : np.stack([[item[0]] * sample_number for item in batch]).reshape(-1), #拉成一列，list:batchsize
        'exp_gene_idx' : np.stack([item[1] for item in batch]).reshape(-1),
        'exp_value' : np.stack([item[2] for item in batch]).reshape(-1,1),
        'unexp_gene_idx' : np.stack([item[3] for item in batch]).reshape(-1)
    }

    if len(batch[0]) == 7:
        res.update({
            'sz_factor': np.stack([[item[4]] * sample_number for item in batch]).reshape(-1, 1),
            'exp_ge_factor' : np.stack([item[5] for item in batch]).reshape(-1,1),
            'unexp_ge_factor' : np.stack([item[6] for item in batch]).reshape(-1,1)})

    for (k, v) in res.items(): #循环取出字典里面的k,v
        res[k] = np.stack(np.array(v), axis=0)

    return list(res.values())

def collate_val(batch):
    res = {
        'cell_idx' : np.stack([item[0] for item in batch]).reshape(-1),
        'gene_idx' : np.stack([item[1] for item in batch]).reshape(-1),
        'exp_val_value' : np.stack([item[2] for item in batch]).reshape(-1, 1)
    }

    if len(batch[0]) == 5:
        res.update({
            'sz_factor': np.stack([item[3] for item in batch]).reshape(-1, 1),
            'ge_factor': np.stack([item[4] for item in batch]).reshape(-1, 1)})

    for (k, v) in res.items():
        res[k] = np.stack(np.array(v), axis=0)

    return list(res.values())

from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self, matrix):
        self.matrix = matrix

    def __getitem__(self, index):
        return torch.from_numpy(np.array(self.matrix[index])).float(), \
               torch.from_numpy(np.array(index))

    def __len__(self):
        return self.matrix.shape[0]

def collate_denoise(batch):
    res = {
        'cell_idx' : np.stack([item[0] for item in batch]).reshape(-1),
        'gene_idx' : np.stack([item[1] for item in batch]).reshape(-1),
        'value' : np.stack([item[2] for item in batch]).reshape(-1, 1)
    }

    if len(batch[0]) == 5:
        res.update({
            'sz_factor': np.stack([item[3] for item in batch]).reshape(-1, 1),
            'ge_factor': np.stack([item[4] for item in batch]).reshape(-1, 1)})

    for (k, v) in res.items():
        res[k] = np.stack(np.array(v), axis=0)

    return list(res.values())
