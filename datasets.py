from torch.utils.data import Dataset
import torch
import numpy as np

class ExpressDataset(Dataset):
    def __init__(self, cell_list, gene_list, express_value, decoder='MLP', size_factor=None, gene_factor=None):
        super(ExpressDataset, self).__init__()
        self.cell_list = cell_list #表达的位置的细胞索引
        self.gene_list = gene_list #表达的位置的基因索引
        self.express_value = express_value

        self.decoder = decoder

        self.size_factor = size_factor #shape 细胞数*1
        self.gene_factor = gene_factor

    def __getitem__(self, index):
        cell_idx = self.cell_list[index]
        gene_idx = self.gene_list[index]
        exp_value = self.express_value[index]

        if self.decoder == 'MLP':
            return cell_idx, gene_idx, exp_value
        else:
            # ZINB
            sz_factor = self.size_factor[cell_idx] #细胞个数
            ge_factor = self.gene_factor[gene_idx]  #基因个数

            return cell_idx, gene_idx, exp_value, sz_factor, ge_factor

    def __len__(self):
        return len(self.cell_list)


class UnExpressDataset(Dataset):
    def __init__(self, cell_list, gene_list, decoder='MLP', size_factor=None, gene_factor=None):
        super(UnExpressDataset, self).__init__()
        self.cell_list = cell_list
        self.gene_list = gene_list

        self.decoder = decoder

        self.size_factor = size_factor
        self.gene_factor = gene_factor

    def __getitem__(self, index):
        cell_idx = self.cell_list[index]
        gene_idx = self.gene_list[index]

        if self.decoder == 'MLP':
            return cell_idx, gene_idx
        else:
            # ZINB
            sz_factor = self.size_factor[cell_idx]
            ge_factor = self.gene_factor[gene_idx]

            return cell_idx, gene_idx, sz_factor, ge_factor

    def __len__(self):
        return len(self.cell_list)

class Unexp_Sampler:
    def __init__(self, gene_c, power, sample_size): #负采样
        self.sample_size = sample_size
        self.n_genes = len(gene_c)

        self.gene_p = np.power(gene_c, power) #对gene_c中各元素求power次方
        self.gene_p /= np.sum(self.gene_p)  #转到[0,1]

    def get_negative_sample(self, exp_gene_idx, val_mask):
        p = self.gene_p.copy()
        target_idx = np.concatenate([exp_gene_idx, val_mask]) #列拼接 同一细胞中sample_size个基因idx和验证集中所有基因idx
        p[target_idx] = 0
        p /= p.sum()  #除训练、验证集外的基因
        unexp_gene_idx = np.random.choice(self.n_genes, size=self.sample_size, replace=True, p=p)
        # p:表示列表中某数被选取的概率，哪个基因表达得少，p越大，被选取的概率更大。随机选取sample_size个负样本
        return unexp_gene_idx


class TrainDataset(Dataset):
    def __init__(self, exp_mat, val_mask, sample_size=100, power=0., decoder='MLP', size_factor=None, gene_factor=None):
        super(TrainDataset, self).__init__()

        n_cells, n_genes = exp_mat.shape #观测矩阵的大小
        gene_c = np.sum(exp_mat != 0, axis=0) #每一列中！0的元素个数之和，每个基因中细胞表达数量，基因数*1
        #gene_g = np.sum(exp_mat != 0, axis=1)
        masked_exp_mat = np.multiply((1 - val_mask.todense()), exp_mat) #除去验证集后的非0表达矩阵 （训练集）

        del exp_mat  #删除变量

        self.val_mask = val_mask
        self.exp_gene_idx = [np.where(masked_exp_mat[cell_idx, :] != 0)[1] for cell_idx in range(n_cells)]
        self.exp_value = [masked_exp_mat[cell_idx, self.exp_gene_idx[cell_idx]] for cell_idx in range(n_cells)]
        #self.exp_gene_idx把每行细胞中非0值的基因序号记录存为array格式，再把所有细胞的array存为list格式，self.exp_value把每行细胞中非0值表达值记录存为list格式
        self.sample_size = sample_size
        self.negative_sampler = Unexp_Sampler(gene_c, power=power, sample_size=sample_size)

        self.decoder = decoder
        self.size_factor = size_factor
        self.gene_factor = gene_factor

    def __getitem__(self, cell_idx): #cell_idx [0,...,cell num-1]
        exp_gene_idx_tmp = self.exp_gene_idx[cell_idx]
        #在cell_idx行 取self.sample_size个 不同的exp_gene_idx 作为exp_idx
        exp_idx = np.random.choice(range(len(exp_gene_idx_tmp)), size=self.sample_size, replace=True)
        val_idx = self.val_mask[cell_idx, :].indices  #取验证集中同样cell_idx行的非0表达的基因idx

        exp_gene_idx = exp_gene_idx_tmp[exp_idx] #随机取sample_size个基因idx
        exp_value = self.exp_value[cell_idx][0, exp_idx] #取对应的值
        unexp_gene_idx = self.negative_sampler.get_negative_sample(exp_gene_idx, val_idx)

        if self.decoder == 'MLP':
            return cell_idx, exp_gene_idx, exp_value, unexp_gene_idx
        else:
            # ZINB
            sz_factor = self.size_factor[cell_idx]
            exp_ge_factor = self.gene_factor[exp_gene_idx]
            unexp_ge_factor = self.gene_factor[unexp_gene_idx]

            return cell_idx, exp_gene_idx, exp_value, unexp_gene_idx, sz_factor, exp_ge_factor, unexp_ge_factor

    def __len__(self):
        return self.val_mask.shape[0] #矩阵行数

class DataPrefetcher():
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                if k != 'meta':
                    self.batch[k] = self.batch[k].to(device=self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch