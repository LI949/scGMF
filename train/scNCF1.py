#!/usr/bin/env python
# coding: utf-8
import torch
from torch import nn, optim
import numpy as np
from torch.utils.data import DataLoader
from model.NeuMF import GMF, MLP, NeuCF
import scanpy as sc
import h5py
from data.data_utils import preprocess, prepare_data
from train.utils import calculate_metric, ZINBLoss
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from data.datasets import DataPrefetcher
from data.early_stopping import EarlyStopping

def kmeans(adata, n_clusters):  #n_clusters聚类数，n_init用不同的聚类中心初始化值运行算法的次数
    k_means = KMeans(n_clusters, n_init=20)
    y_pred = k_means.fit_predict(adata.obsm['feat'])
    adata.obs['kmeans'] = y_pred
    adata.obs['kmeans'] = adata.obs['kmeans'].astype(str).astype('category')
    return adata

def louvain(adata,resolution=None):
    sc.pp.neighbors(adata, use_rep='feat',metric='cosine')
    sc.tl.louvain(adata, resolution=resolution)
    return adata

def scNCF1(adata,
          n_clusters=None,
          cl_type=None,
          feats_dim=128,
          batch_size=32768,
          drop_out=0.3,
          gamma=1,
          encoder='GMF',
          decoder='MLP',
          lr=0.01,
          n_epoch=300,
          numworker = 4,
          ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    assert encoder in ['GMF', 'MLP', 'NeuCF'], "Please choose model in ['GMF', 'MLP', 'NeuMF']"
    if encoder in ['GMF', 'NeuMF']:
        assert decoder in ['MLP', 'ZINB'], "Please choose decoder in ['MLP','ZINB']"

    # assert sample_rate <= 1, "Please set 0<sample_rate<=1"

    ####################   Prepare data for training   ####################
    cell_type = adata.obs[cl_type].values if cl_type else None
    raw_exp = True if decoder == 'ZINB' else False
    stop_flag = False

    if n_clusters is None:
        if cell_type:
            n_clusters = len(np.unique(adata['cl_type']))
        else:
            raise Exception('Please input number of clusters or set cell type information in adata[\'cl_type\']')

    adata, exp_datasets,exp_val_datasets, unexp_datasets,unexp_val_datasets = \
        prepare_data(adata,decoder,raw_exp,val_ratio=0.02)

    exp_dataloader = DataLoader(exp_datasets, batch_size=batch_size, shuffle=True, drop_last=False,
                                pin_memory=True,num_workers=numworker)
    exp_val_dataloader = DataLoader(exp_val_datasets, batch_size=batch_size, shuffle=True, drop_last=False,
                                pin_memory=True,num_workers=numworker)

    n_cells, n_genes = adata.X.shape
    #######################   Prepare models   #######################
    if encoder == 'GMF':
        model = GMF(n_cells, n_genes, feats_dim, decoder=decoder)
    elif encoder == 'MLP':
        model = MLP(n_cells, n_genes, layers=[2*feats_dim,feats_dim,feats_dim//2], dropout=drop_out)
    else: # NeuralCF
        model = NeuCF(n_cells, n_genes, latent_dim_mf=feats_dim, layers=[2*feats_dim, feats_dim,feats_dim//2], dropout=drop_out, decoder=decoder)
    print(model)

    model = model.to(device)
    if decoder == 'MLP':
        criterion = nn.MSELoss()   #损失函数
    else:
        criterion = ZINBLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    ############
    best_ari_k, best_ari_l = 0, 0
    best_nmi_k, best_nmi_l = 0, 0
    # no_better_valid = 0
    all_ari_k, all_ari_l = [], []
    all_nmi_k, all_nmi_l = [], []
    all_sil_k, all_sil_l = [], []

    best_epoch_k, best_epoch_l = -1, -1
    count_exp_loss = 0
    count_unexp_loss = 0
    trainloss_list=[]
    valloss_list=[]
    trainloss=0
    valloss=0

    # save_path = "../../experiment/"  # 当前目录下
    # early_stopping = EarlyStopping(save_path)
    ############ training
    print(f"Start training on {device}...")
    unexp_value = torch.zeros([batch_size,1]).to(device)

    for epoch in range(n_epoch+1):
        # if epoch == 10:
        #     adata.obsm['feat'] = model.embedding_cell.weight.data.cpu().numpy()
        #     # kmeans
        #     adata = kmeans(adata, n_clusters)
        #     y_pred = np.array(adata.obs['kmeans'])
        exp_prefetcher = DataPrefetcher(exp_dataloader, device=device)
        unexp_dataloader = DataLoader(unexp_datasets, batch_size=batch_size, shuffle=True, drop_last=True,
                                      pin_memory=True, num_workers=numworker)
        # unexp_iter = iter(unexp_dataloader)
        unexp_prefetcher = DataPrefetcher(unexp_dataloader, device=device)

        exp_data = exp_prefetcher.next()
        while exp_data is not None:
            #iter中训练计算
            unexp_data = unexp_prefetcher.next()
            if decoder == 'ZINB':
                cell_idx, gene_idx, exp_value, sz_factor, ge_factor = exp_data
                cell_idx, gene_idx, exp_value, sz_factor, ge_factor \
                    = cell_idx.to(device), gene_idx.to(device), exp_value.to(device), sz_factor.to(device), ge_factor.to(device)

                pred_exp = model(cell_idx, gene_idx, sz_factor, ge_factor)
                loss_exp = criterion(pred_exp[0], pred_exp[1], pred_exp[2], exp_value)

                cell_idx, gene_idx, sz_factor, ge_factor = unexp_data
                cell_idx, gene_idx, sz_factor, ge_factor = cell_idx.to(device), gene_idx.to(device), sz_factor.to(
                    device), ge_factor.to(device)

                pred_unexp = model(cell_idx, gene_idx, sz_factor, ge_factor)
                loss_unexp = criterion(pred_unexp[0], pred_unexp[1], pred_unexp[2])

            else:
                cell_idx, gene_idx, exp_value = exp_data
                cell_idx, gene_idx, exp_value = cell_idx.to(device), gene_idx.to(device), exp_value.to(device)

                pred_exp = model(cell_idx, gene_idx)
                loss_exp = criterion(pred_exp, exp_value)

                cell_idx, gene_idx = unexp_data
                cell_idx, gene_idx = cell_idx.to(device), gene_idx.to(device)

                pred_unexp = model(cell_idx, gene_idx)
                loss_unexp = criterion(pred_unexp, unexp_value)

            count_exp_loss += loss_exp.item()
            count_unexp_loss += loss_unexp.item()

            optimizer.zero_grad()
            loss = loss_exp + gamma*loss_unexp
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            exp_data = exp_prefetcher.next()
        #训练完一个epoch
        trainloss = (count_exp_loss + gamma * count_unexp_loss) / len(exp_dataloader)
        print("[{}/{}-epoch] | [train] exp loss : {:.4f}, unexp loss : {:.4f}, total loss : {:.4f}".
              format(epoch, n_epoch, count_exp_loss/len(exp_dataloader), count_unexp_loss/len(exp_dataloader), trainloss))
        trainloss_list.append(trainloss)
        count_exp_loss, count_unexp_loss = 0, 0

        count_val_exp_loss,count_val_unexp_loss=0,0
        #unexp_val_dataloader = DataLoader(unexp_val_datasets, batch_size=batch_size, shuffle=True, drop_last=True,
        #                                  pin_memory=True, num_workers=numworker)
        #print("Start Validating!")
        model.eval()
        with torch.no_grad():
            #exp_val_prefetcher = DataPrefetcher(exp_val_dataloader, device=device)
            #unexp_val_prefetcher=DataPrefetcher(unexp_val_dataloader, device=device)
            #exp_val_data = exp_val_prefetcher.next()
            #while exp_val_data is not None:
            for data in exp_val_dataloader:
                # iter中训练计算
                #unexp_val_data = unexp_val_prefetcher.next()
                if decoder == 'ZINB':
                    cell_idx, gene_idx, exp_val_value, sz_factor, ge_factor = data
                    cell_idx, gene_idx, exp_val_value, sz_factor, ge_factor \
                        = cell_idx.to(device), gene_idx.to(device), exp_val_value.to(device), sz_factor.to(
                        device), ge_factor.to(device)

                    pred_val_exp = model(cell_idx, gene_idx, sz_factor, ge_factor)
                    loss_val_exp = criterion(pred_val_exp[0], pred_val_exp[1], pred_val_exp[2], exp_val_value)

                    # cell_idx, gene_idx, sz_factor, ge_factor = unexp_val_data
                    # cell_idx, gene_idx, sz_factor, ge_factor = cell_idx.to(device), gene_idx.to(
                    #     device), sz_factor.to(device), ge_factor.to(device)
                    #
                    # pred_val_unexp = model(cell_idx, gene_idx, sz_factor,ge_factor)
                    # loss_val_unexp = criterion(pred_val_unexp[0], pred_val_unexp[1], pred_val_unexp[2])
                else:
                    cell_idx, gene_idx, exp_val_value = data
                    cell_idx, gene_idx, exp_val_value = cell_idx.to(device), gene_idx.to(device), exp_val_value.to(device)

                    pred_val_exp = model(cell_idx, gene_idx)
                    loss_val_exp = criterion(pred_val_exp, exp_val_value)

                    # cell_idx, gene_idx = unexp_val_data
                    # cell_idx, gene_idx = cell_idx.to(device), gene_idx.to(device)
                    #
                    # pred_val_unexp = model(cell_idx, gene_idx)
                    # loss_val_unexp = criterion(pred_val_unexp, unexp_value)

                count_val_exp_loss += loss_val_exp.item()
                #count_val_unexp_loss += loss_val_unexp.item()
            valloss = (count_val_exp_loss)/len(exp_val_dataloader)
            print("[{}/{}-epoch] | [val] val loss : {:.4f}".format(epoch, n_epoch, count_val_exp_loss/len(exp_val_dataloader)))
            valloss_list.append(valloss)
        #early_stopping(valloss, model)
        model.train()

        if valloss > trainloss:
           print("Reach stop critrion!")
        #     stop_flag = True
        #
        # if stop_flag:
        #     break

        if True:
        # if epoch > 0 and epoch % (log_interval * 5) == 0 or epoch == n_epoch:
            adata.obsm['feat'] = np.concatenate([model.embedding_cell_mlp.weight.data.cpu().numpy(), model.embedding_cell_mf.weight.data.cpu().numpy()],axis=1)
            # kmeans
            adata = kmeans(adata, n_clusters)
            y_pred_k = np.array(adata.obs['kmeans'])

            # louvain
            adata = louvain(adata, resolution=None)
            y_pred_l = np.array(adata.obs['louvain'])
            print('Number of clusters identified by Louvain is {}'.format(len(np.unique(y_pred_l))))

            sil_k = silhouette_score(adata.obsm['feat'], adata.obs['kmeans'])
            sil_l = silhouette_score(adata.obsm['feat'], adata.obs['louvain'])
            all_sil_k.append(sil_k)
            all_sil_l.append(sil_l)

            if cl_type:
                acc, nmi_k, ari_k = calculate_metric(cell_type, y_pred_k)
                print('Clustering Kmeans %d: ACC= %.4f, NMI= %.4f, ARI= %.4f, Silhouette= %.4f' % (
                    epoch, acc, nmi_k, ari_k, sil_k))

                acc, nmi_l, ari_l = calculate_metric(cell_type, y_pred_l)
                print('Clustering Louvain %d: ACC= %.4f, NMI= %.4f, ARI= %.4f, Silhouette= %.4f' % (
                    epoch, acc, nmi_l, ari_l, sil_l))

                all_ari_k.append(ari_k)
                all_ari_l.append(ari_l)
                all_nmi_k.append(nmi_k)
                all_nmi_l.append(nmi_l)

                if ari_k > best_ari_k:
                    best_ari_k = ari_k
                    best_nmi_k = nmi_k
                    # no_better_valid = 0
                    best_epoch_k = epoch

                    # torch.save(model.state_dict(), './ckpt/model.pth')
                if ari_l > best_ari_l:
                    best_ari_l = ari_l
                    best_nmi_l = nmi_l
                    # no_better_valid = 0
                    best_epoch_l = epoch

                # else:
                #     no_better_valid += 1
                #     if no_better_valid > early_stopping:
                #         print("Early stopping threshold reached. Stop training.")
                #         break
                #     if no_better_valid > lr_intialize_step:
                #         new_lr = max(lr * lr_decay, train_min_lr)
                #         if new_lr < lr:
                #             lr = new_lr
                #             print("\tChange the LR to %g" % new_lr)
                #             for p in optimizer.param_groups:
                #                 p['lr'] = lr
                #             no_better_valid = 0

        #达到早停止条件时，early_stop会被置为True
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break  # 跳出迭代，结束训练

    print(
        f'[END] For Kmeans, Best epoch : {best_epoch_k} Best ARI : {best_ari_k:.4f}, Best NMI : {best_nmi_k:.4f}')

    print(
        f'[END] For Louvain, Best epoch : {best_epoch_l} Best ARI : {best_ari_l:.4f}, Best NMI : {best_nmi_l:.4f}')

    trainloss_list = np.array(torch.tensor(trainloss_list, device='cpu'))
    valloss_list = np.array(torch.tensor(valloss_list, device='cpu'))
    record = {
        'all_loss_train': trainloss_list,
        'all_loss_val': valloss_list,
        'sil_k' : all_sil_k,
        'sil_l' : all_sil_l
    }

    if cl_type:
        record_ = {
            'all_loss_train': trainloss_list,
            'all_loss_val': valloss_list,
            'ari_k': all_ari_k,
            'ari_l': all_ari_l,
            'nmi_k': all_nmi_k,
            'nmi_l': all_nmi_l
        }

        record.update(record_)

    return adata,record,trainloss_list


# if __name__ == '__main__':
#     #data_mat = h5py.File("../datasets/10X_PBMC.h5", "r")
#     data_mat = h5py.File("../datasets/mouse_bladder_cell.h5", "r")
#     # data_mat = h5py.File("datasets/Small_Datasets/human_kidney_counts.h5", "r")
#
#     X = np.array(data_mat['X'])
#     Y = np.array(data_mat['Y'])
#     X = np.ceil(X).astype(np.int_)
#     n_clusters = len(np.unique(Y))
#
#     adata = sc.AnnData(X)
#     adata = preprocess(adata)
#     adata.obs['cl_type'] = Y
#     adata.obs['cl_type'] = adata.obs['cl_type'].astype(str).astype('category')
#
#     print("Sparsity: ", np.where(adata.X != 0)[0].shape[0] / (adata.X.shape[0] * adata.X.shape[1]))
#
#     adata,record,epochloss_list = scNCF1(adata, n_clusters=n_clusters, cl_type='cl_type',lr=0.01, batch_size=8192,
#                       encoder='NeuCF', decoder='ZINB', gamma=1, n_epoch=3)
#
#     print(adata)
#     adata.obs['cluster'] = adata.obs['pred']
