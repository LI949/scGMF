#!/usr/bin/env python
# coding: utf-8
import torch
from torch import nn, optim
import numpy as np
from torch.utils.data import DataLoader
from model.NeuMF import GMF, MLP, NeuCF,GMF_Dot
import scanpy as sc
from data.data_utils import prepare_data2,prepare_data3,prepare_data4, collate_train, collate_val,collate_denoise
from train.utils import calculate_metric, ZINBLoss
from sklearn.cluster import KMeans
from Bio.Cluster import kcluster
from sklearn.metrics import silhouette_score
from data.datasets import DataPrefetcher
from data.early_stopping import EarlyStopping
from scipy.sparse import csr_matrix

def kmeans(adata, n_clusters, use_rep='feat'):  #n_clusters聚类数，n_init用不同的聚类中心初始化值运行算法的次数
    k_means = KMeans(n_clusters, n_init=20)
    y_pred = k_means.fit_predict(adata.obsm[use_rep])
    adata.obs['kmeans'] = y_pred
    adata.obs['kmeans'] = adata.obs['kmeans'].astype(str).astype('category')
    return adata

def louvain(adata,resolution=None, use_rep='feat'):
    sc.pp.neighbors(adata, use_rep=use_rep)  #,metric='cosine'
    sc.tl.louvain(adata, resolution=resolution)
    return adata

def scNCF(adata,
          n_clusters=None,
          cl_type=None,
          feats_dim=48,
          batch_size=64,
          sample_size=128,
          gamma=2,
          power=0.75,
          val_ratio=0.02,
          encoder='GMF',
          decoder='MLP',
          lr=0.01,
          n_epoch=200,
          numworker = 4,
          log_interval=20,
          resolution=1,
          mode='undenoise'
          ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    assert encoder in ['GMF', 'MLP', 'NeuCF','GMF_Dot'], "Please choose model in ['GMF', 'MLP', 'NeuCF','GMF_Dot']"
    if encoder in ['GMF', 'NeuCF']:
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
            print('Please input number of clusters or set cell type information in adata[\'cl_type\'], otherwise cannot perform cell clustering operations ')

    adata, train_datasets, val_datasets = prepare_data2(adata, decoder, raw_exp, sample_size=sample_size, power=power, val_ratio=val_ratio)

    train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, drop_last=False,
                                pin_memory=True, num_workers=numworker, collate_fn=collate_train)
    val_dataloader = DataLoader(val_datasets, batch_size=batch_size*sample_size, shuffle=True, drop_last=False,
                                pin_memory=True, num_workers=numworker, collate_fn=collate_val)
    # n_samples = len(train_dataloader)*sample_size

    n_cells, n_genes = adata.X.shape
    #######################   Prepare models   #######################
    if encoder == 'GMF':
        model = GMF(n_cells, n_genes, feats_dim, decoder=decoder)
    elif encoder == 'MLP':
        model = MLP(n_cells, n_genes, layers=[2*feats_dim,feats_dim,feats_dim//2],decoder=decoder)
    elif encoder == 'GMF_Dot':
        model = GMF_Dot(n_cells, n_genes, feats_dim, decoder=decoder)
    else: # NeuralCF
        model = NeuCF(n_cells, n_genes, latent_dim_mf=feats_dim, layers=[2*feats_dim, feats_dim,feats_dim//2], decoder=decoder)
    print(model)

    model = model.to(device)
    if decoder == 'MLP':
        criterion = nn.MSELoss()   #损失函数
    else:
        criterion = ZINBLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,201], gamma=0.5)

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
    cluster_l = []

    # save_path = "../../experiment/"  # 当前目录下
    # early_stopping = EarlyStopping(save_path)
    ############ training
    print(f"Start training on {device}...")
    loss_num = 0

    for epoch in range(n_epoch+1):
        for train_data in train_dataloader:

            if decoder == 'ZINB':
                cell_idx, exp_gene_idx, exp_value, unexp_gene_idx, sz_factor, exp_ge_factor, unexp_ge_factor = train_data
                cell_idx, exp_gene_idx, exp_value, unexp_gene_idx, sz_factor, exp_ge_factor, unexp_ge_factor = \
                    torch.from_numpy(cell_idx).long().to(device),torch.from_numpy(exp_gene_idx).long().to(device), torch.from_numpy(exp_value).float().to(device),\
                    torch.from_numpy(unexp_gene_idx).long().to(device),torch.from_numpy(sz_factor).float().to(device),\
                    torch.from_numpy(exp_ge_factor).float().to(device),torch.from_numpy(unexp_ge_factor).float().to(device)

                pred_exp = model(cell_idx, exp_gene_idx, sz_factor, exp_ge_factor)
                #print(exp_value.shape)  #torch.Size([16384, 1])
                #print(pred_exp[0].shape) # torch.Size([16384, 1])
                loss_exp = criterion(pred_exp[0], pred_exp[1], pred_exp[2], exp_value)

                pred_unexp = model(cell_idx, unexp_gene_idx, sz_factor, unexp_ge_factor)
                loss_unexp = criterion(pred_unexp[0], pred_unexp[1], pred_unexp[2])

            else:
                cell_idx, exp_gene_idx, exp_value, unexp_gene_idx= train_data
                cell_idx, exp_gene_idx, exp_value, unexp_gene_idx = torch.from_numpy(cell_idx).long().to(device),torch.from_numpy(exp_gene_idx).long().to(device), \
                                                                    torch.from_numpy(exp_value).float().to(device),torch.from_numpy(unexp_gene_idx).long().to(device)

                pred_exp = model(cell_idx, exp_gene_idx)
                #print(exp_value.shape)#torch.Size([16384, 1])
                #print(pred_exp.shape)#torch.Size([16384, 1])
                loss_exp = criterion(pred_exp, exp_value)  #torch.float32

                pred_unexp = model(cell_idx, unexp_gene_idx)
                unexp_value = torch.zeros([unexp_gene_idx.numel(), 1]).to(device)
                loss_unexp = criterion(pred_unexp, unexp_value)

            count_exp_loss += loss_exp.item()
            count_unexp_loss += loss_unexp.item()

            optimizer.zero_grad()
            loss = loss_exp + gamma*loss_unexp

            if decoder == 'ZINB':
                ridge = torch.square(pred_exp[2]).mean() + torch.square(pred_unexp[2]).mean()
                loss = loss + 0.01 * ridge

            loss.backward()
            nn.utils.clip_grad_norm_(parameters = model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()

        #训练完一个epoch
        trainloss = (count_exp_loss + gamma * count_unexp_loss) / len(train_dataloader)
        print("[{}/{}-epoch] | [train] exp loss : {:.4f}, unexp loss : {:.4f}, total loss : {:.4f}".
              format(epoch, n_epoch, count_exp_loss/len(train_dataloader), count_unexp_loss/len(train_dataloader), trainloss))
        trainloss_list.append(trainloss)
        count_exp_loss, count_unexp_loss, count_val_exp_loss = 0, 0,0

        ###############################   Start Validating   ###############################

        #print("Start Validating!")
        model.eval()
        for exp_val_data in val_dataloader:
            with torch.no_grad():
                if decoder == 'ZINB':
                    cell_idx, gene_idx, exp_val_value, sz_factor, ge_factor = exp_val_data
                    cell_idx, gene_idx, exp_val_value, sz_factor, ge_factor \
                        = torch.from_numpy(cell_idx).to(device), torch.from_numpy(gene_idx).to(device), torch.from_numpy(
                        exp_val_value).float().to(device), torch.from_numpy(sz_factor).float().to(device),torch.from_numpy(ge_factor).float().to(device)

                    pred_val_exp = model(cell_idx, gene_idx, sz_factor, ge_factor)
                    loss_val_exp = criterion(pred_val_exp[0], pred_val_exp[1], pred_val_exp[2], exp_val_value)

                else:
                    cell_idx, gene_idx, exp_val_value = exp_val_data
                    cell_idx, gene_idx, exp_val_value = torch.from_numpy(cell_idx).to(device),\
                                                        torch.from_numpy(gene_idx).to(device), torch.from_numpy(exp_val_value).float().to(device)

                    pred_val_exp = model(cell_idx, gene_idx)
                    loss_val_exp = criterion(pred_val_exp, exp_val_value)

                count_val_exp_loss += loss_val_exp.item()
                #count_val_unexp_loss += loss_val_unexp.item()
        valloss = (count_val_exp_loss)/len(val_dataloader)
        print("[{}/{}-epoch] | [val] val loss : {:.4f}".format(epoch, n_epoch, valloss))
        valloss_list.append(valloss)
        #early_stopping(valloss, model)

        model.train()
        #scheduler.step()

        if epoch == 0:
            temp_epoch0 = epoch
            if valloss > trainloss:
                loss_num = loss_num + 1
        else:
            if valloss > trainloss:
                loss_num = loss_num + 1
                temp_epoch0 = epoch
            else:
                temp_epoch0 = epoch
                loss_num = 0

        if loss_num >= 3:
            print("Reach stop critrion!")
            stop_flag = True

        #if True:
        if (epoch + 1) % log_interval == 0 or epoch == n_epoch or stop_flag == True:
        # if epoch > 0 and epoch % (log_interval * 5) == 0 or epoch == n_epoch:
            if encoder in ['GMF', 'MLP','GMF_Dot']:
                adata.obsm['feat'] = model.embedding_cell.weight.cpu().detach().numpy()
            else:  # NeuCF
                adata.obsm['feat'] = np.concatenate([model.embedding_cell_mlp.weight.cpu().detach().numpy(),
                                                 model.embedding_cell_mf.weight.cpu().detach().numpy()],axis=-1)

            if cl_type:
                # kmeans
                adata = kmeans(adata, n_clusters)
                y_pred_k = np.array(adata.obs['kmeans'])

                # louvain
                adata = louvain(adata, resolution=resolution)
                y_pred_l = np.array(adata.obs['louvain'])

                sil_k = silhouette_score(adata.obsm['feat'], adata.obs['kmeans'])
                sil_l = silhouette_score(adata.obsm['feat'], adata.obs['louvain'])

                acc, nmi_k, ari_k = calculate_metric(cell_type, y_pred_k)
                print('Clustering Kmeans %d: ACC= %.4f, NMI= %.4f, ARI= %.4f, Silhouette= %.4f' % (
                    epoch, acc, nmi_k, ari_k, sil_k))

                acc, nmi_l, ari_l = calculate_metric(cell_type, y_pred_l)
                print('Number of clusters identified by Louvain is {}'.format(len(np.unique(y_pred_l))))
                print('Clustering Louvain %d: ACC= %.4f, NMI= %.4f, ARI= %.4f, Silhouette= %.4f' % (
                    epoch, acc, nmi_l, ari_l, sil_l))

                all_sil_k.append(sil_k)
                all_sil_l.append(sil_l)
                all_ari_k.append(ari_k)
                all_ari_l.append(ari_l)
                all_nmi_k.append(nmi_k)
                all_nmi_l.append(nmi_l)
                cluster_l.append(len(np.unique(y_pred_l)))

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
        if stop_flag:
            break

    if cl_type:
        print(
            f'[END] For Kmeans, Best epoch : {best_epoch_k} Best ARI : {best_ari_k:.4f}, Best NMI : {best_nmi_k:.4f}')

        print(
            f'[END] For Louvain, Best epoch : {best_epoch_l} Best ARI : {best_ari_l:.4f}, Best NMI : {best_nmi_l:.4f}')

    trainloss_list = np.array(torch.tensor(trainloss_list, device='cpu'))
    valloss_list = np.array(torch.tensor(valloss_list, device='cpu'))

    # after training ,to impute matrix
    if mode == 'denoise':
        print("Start imputing!")
        M, N = adata.X.shape[0], adata.X.shape[1]
        tensor = torch.empty((M, N))
        size_factor = torch.from_numpy(adata.obs['size_factor'].values)
        gene_factor = torch.from_numpy(adata.var['gene_factor'].values)

        adata, denoise_datasets = prepare_data4(adata, decoder, raw_exp)
        denoise_dataloader = DataLoader(denoise_datasets, batch_size=1, shuffle=False,
                                        drop_last=False, num_workers=numworker)
        ge_factor = torch.unsqueeze(gene_factor.float(),1).to(device)  #size(N,)
        sz_factor_1 = size_factor.float().to(device)
        gene_idx = torch.arange(0, N).reshape(-1).long().to(device)

        for batch_idx,data in enumerate(denoise_dataloader):
            if decoder == 'ZINB':
                all_value,cell_idx1 = data
                cell_idx = torch.squeeze(torch.unsqueeze(cell_idx1, 0).repeat(N, 1)).long().to(device)
                sz_factor = sz_factor_1[cell_idx1.long()]
                sz_factor = torch.unsqueeze(sz_factor, 0).repeat(N,1).to(device)

                pred_exp = model(cell_idx, gene_idx, sz_factor, ge_factor)
                new_tensor = torch.squeeze(pred_exp[0])
                tensor[cell_idx1.long(),:] = new_tensor.detach().cpu()

            else:
                cell_idx1, all_value = data
                cell_idx = torch.squeeze(torch.unsqueeze(cell_idx1, 0).repeat(N, 1)).long().to(device)

                pred_exp = model(cell_idx, gene_idx)
                new_tensor = torch.squeeze(pred_exp)
                tensor[cell_idx1.long(),:] = new_tensor.detach().cpu()

        if decoder == 'ZINB':
            imputed_matrix = tensor.numpy()

        else:
            X_imputed = tensor * gene_factor
            adata.obsm['X_imputed_processed'] = X_imputed.numpy()
            XPred = torch.expm1(X_imputed)
            size_factor_diag = torch.diag(size_factor)
            imputed_matrix = torch.matmul(size_factor_diag, XPred)
            imputed_matrix = imputed_matrix.numpy()

        adata.obsm['X_imputed'] = imputed_matrix

    record = {
        'all_loss_train': trainloss_list,
        'all_loss_val': valloss_list
    }

    if cl_type:
        record_ = {
            'all_loss_train': trainloss_list,
            'all_loss_val': valloss_list,
            'sil_k': all_sil_k,
            'sil_l': all_sil_l,
            'ari_k': all_ari_k,
            'ari_l': all_ari_l,
            'nmi_k': all_nmi_k,
            'nmi_l': all_nmi_l,
            'pred_cluster_l': cluster_l
        }

        record.update(record_)

    return adata,record,trainloss_list

# import h5py
# from data.data_utils import preprocess
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
#     adata,record,epochloss_list = scNCF(adata,n_clusters=n_clusters,cl_type='cl_type',lr=0.01,batch_size=128,
#                                       power=0.75,encoder='NeuCF', decoder='ZINB', gamma=1, n_epoch=10)
#
#     print(adata)
#     adata.obs['cluster'] = adata.obs['pred']
