import torch
import torch.nn as nn
from model.decoder import MLPDecoder, ZINBDecoder

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x)) # 激活函数 Mish 函数

def buildNetwork(layers, activation="relu"):
    net = torch.nn.ModuleList()
    for in_size, out_size in zip(layers[:-2], layers[1:-1]):
        layer = nn.Linear(in_size, out_size)
        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
        # nn.init.kaiming_normal_(layer.weight) ##正态分布
        # nn.init.constant_(layer.bias,0)
        net.append(layer)
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
        elif activation == "mish":
            net.append(Mish())
        elif activation == "tanh":
            net.append(nn.Tanh())

    layer2 = nn.Linear(in_features=layers[-2], out_features=layers[-1])
    nn.init.xavier_uniform_(layer2.weight, gain=nn.init.calculate_gain('relu'))
    # nn.init.kaiming_normal_(layer2.weight)
    # nn.init.constant_(layer2.bias, 0)
    net.append(layer2)

    # for m in net.modules():
    #     if isinstance(m, nn.Linear):
    #         nn.init.kaiming_uniform_(m.weight)
    #         nn.init.constant_(m.bias, 0)

    return net

class GMF(nn.Module):
    def __init__(self, num_cells, num_genes, latent_dim, decoder='MLP'):
        super(GMF, self).__init__()
        self.num_cells = num_cells
        self.num_genes = num_genes
        self.latent_dim = latent_dim
        self.decoder_m = decoder

        self.embedding_cell = nn.Embedding(num_embeddings=self.num_cells, embedding_dim=self.latent_dim)
        self.embedding_gene = nn.Embedding(num_embeddings=self.num_genes, embedding_dim=self.latent_dim)

        if decoder == 'MLP':
            self.decoder = MLPDecoder([self.latent_dim, self.latent_dim//2])
        else:
            self.decoder = ZINBDecoder(self.latent_dim)
        # self.out = nn.Sequential(nn.Linear(in_features=self.latent_dim, out_features=1), nn.Sigmoid())
    def forward(self, cell_indices, gene_indices, sz_factor=None, ge_factor=None):
        cell_embedding = self.embedding_cell(cell_indices)
        gene_embedding = self.embedding_gene(gene_indices)

        element_product = torch.mul(cell_embedding, gene_embedding)#向量 batchsize * feats_dim

        if self.decoder_m == 'ZINB':
            out = self.decoder(element_product, sz_factor, ge_factor)
        else:
            out = self.decoder(element_product)

        return out

class GMF_Dot(nn.Module):
    def __init__(self, num_cells, num_genes, latent_dim, decoder='MLP'):
        super(GMF_Dot, self).__init__()
        self.num_cells = num_cells
        self.num_genes = num_genes
        self.latent_dim = latent_dim
        self.decoder_m = decoder

        self.embedding_cell = nn.Embedding(num_embeddings=self.num_cells, embedding_dim=self.latent_dim)
        self.embedding_gene = nn.Embedding(num_embeddings=self.num_genes, embedding_dim=self.latent_dim)

        if decoder == 'MLP':
            self.decoder = MLPDecoder(layers=[1])
        else:
            self.decoder = ZINBDecoder(1)
        # self.out = nn.Sequential(nn.Linear(in_features=self.latent_dim, out_features=1), nn.Sigmoid())

    def forward(self, cell_indices, gene_indices, sz_factor=None, ge_factor=None):
        cell_embedding = self.embedding_cell(cell_indices)
        gene_embedding = self.embedding_gene(gene_indices)

        element_product1 = torch.mul(cell_embedding, gene_embedding)#向量(2samplesize*batchsize) * feats_dim，每行加权和是最终预测结果
        element_product = element_product1.sum(dim=1)
        element_product = element_product.unsqueeze(1)

        if self.decoder_m == 'ZINB':
            out = self.decoder(element_product, sz_factor, ge_factor)
        else:
            out = self.decoder(element_product)

        return out

class MLP(torch.nn.Module):
    def __init__(self, num_cells, num_genes, layers, decoder='MLP'):
        super(MLP, self).__init__()
        self.num_cells = num_cells
        self.num_genes = num_genes
        self.latent_dim = layers[0]//2
        self.decoder_m = decoder

        self.embedding_cell = torch.nn.Embedding(num_embeddings=self.num_cells, embedding_dim=self.latent_dim)
        self.embedding_gene = torch.nn.Embedding(num_embeddings=self.num_genes, embedding_dim=self.latent_dim)

        self.fc_layers = buildNetwork(layers)

        if decoder == 'MLP':
            self.decoder = MLPDecoder([layers[-1], layers[-1]//2])
        else:
            self.decoder = ZINBDecoder(layers[-1])

    def forward(self, cell_indices, gene_indices, sz_factor=None, ge_factor=None):
        cell_embedding = self.embedding_cell(cell_indices) #shape  batchsize * feats_dim
        gene_embedding = self.embedding_gene(gene_indices) #shape  batchsize * feats_dim
        vector = torch.cat([cell_embedding, gene_embedding], dim=-1)  # the concat latent vector,batchsize * 2feats_dim

        for layer in self.fc_layers:
            vector = layer(vector)
        #print(vector.shape)
        if self.decoder_m == 'ZINB':
            out = self.decoder(vector, sz_factor, ge_factor)
        else:
            out = self.decoder(vector)

        return out


class NeuCF(nn.Module):
    def __init__(self, num_cells, num_genes, latent_dim_mf, layers, decoder='MLP'):
        super(NeuCF, self).__init__()
        self.num_cells = num_cells
        self.num_genes = num_genes
        self.latent_dim_mf = latent_dim_mf
        self.latent_dim_mlp = layers[0]//2  #feats_dim
        self.decoder_m = decoder

        self.embedding_cell_mlp = nn.Embedding(num_embeddings=self.num_cells, embedding_dim=self.latent_dim_mlp)
        self.embedding_gene_mlp = nn.Embedding(num_embeddings=self.num_genes, embedding_dim=self.latent_dim_mlp)
        self.embedding_cell_mf = nn.Embedding(num_embeddings=self.num_cells, embedding_dim=self.latent_dim_mf)
        self.embedding_gene_mf = nn.Embedding(num_embeddings=self.num_genes, embedding_dim=self.latent_dim_mf)

        self.fc_layers = buildNetwork(layers)
        # self.fc_layers.append(nn.Linear(layers[0], layers[1]))
        # self.fc_layers.append(nn.ReLU())
        # self.fc_layers.append(nn.Linear(layers[1], layers[2]))

        self.decoder_dim = latent_dim_mf + layers[-1]  #feats_dim + feats_dim // 2

        if self.decoder_m == 'MLP':
            self.decoder = MLPDecoder([self.decoder_dim, self.decoder_dim // 2])
        else:
            self.decoder = ZINBDecoder(self.decoder_dim)

        # self.out = nn.Sequential(nn.Linear(in_features=layers[-1] + latent_dim_mf, out_features=1), nn.Sigmoid())


    def forward(self, cell_indices, gene_indices, sz_factor=None, ge_factor=None):

        cell_embedding_mlp = self.embedding_cell_mlp(cell_indices)
        gene_embedding_mlp = self.embedding_gene_mlp(gene_indices)
        cell_embedding_mf = self.embedding_cell_mf(cell_indices)
        gene_embedding_mf = self.embedding_gene_mf(gene_indices)
        #按列拼接
        mlp_vector = torch.cat([cell_embedding_mlp, gene_embedding_mlp], dim=-1)  # the concat latent vector
        mf_vector = torch.mul(cell_embedding_mf, gene_embedding_mf) #返回的 batchsize * feats_dim

        for layer in self.fc_layers:
            mlp_vector = layer(mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        #把训练出的vector 去求 mu, disp, pi
        if self.decoder_m == 'ZINB':
            out = self.decoder(vector, sz_factor, ge_factor)
        else:
            out = self.decoder(vector)

        return out

