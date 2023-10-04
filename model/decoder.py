import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPDecoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        """MLP decoder for link prediction
        predict link existence
        """
        self.fc_layers = torch.nn.ModuleList()
        for in_size, out_size in zip(layers[:-1], layers[1:]):
            self.fc_layers.append(nn.Linear(in_size, out_size))
            self.fc_layers.append(nn.ReLU())
            # self.fc_layers.append(nn.BatchNorm1d(out_size))
            # self.fc_layers.append(nn.Dropout(p=dropout))

        self.out = nn.Sequential(nn.Linear(in_features=layers[-1], out_features=1), nn.Sigmoid())

    def forward(self, vector):
        for layer in self.fc_layers:
            vector = layer(vector)
        out = self.out(vector)

        return out


class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x)-1., min=1e-5, max=1e6)

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)  #softplus函数可以用来产生正态分布的β和σ参数，=ln(1+exp(x))
        #torch.clamp将F.softplus(x)的每个元素的夹紧到区间 [min,max],并返回到一个新张量

class ZINBDecoder(nn.Module):
    def __init__(self, feats_dim):
        super().__init__()
        """ZINB decoder for link prediction
        predict link existence (not edge type)
        """
        self.dec_mean = nn.Sequential(nn.Linear(feats_dim, 1), nn.Sigmoid())
        self.dec_disp = nn.Linear(feats_dim, 1)
        self.dec_disp_act = DispAct()
        self.dec_pi = nn.Sequential(nn.Linear(feats_dim, 1), nn.Sigmoid())
        self.dec_mean_act = MeanAct()

    def forward(self, vector, sz_factor, ge_factor):
        """
        Paramters
        pred : torch.FloatTensor
            shape : (n_cells, 1)
        """
        mu_ = self.dec_mean(vector)  #一个数值
        disp_ = self.dec_disp(vector)
        pi = self.dec_pi(vector)

        mu_ = ge_factor * mu_
        mu = sz_factor * self.dec_mean_act(mu_)
        disp = self.dec_disp_act(ge_factor * disp_)

        return mu, disp, pi

class ZINBDecoder2(nn.Module):
    def __init__(self, feats_dim):
        super().__init__()
        """ZINB decoder for link prediction
        predict link existence (not edge type)
        """
        self.dec_mean = nn.Sequential(nn.Linear(int(feats_dim/3), 1), nn.Sigmoid())
        self.dec_mean_act = MeanAct()
        self.dec_disp = nn.Sequential(nn.Linear(int(feats_dim/3), 1), DispAct())
        self.dec_pi = nn.Sequential(nn.Linear(int(feats_dim/3), 1), nn.Sigmoid())
        # self.dec_pi = nn.Linear(feats_dim, 1)   ## logits


    def forward(self, vector, sz_factor, ge_factor):

        h_d = torch.chunk(vector, chunks=3, dim=1)
        mu_ = self.dec_mean(h_d[0])
        disp = self.dec_disp(h_d[1])
        pi = self.dec_pi(h_d[2])

        mu_ = ge_factor * mu_
        mu = sz_factor * self.dec_mean_act(mu_)

        return mu, disp, pi
