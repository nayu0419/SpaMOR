import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
class SelfAttention(nn.Module):
    """
    attention_1
    """
    def __init__(self, dropout):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):

        queries = q
        keys = k
        values = v
        n, d = queries.shape
        scores = torch.mm(queries, keys.t()) / math.sqrt(d)
        att_weights = F.softmax(scores, dim=1)
        att_emb = torch.mm(self.dropout(att_weights), values)
        return att_weights, att_emb

class MLP(nn.Module):

    def __init__(self, z_emb, dropout_rate):
        super(MLP, self).__init__()
        self.mlpx = nn.Sequential(
            nn.Linear(z_emb, z_emb),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.mlpi = nn.Sequential(
            nn.Linear(z_emb, z_emb),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
    def forward(self, z_x, z_y):
        q_x = self.mlpx(z_x)
        q_y = self.mlpi(z_y)
        return q_x, q_y


class MGCN(nn.Module):
    def __init__(self, nfeatX, nfeatI, hidden_dims):
        super(MGCN, self).__init__()
        self.GCNA1_1 = GraphConvolution(nfeatX, hidden_dims[0])
        # self.GCNA1_2 = GraphConvolution(hidden_dims[0], hidden_dims[0])
        self.GCNA1_3 = GraphConvolution(hidden_dims[0], hidden_dims[1])
        # self.GCNA1_3 = GraphConvolution(hidden_dims, hidden_dims)
        self.GCNA2_1 = GraphConvolution(nfeatI, hidden_dims[0])
        # self.GCNA2_2 = GraphConvolution(hidden_dims[0], hidden_dims[0])
        self.GCNA2_3 = GraphConvolution(hidden_dims[0], hidden_dims[1])
        # self.GCNA2_3 = GraphConvolution(hidden_dims, hidden_dims)

    def forward(self, x, i, adj):
        emb1 = self.GCNA1_1(x, adj)
        # emb1 = self.GCNA1_2(emb1, a)
        emb1 = self.GCNA1_3(emb1, adj)
        emb2 = self.GCNA2_1(i, adj)
        # emb2 = self.GCNA2_2(emb2, a)
        emb2 = self.GCNA2_3(emb2, adj)
        return emb1, emb2


class Decoder(torch.nn.Module):
    def __init__(self, nhid1, nfeat):
        super(Decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid1, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.Linear(nhid1, nfeat),
            torch.nn.BatchNorm1d(nfeat),
            torch.nn.ReLU()
        )

    def forward(self, emb):
        return self.decoder(emb)

class ZINBDecoder(torch.nn.Module):
    def __init__(self, nhid1, nfeat):
        super(ZINBDecoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid1, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1,  nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)


    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]



class ZIPDecoder(torch.nn.Module):
    def __init__(self, nhid1, nfeat):
        super(ZIPDecoder, self).__init__()
        # 定义网络的结构
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid1, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        # π和λ的全连接层
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)
        # 对λ (mean) 使用激活函数
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)  # 防止数值不稳定

    def forward(self, emb):
        # 前向传播
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))  # 使用sigmoid激活函数计算π
        mean = self.MeanAct(self.mean(x))  # 使用指数激活函数计算λ
        return [pi,None, mean]

class BernoulliDecoder(torch.nn.Module):
    def __init__(self, nhid1, nfeat):
        super(BernoulliDecoder, self).__init__()
        # 定义隐藏层和输出层
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid1, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        # Bernoulli分布的概率p输出层
        self.prob = torch.nn.Linear(nhid1, nfeat)

    def forward(self, emb):
        # 前向传播计算
        x = self.decoder(emb)
        # 使用sigmoid函数将输出转换为概率p
        p = torch.sigmoid(self.prob(x))
        return p

    # loss_fn = torch.nn.BCELoss()
    #
    # # 计算损失
    # loss = loss_fn(y_pred, y_true)

class MixtureNBLogNormal(torch.nn.Module):
    def __init__(self, nhid1, nfeat):
        super(MixtureNBLogNormal, self).__init__()
        # 解码器网络
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid1, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )

        # 背景和前景的神经网络参数
        self.back_mean = torch.nn.Linear(nhid1, nfeat)  # m_i^back
        self.back_log_sigma = torch.nn.Linear(nhid1, nfeat)  # log(σ_i^back)
        self.pi = torch.nn.Linear(nhid1, nfeat)  # π_i^protein
        self.alpha = torch.nn.Linear(nhid1, nfeat)  # α_i^protein
        self.back_dispersion = torch.nn.Linear(nhid1, nfeat)  # ϕ for background
        self.fore_dispersion = torch.nn.Linear(nhid1, nfeat)  # ϕ for foreground

    def forward(self, emb, y_protein):
        # 解码器生成隐藏层表示
        x = self.decoder(emb)

        # 背景强度的均值和log标准差
        m_back = self.back_mean(x)  # 背景的 m_i^back
        log_sigma_back = self.back_log_sigma(x)  # 背景的 log(σ_i^back)
        sigma_back = torch.exp(log_sigma_back)  # 背景的 σ_i^back

        # 通过log-normal分布对背景强度 ν_i^back 进行采样
        eps = torch.randn_like(m_back)  # 采样标准正态分布
        v_back = torch.exp(m_back + sigma_back * eps)  # 背景强度 ν_i^back ~ LogNormal

        # 计算前景强度
        alpha_protein = torch.exp(self.alpha(x))  # α_i^protein
        v_fore = (1 + alpha_protein) * v_back  # 前景强度 ν_i^fore

        # 零膨胀参数
        pi_protein = torch.sigmoid(self.pi(x))  # π_i^protein

        # 前景和背景的离散度
        dispersion_back = F.softplus(self.back_dispersion(x))  # 背景的 dispersion
        dispersion_fore = F.softplus(self.fore_dispersion(x))  # 前景的 dispersion

        # 负二项分布的计算（背景和前景）
        nb_back = self.negative_binomial(y_protein, v_back, dispersion_back)
        nb_fore = self.negative_binomial(y_protein, v_fore, dispersion_fore)

        # 使用更稳定的 logsumexp 计算混合负二项分布的对数似然
        mixture_nb = torch.logsumexp(torch.stack([
            torch.log(pi_protein + 1e-8) + nb_back,
            torch.log(1 - pi_protein + 1e-8) + nb_fore
        ]), dim=0)

        # 返回log likelihood的均值
        return torch.mean(mixture_nb)

    def negative_binomial(self, y, mean, dispersion):
        """计算负二项分布的log likelihood"""
        eps = 1e-8  # 防止log(0)的情况
        log_prob = (
            torch.lgamma(y + dispersion) - torch.lgamma(dispersion) - torch.lgamma(y + 1)
            + dispersion * (torch.log(dispersion + eps) - torch.log(dispersion + mean + eps))
            + y * (torch.log(mean + eps) - torch.log(dispersion + mean + eps))
        )
        return log_prob


class Omics_label_Predictor(nn.Module):
    def __init__(self, z_emb_size1):
        super(Omics_label_Predictor, self).__init__()

        # input to first hidden layer
        self.hidden1 = nn.Linear(z_emb_size1, 5)

        # second hidden layer and output
        self.hidden2 = nn.Linear(5, 2)

    def forward(self, X):

        X = F.sigmoid(self.hidden1(X))
        y_pre = F.softmax(self.hidden2(X), dim=1)
        # y_pre = F.sigmoid(self.hidden2(X))
        return y_pre


