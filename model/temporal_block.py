import torch
import torch.nn as nn
import torch.nn.functional as F

class temporal_graph_generator(nn.Module):
    def __init__(self, layers, node_num, N_t, temporal_dim, device):
        super(temporal_graph_generator, self).__init__()
        self.layers = layers
        self.nodetime = nn.Parameter(torch.randn(layers, N_t, temporal_dim[0]).to(device),
                                    requires_grad=True).to(device)
        self.nodenum = nn.Parameter(torch.randn(layers, node_num, temporal_dim[1]).to(device),
                                     requires_grad=True).to(device)
        self.k = nn.Parameter(torch.randn(layers, temporal_dim[0], temporal_dim[1], temporal_dim[2], temporal_dim[3]).to(device),
                              requires_grad=True).to(device)
    def graph_generator(self, i, t):
        nodetime = self.nodetime[i][t]
        nodenum = self.nodenum[i]
        k = self.k[i]
        adp1 = torch.einsum("ad, defg->aefg", [nodetime, k])
        adp2 = torch.einsum("he, aefg->ahfg", [nodenum, adp1])
        adp = torch.einsum("ahbc, ahdc->ahbd", [adp2, adp2])
        return adp

    def forward(self, i, t_i):
        adp = self.graph_generator(i, t_i)
        adp = F.softmax(F.relu(adp), dim=3)
        return adp

class temporal_block(nn.Module):
    def __init__(self, drop, node_num, N_t, layers, temporal_dim, hid_dim, device):
        super(temporal_block, self).__init__()
        self.drop = drop
        self.temporal_graph_generator = temporal_graph_generator(layers, node_num, N_t, temporal_dim, device)

    def forward(self, x, i, t_i):
        adp = self.temporal_graph_generator(i, t_i)
        out = torch.einsum("ncvl, nvlw->ncvw", [x, adp])
        out = F.dropout(out, self.drop, training=self.training)
        return out
