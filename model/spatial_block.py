import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Graph_Generator(nn.Module):
    def __init__(self, spatial_dim, hid_dim, node_num, N_t, drop, device):
        super(Graph_Generator, self).__init__()
        self.drop = drop
        self.norm = 1 / sqrt(hid_dim)
        self.weight_t = nn.Parameter(torch.randn(N_t, spatial_dim[0]).to(device),
                                     requires_grad=True).to(device)
        self.weight_n = nn.Parameter(torch.randn(node_num, spatial_dim[1]).to(device),
                                     requires_grad=True).to(device)
        self.k = nn.Parameter(torch.randn(spatial_dim[0], spatial_dim[1], spatial_dim[2]).to(device),
                                     requires_grad=True).to(device)
        self.n_emb = nn.Parameter(torch.randn(1, hid_dim, node_num, 12).to(device),
                                     requires_grad=True).to(device)

    def forward(self, x, t_i):
        x = x + self.n_emb
        weight_t = self.weight_t[t_i]
        weight_n = self.weight_n
        k = self.k
        adp1 = torch.einsum("ab, bcd->acd", [weight_t, k])
        adp2 = torch.einsum("ec, acd->aed", [weight_n, adp1])
        adp2 = F.softmax(adp2, dim=2)
        e_n = torch.einsum('ncvl,nvl->ncv', x, adp2)
        adp = torch.matmul(e_n.transpose(1, 2), e_n) * self.norm
        adp = F.softmax(F.relu(adp), dim=2)
        return adp

class spatial_block(nn.Module):
    def __init__(self, drop, hid_dim, node_num, spatial_dim, N_t, device):
        super(spatial_block, self).__init__()
        self.drop = drop
        self.spatial_graph = Graph_Generator(spatial_dim, hid_dim, node_num, N_t, drop, device)
        
    def forward(self, x, t):
        adp = self.spatial_graph(x, t)
        out = torch.einsum('ncvl,nvw->ncwl', (x, adp))
        out = F.dropout(out, self.drop, training=self.training)
        return out
