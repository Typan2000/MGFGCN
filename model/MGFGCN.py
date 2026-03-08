from model.temporal_block import *
from model.spatial_block import *
from sklearn.cluster import SpectralClustering


class SpectralClust(nn.Module):
    def __init__(self, n_clusters, device):
        super(SpectralClust, self).__init__()
        self.n_clusters = n_clusters
        self.device = device
        self.spectral = SpectralClustering(n_clusters = n_clusters, affinity = 'precomputed',
                                           assign_labels = 'kmeans', random_state = 9, n_init = 10)

    def forward(self, distance_matrix):
        sigma = torch.median(distance_matrix)
        similarity_matrix = torch.exp(-distance_matrix ** 2 / (2 * sigma ** 2))
        labels = self.spectral.fit_predict(similarity_matrix)
        labels = torch.from_numpy(labels).to(self.device)
        return labels

class Region_Gragh_Construct(nn.Module):
    def __init__(self, region_num):
        super(Region_Gragh_Construct, self).__init__()
        self.region_num = region_num

    def forward(self, x, labels):
        B, C, N, D = x.shape
        device = x.device
        labels_expanded = labels.view(1, 1, x.size(2), 1).expand(B, C, -1, D).long()
        output = torch.zeros((B, C, self.region_num, D), device=device)
        output = output.scatter_reduce(dim=2, index=labels_expanded, src=x, reduce="mean", include_self=True)
        return output

class X_Construct(nn.Module):
    def __init__(self, node_num):
        super(X_Construct, self).__init__()
        self.node_num = node_num

    def forward(self, x, labels):
        B, C, K, D = x.shape
        labels = labels.view(1, 1, -1, 1).expand(B, C, -1, D).long()
        output = torch.gather(x, dim=2, index=labels)
        return output

class ST_block(nn.Module):
    def __init__(self, drop, node_num, spatial_dim, N_t, temporal_dim, layers, hid_dim, device):
        super(ST_block, self).__init__()
        self.spatial_block = spatial_block(drop,  hid_dim, node_num, spatial_dim, N_t, device)
        self.temporal_block = temporal_block(drop, node_num, N_t, layers, temporal_dim, hid_dim, device)
        self.linear = torch.nn.Conv2d(hid_dim, hid_dim, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x, i, t):
        x_f = self.linear(x)
        x_s = self.spatial_block(x, t)
        x_t = self.temporal_block(x, i, t)
        x = torch.cat([x_f, x_s, x_t], dim=1)
        return x


class MGFGCN(nn.Module):
    def __init__(self, device, config, distance_matrix, skip_channels):
        super(MGFGCN, self).__init__()
        self.drop = config['drop']
        self.layers = config['layers']
        self.node_num = config['node_num']
        self.hid_dim = config['hidden_dimension']
        self.residual_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.SpectralClust = SpectralClust(config['region_num'], device)
        self.labels = self.SpectralClust(distance_matrix)
        self.result_fuse = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.Region_Gragh_Construct = Region_Gragh_Construct(config['region_num'])
        self.X_Construct = X_Construct(self.node_num)
        self.ST_block_cu = ST_block(self.drop, config['region_num'], config['spatial_dim_region'],
                                 config['N_t'], config['temporal_dim_region'], config['layers'], self.hid_dim, device)
        self.ST_block_xi = ST_block(self.drop, config['node_num'], config['spatial_dim_node'],
                                 config['N_t'], config['temporal_dim_node'], config['layers'], self.hid_dim, device)

        for layers in range(self.layers):
            self.residual_convs.append(nn.Conv2d(in_channels=self.hid_dim,
                                                 out_channels=self.hid_dim,
                                                 kernel_size=(1, 1)))
            self.skip_convs.append(nn.Conv2d(in_channels=self.hid_dim,
                                             out_channels=skip_channels,
                                             kernel_size=(1, 1)))
            self.bn.append(nn.BatchNorm2d(self.hid_dim))

            self.result_fuse.append(torch.nn.Conv2d(self.hid_dim * 6, self.hid_dim, kernel_size=(1, 1), padding=(0, 0),
                                stride=(1, 1), bias=True))

        self.start_conv = nn.Conv2d(in_channels=config['input_dim'],
                                    out_channels=self.hid_dim,
                                    kernel_size=(1, 1))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=self.hid_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=config['output_dim'] * self.hid_dim,
                                    out_channels=config['output_dim'],
                                    kernel_size=(1, 1),
                                    bias=True)

    def forward(self, input, t_i):
        x = input
        x = self.start_conv(x)
        skip = 0
        x_r = self.Region_Gragh_Construct(x, self.labels)
        for i in range(self.layers):
            residual = x
            x_region = self.ST_block_cu(x_r, i, t_i)
            x_region = self.X_Construct(x_region, self.labels)
            x_node = self.ST_block_xi(x, i, t_i)
            x = torch.cat([x_region, x_node], dim=1)
            x = F.relu(x)
            x = self.result_fuse[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = torch.transpose(x, 3, 2)
        x = torch.reshape(x, (x.size(0), x.size(1) * x.size(2), x.size(3), 1))
        x = self.end_conv_2(x)
        return x
