from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
from utils import *
from torch import nn

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dp):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=True, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=True, normalize=False))

        self.dropout = dp

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, adj_t)

        return x

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dp):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dp

    def forward(self, x, adj_t=None):

        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lins[-1](x)

        return x
    
    def score(self, x, adj_t=None):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
        
        x = self.lins[-1](x)

        return torch.sigmoid(x)

class DropAdj(nn.Module):
    def __init__(self, dp: float = 0.0, doscale=True):
        super().__init__()
        self.dp = dp
        self.register_buffer("ratio", torch.tensor(1/(1 - dp)))
        self.doscale = doscale

    def forward(self, adj):
        if self.dp < 1e-6 or not self.training:
            return adj

        mask = torch.rand_like(adj.storage.col(), dtype=torch.float) > self.dp
        adj = torch_sparse.masked_select_nnz(adj, mask, layout="coo")

        if self.doscale:
            if adj.storage.has_value():
                adj.storage.set_value_(adj.storage.value()*self.ratio, layout="coo")
            else:
                adj.fill_value_(1/(1-self.dp), dtype=torch.float)

        return adj