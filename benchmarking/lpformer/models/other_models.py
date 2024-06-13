import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_sparse.matmul import spmm_max, spmm_mean, spmm_add

from typing import Final
from torch_sparse import SparseTensor, masked_select_nnz

class DropAdj(torch.nn.Module):
    doscale: Final[bool]
    def __init__(self, dp: float = 0.0, doscale=True) -> None:
        super().__init__()
        self.dp = dp
        self.register_buffer("ratio", torch.tensor(1/(1-dp)))
        self.doscale = doscale

    def forward(self, adj: SparseTensor)->SparseTensor:
        if self.dp < 1e-6 or not self.training:
            return adj
        mask = torch.rand_like(adj.storage.col(), dtype=torch.float) > self.dp
        adj = masked_select_nnz(adj, mask, layout="coo") 
        if self.doscale:
            if adj.storage.has_value():
                adj.storage.set_value_(adj.storage.value()*self.ratio, layout="coo")
            else:
                adj.fill_value_(1/(1-self.dp), dtype=torch.float)
        return adj


class GCN(torch.nn.Module):
    """
    GCN Model
    """
    def __init__(
        self, 
        in_channels, 
        hidden_channels, 
        out_channels, 
        num_layers,
        dropout,
        edge_drop,
        residual=False,
        cached=False,
        normalize=True,
        layer_norm=True,
        relu=True,
    ):
        super(GCN, self).__init__()

        self.relu = relu
        self.convs = torch.nn.ModuleList()
        self.adjdrop = DropAdj(edge_drop)

        if num_layers == 1:
            hidden_channels = out_channels
            
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=cached, normalize=normalize))
        
        if layer_norm:
            self.lns = torch.nn.ModuleList()
            self.lns.append(nn.LayerNorm(hidden_channels))
        else:
            self.lns = None

        if num_layers > 1:    
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=normalize))
                self.lns.append(nn.LayerNorm(hidden_channels))

            self.convs.append(GCNConv(hidden_channels, out_channels, cached=cached, normalize=normalize))
            if layer_norm:
                self.lns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.residual = residual


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
     

    def forward(self, x, adj_t):
        """
        Propagation
        """
        adj_t_drop = self.adjdrop(adj_t)
        for i, conv in enumerate(self.convs):
            xi = conv(x, adj_t_drop)
            xi = self.lns[i](xi) if self.lns is not None else xi
            xi = F.dropout(xi, p=self.dropout, training=self.training)
            xi = F.relu(xi) if self.relu else xi

            if self.residual and x.shape[-1] == xi.shape[-1]:
                x = x + xi
            else:
                x = xi

        return x



class MLP(nn.Module):
    """
    L Layer MLP
    """
    def __init__(
        self, 
        num_layers,
        in_channels, 
        hid_channels, 
        out_channels, 
        drop=0,
        norm="layer",
        sigmoid=False,
        bias=True
    ):
        super().__init__()
        self.dropout = drop
        self.sigmoid = sigmoid

        if norm == "batch":
            self.norm = nn.BatchNorm1d(hid_channels)
        elif norm == "layer":
            self.norm = nn.LayerNorm(hid_channels)
        else:
            self.norm = None

        self.linears = torch.nn.ModuleList()

        if num_layers == 1:
            self.linears.append(nn.Linear(in_channels, out_channels, bias=bias))
        else:
            self.linears.append(nn.Linear(in_channels, hid_channels, bias=bias))
            for _ in range(num_layers-2):
                self.linears.append(nn.Linear(hid_channels, hid_channels, bias=bias))
            self.linears.append(nn.Linear(hid_channels, out_channels, bias=bias))
    

    def reset_parameters(self):
        for lin in self.linears:
            lin.reset_parameters()
        
        if self.norm is not None:
            self.norm.reset_parameters()


    def forward(self, x):
        """
        Forward Pass
        """
        for i, lin in enumerate(self.linears[:-1]):
            x = lin(x)
            x = self.norm(x) if self.norm is not None else x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.linears[-1](x)
        x = x.squeeze(-1)

        return torch.sigmoid(x) if self.sigmoid else x



class mlp_score(torch.nn.Module):
    """
    MLP score function
    """
    def __init__(
        self, 
        in_channels, 
        hidden_channels, 
        out_channels, 
        num_layers,
        dropout=0
    ):
        super(mlp_score, self).__init__()

        self.lins = torch.nn.ModuleList()
        if num_layers == 1: 
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout


    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()


    def forward(self, x):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x).squeeze(-1)





