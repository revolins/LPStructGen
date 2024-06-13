import torch
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor,masked_select_nnz
from torch.nn import Linear, Parameter, BatchNorm1d as BN
from typing import Final

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

class SIGNBaseClass(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K, dropout, edge_drop):
        super(SIGNBaseClass, self).__init__()

        self.K = K
        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(self.K + 1):
            self.lins.append(Linear(in_channels, hidden_channels))
            self.bns.append(BN(hidden_channels))
        self.lin_out = Linear((K + 1) * hidden_channels, out_channels)
        self.dropout = dropout
        self.edge_drop = edge_drop
        self.adj_t = None
        self.adjdrop = DropAdj(edge_drop)

    def reset_parameters(self):
        for lin, bn in zip(self.lins, self.bns):
            lin.reset_parameters()
            bn.reset_parameters()

    def cache_adj_t(self, edge_index, num_nodes):
        row, col = edge_index
        adj_t = SparseTensor(row=col, col=row,
                             sparse_sizes=(num_nodes, num_nodes))
        adj_t = self.adjdrop(adj_t)
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

    def forward(self, *args):
        raise NotImplementedError
    
class SIGNEmbedding(SIGNBaseClass):
    def __init__(self, in_channels, hidden_channels, out_channels, K, dropout, edge_drop):
        super(SIGNEmbedding, self).__init__(in_channels, hidden_channels, out_channels, K, dropout, edge_drop)

    def forward(self, x, adj_t, num_nodes):
        if self.adj_t is None:
            print("WARNING ADJ_T IS NOT ACCESSED IN TC")
            exit()
            self.adj_t = self.cache_adj_t(adj_t, num_nodes)
        hs = []
        for lin, bn in zip(self.lins, self.bns):
            h = lin(x)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            hs.append(h)
            x = self.adj_t @ x
        h = torch.cat(hs, dim=-1)
        x = self.lin_out(h)
        return x


class SIGN(SIGNBaseClass):
    def __init__(self, in_channels, hidden_channels, out_channels, K, dropout, edge_drop):
        super(SIGN, self).__init__(in_channels, hidden_channels, out_channels, K, dropout, edge_drop)

    def forward(self, xs):
        """
        apply the sign feature transform where each component of the polynomial A^n x is treated independently
        @param xs: [batch_size, 2, n_features * (K + 1)]
        @return: [batch_size, 2, hidden_dim]
        """
        xs = torch.tensor_split(xs, self.K + 1, dim=-1)
        hs = []
        # split features into k+1 chunks and put each tensor in a list
        for lin, bn, x in zip(self.lins, self.bns, xs):
            h = lin(x)
            # the next line is a fuggly way to apply the same batch norm to both source and destination edges
            h = torch.cat((bn(h[:, 0, :]).unsqueeze(1), bn(h[:, 1, :]).unsqueeze(1)), dim=1)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            hs.append(h)
        h = torch.cat(hs, dim=-1)
        x = self.lin_out(h)
        return x