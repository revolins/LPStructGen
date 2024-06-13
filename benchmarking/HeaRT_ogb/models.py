import argparse

import torch
import torch.nn.functional as F

from torch_sparse import SparseTensor
from torch_sparse import sum as sparse_sum

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_sparse import SparseTensor
from baseline_models.BUDDY.model import BUDDY

from torch.nn import (Linear, LayerNorm)

from typing import Final
import torch_sparse


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, x_j = None):
        if x_j is not None:
            x = x * x_j
        for idx, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
    
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
        adj = torch_sparse.masked_select_nnz(adj, mask, layout="coo") 
        if self.doscale:
            if adj.storage.has_value():
                adj.storage.set_value_(adj.storage.value()*self.ratio, layout="coo")
            else:
                adj.fill_value_(1/(1-self.dp), dtype=torch.float)
        #print("Adjacency Called in AdjDrop - GCN: ", adj.has_value())
        return adj

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, edge_drop, mlp_layer=None, head=None, node_num=None,  cat_node_feat_mf=False, data_name=None):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.adjdrop = DropAdj(edge_drop)

        if data_name == 'ogbl-citation2':
            if num_layers == 1:
                self.convs.append(GCNConv(in_channels, out_channels,normalize=False ))

            elif num_layers > 1:
                self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False))
                
                for _ in range(num_layers - 2):
                    self.convs.append(
                        GCNConv(hidden_channels, hidden_channels, normalize=False))
                self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False))
        
        else:
            if num_layers == 1:
                self.convs.append(GCNConv(in_channels, out_channels))
                self.lins.append(LayerNorm(hidden_channels))

            elif num_layers > 1:
                self.convs.append(GCNConv(in_channels, hidden_channels))
                self.lins.append(LayerNorm(hidden_channels))
                
                for _ in range(num_layers - 2):
                    self.convs.append(
                        GCNConv(hidden_channels, hidden_channels))
                    self.lins.append(LayerNorm(hidden_channels))
                self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout
        # self.p = args
       
        self.invest = 1

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
     

    def forward(self, x, adj_t):
        #print(f'x in early GCN: {x.shape}')
        if self.invest == 1:
            print('layers in gcn: ', len(self.convs))
            self.invest = 0
            
        for conv, lin in zip(self.convs[:-1], self.lins):
            x = conv(x, self.adjdrop(adj_t))
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        #print(f'x in late GCN: {x.shape}')
        return x

class Ensemble(torch.nn.Module):
    def __init__(self, model1, model2):
        super(Ensemble, self).__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.model1 = model1
        for param in model1.parameters():
            param.requires_grad = False
        self.model2 = model2
        for param in model2.parameters():
            param.requires_grad = False
        
    def reset_parameters(self):
        self.linear.reset_parameters()        

    def forward(self,x , edges, adj):
        x = torch.cat([self.model1(x , edges, adj), self.model2(x , edges, adj) ], dim = 1 )
        x = self.linear(x)
        return torch.sigmoid(x)    

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)
    
class LinkGNN(torch.nn.Module):
    def __init__(self, emb, gnn, linkpred):
        super(LinkGNN, self).__init__()
        self.gnn = gnn
        self.linkpred = linkpred
        self.emb = emb
    
    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.linkpred.reset_parameters()
        if self.emb is not None:
            self.emb.reset_parameters()    
        
    def forward(self,x , edges, adj):
        #print(f"x before concatenate in LinkGNN: {x.shape}")
        if x is None:
            x = self.emb.weight
        #     x = torch.cat([self.emb.weight, x], dim=1)
        # print(f"X in LinkGNN: {x.shape}")
        h = self.gnn(x, adj)
        return self.linkpred(h[edges[0]], h[edges[1]])

class CommonNeighborsPredictor(torch.nn.Module):
    def __init__(self, emb, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, model_type='weighted'):
        super(CommonNeighborsPredictor, self).__init__()
        assert model_type in ['mlpcos', 'simplecos', 'adamic', 'simple', 'adamic_ogb', "resource_allocation", 'katz']
        self.type = model_type
        if self.type == 'mlpcos':
            self.mlp = MLP(in_channels, hidden_channels, out_channels, num_layers,
                     dropout)
            self.emb = emb
        else:
            self.mlp = torch.nn.Identity()
            self.emb = emb
        
    def reset_parameters(self):
        if self.type == 'mlpcos':
            self.mlp.reset_parameters()
        if self.emb is not None:
            self.emb.reset_parameters()

    def forward(self, x, edges, adj):
        if (x is None) and (self.type in ['mlpcos', 'simplecos']):
            x = self.emb.weight
        elif self.emb is not None:
            x = torch.cat([self.emb.weight, x], dim=1)
            
        if self.type in ['adamic_ogb', "resource_allocation", 'katz']:
            return None
        common_neighbors = adj[edges[0]].to_torch_sparse_coo_tensor().mul(adj[edges[1]].to_torch_sparse_coo_tensor())
        
        
        if self.type == 'simple':
            if common_neighbors._nnz() == 0: 
                return torch.zeros((common_neighbors.shape[0])).to(common_neighbors.device)
            return torch.sparse.sum(common_neighbors, 1).to_dense()
        
        common_neighbors = common_neighbors.indices()  
        
        sparse_sizes = adj[edges[0]].sparse_sizes()
        degrees = adj.sum(-1) + 1e-6
        
        if self.type == 'adamic':
            weights = SparseTensor.from_edge_index(common_neighbors, 
                                                   1./torch.log(degrees[common_neighbors[1]]), 
                                                   sparse_sizes = sparse_sizes) # sparse(Q, N)
            weights = sparse_sum(weights, 1)
            return torch.sigmoid(weights)
        
        left_neighbors = common_neighbors.clone()
        left_neighbors[0] = edges[0][common_neighbors[0]]
        
        right_neighbors = common_neighbors.clone()
        right_neighbors[0] = edges[1][common_neighbors[0]]

        x =  x + (adj @ x) / degrees.unsqueeze(1)
        
#         x = self.mlp(x)
        left_edge_features = x[left_neighbors] # (2, Q * sparse(N), F)
        right_edge_features = x[right_neighbors] # (2, Q * sparse(N), F)
        
        left_edge_weights = F.cosine_similarity(left_edge_features[0], left_edge_features[1], dim=1)  # (Q * sparse(N))
        right_edge_weights = F.cosine_similarity(right_edge_features[0], right_edge_features[1], dim=1)                          
                
        weights = SparseTensor.from_edge_index(common_neighbors, 
                                               left_edge_weights * right_edge_weights, 
                                               sparse_sizes = sparse_sizes) # sparse(Q, N)
        weights = sparse_sum(weights, 1)
        return torch.sigmoid(weights) 


def build_model(args, data, device):   
    assert args.model in ['buddy', 'gcn', 'mlpcos', 'simplecos', 'adamic', 'simple', 'adamic_ogb',  "resource_allocation",  'katz']
    emb, gnn, linkpred = None, None, None
    
    if 'ppa' in args.dataset:
        input_dim = 58
    else:
        input_dim = 128
     
    if args.model == 'gcn':
        gnn = GCN(
            input_dim, args.hidden_channels,
            args.hidden_channels, args.num_layers, 
            args.dropout, args.edge_drop).to(device)
        linkpred = LinkPredictor(
            args.hidden_channels, args.hidden_channels, 
            1, args.num_layers, 
            args.dropout).to(device)
        model = LinkGNN(emb, gnn, linkpred)    
    elif args.model == "buddy" and "ppa" in args.dataset.lower():
        print("Reached BUDDY - ppa", flush=True)
        model = BUDDY(args, 58, node_embedding=None)
    elif args.model == "buddy" and "ppa" not in args.dataset.lower():
        print("Reached BUDDY - collab", flush=True)
        model = BUDDY(args, 128, node_embedding=None)  
    elif args.model in ['mlpcos', 'simplecos', 'adamic', 'simple', 'adamic_ogb', 'katz', "resource_allocation"]:
        # 'adamic', 'simple' should have 0 input dim
        # 'adamic_ogb' refers to the ogb implementation; in this case the model is not used
        model = CommonNeighborsPredictor(emb,
            input_dim, args.hidden_channels, 
            args.hidden_channels, args.num_layers, 
            args.dropout, model_type=args.model).to(device)

    return gnn, linkpred, emb, model


def default_model_configs(args):
    default_dict = {
        "num_layers": None,
        "input_dim": None,
        "hidden_channels": None,
        "out_channels": None,
        "dropout": None,
        "batch_size": None,
        "lr": None,
        "epochs": None,
        "use_feature": None,
        "use_learnable_embedding": None,
    }  

    default_dict["use_feature"] = True
    default_dict["use_learnable_embedding"] = False
    default_dict["batch_size"] = 65536  

    if args.model == 'gcn':
        print("OVERRIDE - GCN STATE_DICT")
        default_dict["num_layers"] = 3
        if 'ppa' in args.dataset:
            default_dict["input_dim"] = 58
        else:
            default_dict["input_dim"] = 128
        default_dict["hidden_channels"] = 128
        default_dict["epochs"] = 1000

        gcn_drp1_lr1e3_list = ["collab_SP_00_026_036", "collab_PA_0_100_200","collab_PA_0_150_250",\
                               "ppa_SP_00_026_036", "ppa_CN_4_2_0", "ppa_CN_5_3_0", "ppa_PA_0_10000_20000",\
                                "collab_PA_100_50_0", "collab_PA_200_100_0", "collab_PA_250_150_0",\
                                "ppa_PA_10000_5000_0", "ppa_PA_20000_10000_0", "ppa_PA_25000_15000_0"]
        gcn_drp3_lr1e3_list = ["collab_CN_0_1_2", "collab_CN_0_2_4", "collab_CN_0_3_5", "collab_SP_00_017_026",\
                                "ppa_SP_00_017_026", "collab_PA_0_50_100",\
                                ]
        gcn_drp1_lr1e2_list = ["collab_CN_2_1_0", "ppa_CN_0_3_5",\
                               "ppa_SP_036_026_00"]
        gcn_drp3_lr1e2_list = ["collab_CN_4_2_0", "collab_CN_5_3_0", "collab_SP_026_017_00", "collab_SP_036_026_00",\
                               "ppa_CN_0_1_2", "ppa_CN_0_2_4", "ppa_CN_2_1_0", "ppa_SP_026_017_00", "ppa_PA_0_5000_10000",\
                                "ppa_PA_0_15000_25000",]

        if any(splt_t in args.dataset for splt_t in gcn_drp3_lr1e3_list):
            print("OVERRIDE - DROP - 0.3, LR = 1e-3")
            default_dict["dropout"] = 0.3
            default_dict["lr"] = 0.001
        if any(splt_t in args.dataset for splt_t in gcn_drp1_lr1e3_list):
            print("OVERRIDE - DROP - 0.1, LR = 1e-3")
            default_dict["dropout"] = 0.1
            default_dict["lr"] = 0.001
        if any(splt_t in args.dataset for splt_t in gcn_drp3_lr1e2_list):
            print("OVERRIDE - DROP - 0.3, LR = 1e-2")
            default_dict["dropout"] = 0.3
            default_dict["lr"] = 0.01
        if any(splt_t in args.dataset for splt_t in gcn_drp1_lr1e2_list):
            print("OVERRIDE - DROP - 0.1, LR = 1e-2")
            default_dict["dropout"] = 0.1
            default_dict["lr"] = 0.01
        
        #input_dim only works on neural models "input_dim",
        over_write_list = ["num_layers", "input_dim",  "hidden_channels", "dropout",\
                            "batch_size", "lr", "epochs", "use_feature", "use_learnable_embedding",\
                            ]
    
    elif args.model == 'buddy':
        print("OVERRIDE - BUDDY STATE_DICT")
        default_dict["num_layers"] = 3
        default_dict["input_dim"] = 128
        default_dict["hidden_channels"] = 256
        default_dict["epochs"] = 100
        default_dict['label_dropout'] = 0.1
        default_dict['feature_dropout'] = 0.1
        
        #, , 
        bdy_drp1_lr1e3_list = ["collab_CN_0_3_5", "ppa_SP_00_026_036", "collab_CN_2_1_0", "collab_CN_4_2_0", "collab_CN_5_3_0",\
                               "collab_SP_036_026_00", "ppa_CN_4_2_0", "collab_PA_100_50_0", "collab_PA_200_100_0",\
                                "ppa_CN_5_3_0", "ppa_SP_036_026_00"]
        bdy_drp3_lr1e3_list = ["collab_CN_0_2_4", "collab_SP_00_026_036", "ppa_PA_0_15000_25000"]
        bdy_drp1_lr1e2_list = ["collab_CN_0_1_2", "ppa_SP_00_017_026",\
                               "collab_SP_026_017_00", "ppa_SP_026_017_00",\
                                "ppa_PA_0_5000_10000", "ppa_PA_0_10000_20000",]
        bdy_drp3_lr1e2_list = ["collab_SP_00_017_026", "collab_PA_0_50_100", "collab_PA_0_100_200", "collab_PA_0_150_250",\
                                "ppa_CN_0_1_2", "ppa_CN_0_2_4", "ppa_CN_0_3_5", "ppa_CN_2_1_0", "collab_PA_250_150_0",\
                                    "ppa_PA_10000_5000_0", "ppa_PA_20000_10000_0", "ppa_PA_25000_15000_0"]

        if any(splt_t in args.dataset for splt_t in bdy_drp3_lr1e3_list):
            print("OVERRIDE - DROP - 0.3, LR = 1e-3")
            default_dict["label_dropout"] = 0.3
            default_dict["feature_dropout"] = 0.3
            default_dict["lr"] = 0.001
        if any(splt_t in args.dataset for splt_t in bdy_drp1_lr1e3_list):
            print("OVERRIDE - DROP - 0.1, LR = 1e-3")
            default_dict["label_dropout"] = 0.1
            default_dict["feature_dropout"] = 0.1
            default_dict["lr"] = 0.001
        if any(splt_t in args.dataset for splt_t in bdy_drp3_lr1e2_list):
            print("OVERRIDE - DROP - 0.3, LR = 1e-2")
            default_dict["label_dropout"] = 0.3
            default_dict["feature_dropout"] = 0.3
            default_dict["lr"] = 0.01
        if any(splt_t in args.dataset for splt_t in bdy_drp1_lr1e2_list):
            print("OVERRIDE - DROP - 0.1, LR = 1e-2")
            default_dict["label_dropout"] = 0.1
            default_dict["feature_dropout"] = 0.1
            default_dict["lr"] = 0.01

        over_write_list = ["num_layers", "input_dim",  "hidden_channels", "label_dropout", "feature_dropout",\
                            "batch_size", "lr", "epochs", "use_feature", "use_learnable_embedding",]
    else:
        default_dict["num_layers"] = 3
        default_dict["input_dim"] = 128
        default_dict["hidden_channels"] = 128
        default_dict["epochs"] = 1000
        default_dict["dropout"] = 0.0
        over_write_list = ["num_layers", "input_dim",  "hidden_channels", "dropout",\
                            "batch_size", "lr", "epochs", "use_feature", "use_learnable_embedding",]
    
    for attr in over_write_list:
        if getattr(args, attr) is None:
            setattr(args, attr, default_dict[attr])
         
    if args.model in ['adamic', 'simple', 'adamic_ogb', "resource_allocation" , 'katz']:
        args.use_feature = False
        args.use_learnable_embedding = False
    if args.model == 'simplecos':
        args.use_learnable_embedding = False
        
    return args
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='models')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--hidden_channels', type=int)
    parser.add_argument('--input_dim', type=int)
    parser.add_argument('--out_channels', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--feature_dropout', type=float)
    parser.add_argument('--label_dropout', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--use_feature', type=bool)
    parser.add_argument('--use_learnable_embedding', type=bool)
    parser.add_argument('--edge_drop', type=float, default=0.0)
    args = parser.parse_args()
    
    args = default_model_configs(args)
    print(args)
    
    
    