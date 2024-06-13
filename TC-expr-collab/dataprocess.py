import torch
import numpy as np

from collections import defaultdict, Counter
import os
import pickle as pkl

import torch_geometric.transforms as T
from torch_geometric.utils import degree, train_test_split_edges, add_self_loops, to_undirected
from torch_geometric.datasets import Planetoid
from torch_scatter import scatter_add, segment_csr

from synth_dataset import SynthDataset
from torch_sparse import SparseTensor


from scipy.sparse import coo_array
import numpy as np

from utils import find_root

ROOT_DIR = find_root(__file__)

def randomsplit(dataset, val_ratio: float=0.10, test_ratio: float=0.2):
    def removerepeated(ei):
        ei = to_undirected(ei)
        ei = ei[:, ei[0]<ei[1]]
        return ei

    data = dataset[0]
    data.num_nodes = data.x.shape[0]
    data = train_test_split_edges(data, test_ratio, test_ratio)
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    num_val = int(data.val_pos_edge_index.shape[1] * val_ratio/test_ratio)
    data.val_pos_edge_index = data.val_pos_edge_index[:, torch.randperm(data.val_pos_edge_index.shape[1])]
    split_edge['train']['edge'] = removerepeated(torch.cat((data.train_pos_edge_index, data.val_pos_edge_index[:, :-num_val]), dim=-1)).t()
    split_edge['valid']['edge'] = removerepeated(data.val_pos_edge_index[:, -num_val:]).t()
    split_edge['valid']['edge_neg'] = removerepeated(data.val_neg_edge_index).t()
    split_edge['train']['edge_neg'] = split_edge['valid']['edge_neg']
    split_edge['test']['edge'] = removerepeated(data.test_pos_edge_index).t()
    split_edge['test']['edge_neg'] = removerepeated(data.test_neg_edge_index).t()

    return split_edge

def load_data(args):
    data = SynthDataset(dataset_name=args.dataset).get()
    split_edge = SynthDataset(dataset_name=args.dataset).get_edge_split()
    data.x = data.x.float()
    row, col, value = data.adj_t.coo()
    index = torch.stack([row, col], dim=0)
    data.train_edge_index, data.train_edge_weight_gcn, _ = normalize_edge(index.t(), data.x.shape[0]) 

    if args.model in ['GCN', 'GCN-aug']:
        data.train_adj_gcn = SparseTensor(row=data.train_edge_index[0], col=data.train_edge_index[1], value=data.train_edge_weight_gcn, is_sorted=False)
    
    data.train_edge_weight = data.train_edge_weight_gcn
    data.train_adj = data.adj_t
    
    print(f"data_train_edge_weight in {args.model}: ", data.train_edge_weight)

    idx = torch.sort(data.train_edge_index[1])[1]
    data.train_edge_index = data.train_edge_index[:, idx]
    data.train_edge_weight = data.train_edge_weight[idx]
    A = coo_array((data.train_edge_weight.numpy(), data.train_edge_index.numpy())).tocsr()
    
    data.train_ptr = torch.from_numpy(A.indptr).long()
    
    adj_set_dict = cal_adj_set_dict(split_edge, args.dataset)
    data.deg = {key: torch.tensor([len(adj_set_dict[key][i]) for i in range(data.x.shape[0])]) for key in adj_set_dict}
    data = cal_triangle(data, adj_set_dict)

    while split_edge['train']['edge'].size(0) <= args.train_bsz:
        print("Positive Training Edges are smaller than batch, reducing batch size")
        args.train_bsz = args.train_bsz // 2
        if args.train_bsz <= 0:
            raise Exception("Batch Size Reached 0 in Pos. Train Edges")
    
    while split_edge['valid']['edge'].size(0) <= args.train_bsz:
        print(f"Positive Validation Edges, {split_edge['valid']['edge'].size()} are smaller than batch, reducing batch size from {args.train_bsz}")
        args.train_bsz = args.train_bsz // 2
        print("New Batch Size: ", args.train_bsz)
        if args.train_bsz <= 0:
            raise Exception("Batch Size Reached 0 in Pos. Val. Edges")
        
    while split_edge['test']['edge'].size(0) <= args.eval_bsz:
        print("Positive Testing Edges are smaller than test batch, reducing test batch size")
        args.eval_bsz = args.eval_bsz // 2
        if args.eval_bsz <= 0:
            raise Exception("Batch Size reached 0 in Pos. Testing Edges")

    with open(f'{ROOT_DIR}/benchmarking/HeaRT_ogb/dataset/{args.dataset}Dataset/heart_valid_samples.npy', "rb") as f:
        neg_valid_edge = np.load(f)
        neg_valid_edge = torch.from_numpy(neg_valid_edge)
    with open(f'{ROOT_DIR}/benchmarking/HeaRT_ogb/dataset/{args.dataset}Dataset/heart_test_samples.npy', "rb") as f:
        neg_test_edge = np.load(f)
        neg_test_edge = torch.from_numpy(neg_test_edge)
    
    while neg_valid_edge.size(0) <= args.eval_bsz: 
        print("Negative Validation Edges are smaller than batch, reducing batch size")
        args.eval_bsz = args.eval_bsz // 2
        if args.eval_bsz == 0:
            raise Exception("Batch Size Reached 0 in Neg. Val. Edges")
    
    while neg_test_edge.size(0) <= args.eval_bsz:
        print("Negative Testing Edges are smaller than testing batch, reducing testing batch size")
        args.eval_bsz = args.eval_bsz // 2
        if args.eval_bsz == 0:
            raise Exception("Batch Size Reached 0 in Neg. Test Edges")

    split_edge['valid']['edge_neg'] = neg_valid_edge
    split_edge['test']['edge_neg'] = neg_test_edge

    if split_edge['valid']['edge'].size(0) < split_edge['train']['edge'].size(0):
        idx = torch.randperm(split_edge['train']['edge'].size(0))[:split_edge['valid']['edge'].size(0)] # For predictions, train shouldn't exceed valid
        split_edge['train']['edge'] = split_edge['train']['edge'][idx]
        print("*************** Permute by Valid *****************")
        idx = torch.randperm(split_edge['valid']['edge_neg'].size(0)) # Randomly permute validation edges to make negative training edges
        split_edge['train']['edge_neg'] = split_edge['valid']['edge_neg'][idx]
    else:
        print("*************** Slice by Train *****************")
        idx = torch.randperm(split_edge['valid']['edge_neg'].size(0))[:split_edge['train']['edge'].size(0)]
        assert idx.size(0) == split_edge['train']['edge'].size(0), f"Randomly-permuted negative train edge index: {idx.size()}, is not equal to positive training: {split_edge['train']['edge'].size()}, incorrect access time likely"
        split_edge['train']['edge_neg'] = split_edge['valid']['edge_neg'][idx]

    data.n_nodes, data.n_edges = data.x.shape[0], data.train_edge_index.shape[1]//2
    data.eval_node = {key: torch.unique(split_edge[key]['edge']) for key in split_edge}

    data.one_hot_dict = {}
    for key, unique_elements in data.eval_node.items():
        data.one_hot_dict[key] = torch.zeros(data.n_nodes, dtype = bool)
        data.one_hot_dict[key][unique_elements] = 1

    train_tc, valid_tc, test_tc = update_tc(data, data.train_edge_weight)

    torch.save(train_tc.cpu(), os.path.join(args.path, 'model', args.dataset, args.model, 'train_tc_ori.pt'))
    torch.save(valid_tc.cpu(), os.path.join(args.path, 'model', args.dataset, args.model, 'valid_tc_ori.pt'))
    torch.save(test_tc.cpu(), os.path.join(args.path, 'model', args.dataset, args.model, 'test_tc_ori.pt'))

    data.train_tc = train_tc[(data.deg['train'] != 0) & (data.one_hot_dict['test'])].mean().item()
    data.valid_tc = valid_tc[(data.deg['valid'] != 0) & (data.one_hot_dict['test'])].mean().item()
    data.test_tc = test_tc[(data.deg['test'] != 0) & (data.one_hot_dict['test'])].mean().item()

    return data, split_edge, adj_set_dict, {'train': train_tc.cpu(), 'valid': valid_tc.cpu(), 'test': test_tc.cpu()}

def normalize_edge(edge_index, n_node):
    edge_index = to_undirected(edge_index.t(), num_nodes=n_node)

    edge_index, _ = add_self_loops(edge_index, num_nodes=n_node)

    row, col = edge_index
    deg = degree(col)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.

    edge_weight_gcn = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    edge_weight_sage = deg_inv_sqrt[row] * deg_inv_sqrt[row]

    return edge_index, edge_weight_gcn, edge_weight_sage

class PermIterator:
    '''
    Iterator of a permutation
    '''
    def __init__(self, device, size, bs, training=True) -> None:
        self.bs = bs
        self.training = training
        self.idx = torch.randperm(
            size, device=device) if training else torch.arange(size,
                                                               device=device)

    def __len__(self):
        return (self.idx.shape[0] + (self.bs - 1) *
                (not self.training)) // self.bs

    def __iter__(self):
        self.ptr = 0
        return self

    def __next__(self):
        if self.ptr + self.bs * self.training > self.idx.shape[0]:
            raise StopIteration
        ret = self.idx[self.ptr:self.ptr + self.bs]
        self.ptr += self.bs
        return ret

def cal_adj_set_dict(edge, dataset):
    adj_set_dict = {'train': defaultdict(set),
                     'valid': defaultdict(set),
                     'test': defaultdict(set),
                     'train_valid': defaultdict(set),
                     'train_valid_test': defaultdict(set),
                    }

    for node1, node2 in edge['train']['edge']:
        node1, node2 = node1.item(), node2.item()

        adj_set_dict['train'][node1].add(node2)
        adj_set_dict['train'][node2].add(node1)

        adj_set_dict['train_valid'][node1].add(node2)
        adj_set_dict['train_valid'][node2].add(node1)

        adj_set_dict['train_valid_test'][node1].add(node2)
        adj_set_dict['train_valid_test'][node2].add(node1)

    for node1, node2 in edge['valid']['edge']:
        node1, node2 = node1.item(), node2.item()

        adj_set_dict['train_valid'][node1].add(node2)
        adj_set_dict['train_valid'][node2].add(node1)

        adj_set_dict['train_valid_test'][node1].add(node2)
        adj_set_dict['train_valid_test'][node2].add(node1)

        adj_set_dict['valid'][node1].add(node2)
        adj_set_dict['valid'][node2].add(node1)

    for node1, node2 in edge['test']['edge']:
        node1, node2 = node1.item(), node2.item()

        adj_set_dict['train_valid_test'][node1].add(node2)
        adj_set_dict['train_valid_test'][node2].add(node1)

        adj_set_dict['test'][node1].add(node2)
        adj_set_dict['test'][node2].add(node1)

    pkl.dump(adj_set_dict, open(os.getcwd() + '/data/' + dataset + '/adj_set_dict.pkl', 'wb'))

    return adj_set_dict

def cal_tc(adj_list_dict, n_nodes, deg, K):
    train_train_tc, train_val_tc, train_test_tc = [], [], []

    def recursion_nei(adj_hop, train_adj_list, k, K, prev_nei):
        if k >= K:
            return

        cur_nei = []
        for node in prev_nei:
            cur_nei.extend(train_adj_list[node])

        adj_hop[k + 1] = cur_nei

        next_nei = recursion_nei(adj_hop, train_adj_list, k + 1, K, cur_nei)

        return

    beta = 1 / np.mean(deg)

    for node1 in range(n_nodes):
        train_nei, val_nei, test_nei = adj_list_dict['train'][
            node1], adj_list_dict['valid'][node1], adj_list_dict['test'][node1]

        deg_train, deg_val, deg_test = len(
            train_nei), len(val_nei), len(test_nei)

        adj_hop1 = {}
        recursion_nei(adj_hop1, adj_list_dict['train'], 0, K, [node1])

        nei_1_hop_weight = [Counter(adj_hop1[key]) for key in adj_hop1]
        nei_1_hop = [set(adj_hop1[key]) for key in adj_hop1]

        if deg_val > 0 and deg_train > 0:
            counts = []

            for node2 in val_nei:
                adj_hop2 = {}
                recursion_nei(adj_hop2, adj_list_dict['train'], 0, K, [node2])

                nei_2_hop_weight = [Counter(adj_hop2[key]) for key in adj_hop2]
                nei_2_hop = [set(adj_hop2[key]) for key in adj_hop2]


                inters, norms = [], []

                for i in range(len(nei_1_hop)):
                    for j in range(len(nei_2_hop)):
                        tmp_nei_1, tmp_nei_2 = nei_1_hop[i], nei_2_hop[j]
                        tmp_nei_1_w, tmp_nei_2_w = nei_1_hop_weight[i], nei_2_hop_weight[j]

                        inters.append(sum([tmp_nei_1_w.get(tmp, 0) for tmp in tmp_nei_1.intersection(tmp_nei_2)]) * beta ** (i + j))
                        norms.append(sum(list(tmp_nei_1_w.values())) * beta ** (i + j))

                counts.append(sum(inters) / sum(norms))

            train_val_tc.append(np.mean(counts))

        else:
            train_val_tc.append(0)

        if deg_test > 0 and deg_train > 0:
            counts = []

            for node2 in test_nei:
                adj_hop2 = {}
                recursion_nei(adj_hop2, adj_list_dict['train'], 0, K, [node2])

                nei_2_hop_weight = [Counter(adj_hop2[key]) for key in adj_hop2]
                nei_2_hop = [set(adj_hop2[key]) for key in adj_hop2]


                inters, norms = [], []

                for i in range(len(nei_1_hop)):
                    for j in range(len(nei_2_hop)):
                        tmp_nei_1, tmp_nei_2 = nei_1_hop[i], nei_2_hop[j]
                        tmp_nei_1_w, tmp_nei_2_w = nei_1_hop_weight[i], nei_2_hop_weight[j]

                        inters.append(sum([tmp_nei_1_w.get(tmp, 0) for tmp in tmp_nei_1.intersection(tmp_nei_2)]) * beta ** (i + j))
                        norms.append(sum(list(tmp_nei_1_w.values())) * beta ** (i + j))

                counts.append(sum(inters) / sum(norms))

            train_test_tc.append(np.mean(counts))

        else:
            train_test_tc.append(0)

        if deg_train > 1:
            counts = []

            for node2 in train_nei:
                adj_hop2 = {}
                recursion_nei(adj_hop2, adj_list_dict['train'], 0, K, [node2])

                nei_2_hop_weight = [Counter(adj_hop2[key]) for key in adj_hop2]
                nei_2_hop = [set(adj_hop2[key]) for key in adj_hop2]


                inters, norms = [], []

                for i in range(len(nei_1_hop)):
                    for j in range(len(nei_2_hop)):
                        tmp_nei_1, tmp_nei_2 = nei_1_hop[i], nei_2_hop[j]
                        tmp_nei_1_w, tmp_nei_2_w = nei_1_hop_weight[i], nei_2_hop_weight[j]

                        inters.append(sum([tmp_nei_1_w.get(tmp, 0) for tmp in tmp_nei_1.intersection(tmp_nei_2)]) * beta ** (i + j))

                        if i == 0 and j == 0:
                            norms.append((sum(list(tmp_nei_1_w.values())) - 1) * beta ** (i + j))
                        else:
                            norms.append((sum(list(tmp_nei_1_w.values()))) * beta ** (i + j))

                counts.append(sum(inters) / sum(norms))

            train_train_tc.append(np.mean(counts))
        else:
            train_train_tc.append(0)

    lcc = {'train_train': train_train_tc,
           'train_val': train_val_tc, 'train_test': train_test_tc}

    return lcc

def cal_triangle(data, adj_set_dict):
    train_tri, val_tri, test_tri = [], [], []

    for node1, node2 in data.train_edge_index.t():
        node1, node2 = node1.item(), node2.item()

        train_nei, val_nei, test_nei = adj_set_dict['train'][node2], adj_set_dict['valid'][node2], adj_set_dict['test'][node2]

        train_tri.append(len(train_nei.intersection(adj_set_dict['train'][node1])))
        val_tri.append(len(val_nei.intersection(adj_set_dict['train'][node1])))
        test_tri.append(len(test_nei.intersection(adj_set_dict['train'][node1])))
    
    data.train_tri = torch.tensor(train_tri, dtype=torch.float)
    data.val_tri = torch.tensor(val_tri, dtype=torch.float)
    data.test_tri = torch.tensor(test_tri, dtype=torch.float)

    return data

def update_tc(data, weight):
    new_weight = torch.clone(weight)
    new_weight[data.train_edge_index[0] == data.train_edge_index[1]] = 0
    norm_new = segment_csr(new_weight, data.train_ptr, reduce='sum')[data.train_edge_index[1]]
    new_weight = new_weight/norm_new
    
    train_tc = segment_csr(new_weight*data.train_tri/data.deg['train'][data.train_edge_index[1]], data.train_ptr, reduce='sum')
    val_tc = segment_csr(new_weight*data.val_tri/data.deg['valid'][data.train_edge_index[1]], data.train_ptr, reduce='sum')
    test_tc = segment_csr(new_weight*data.test_tri/data.deg['test'][data.train_edge_index[1]], data.train_ptr, reduce='sum')

    return train_tc, val_tc, test_tc

