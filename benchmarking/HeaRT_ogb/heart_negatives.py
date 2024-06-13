"""
Create heart negative samples for validation and testing evaluation
"""
import os 
import torch
import numpy as np
from tqdm import tqdm
import networkx as nx
from scipy.stats import rankdata
from argparse import ArgumentParser 
from torch.utils.data import DataLoader
from torch_geometric.utils import degree, to_undirected
from synth_dataset import SynthDataset
from torch_sparse import SparseTensor

import scipy.sparse as ssp
import random

# ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")


def prep_data(data, edge_split):
    """
    Various prep
    """
    data.adj_t = data.adj_t.coalesce().bool().float()
    data.adj_t = data.adj_t.to_symmetric()

    train_edge_index = to_undirected(edge_split['train']['edge'].t())

    val_edge_index = to_undirected(edge_split['valid']['edge'].t())
    full_edge_index = torch.cat([train_edge_index, val_edge_index], dim=-1)

    val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=torch.float)
    train_edge_weight = torch.ones([train_edge_index.size(1), 1], dtype=torch.float)
    full_edge_weight = torch.cat([train_edge_weight, val_edge_weight], 0).view(-1)

    data.full_edge_index = full_edge_index
    data.full_edge_weight = full_edge_weight
    data.full_adj = SparseTensor.from_edge_index(full_edge_index, full_edge_weight, [data.num_nodes, data.num_nodes])
    data.full_adj = data.full_adj.to_symmetric()

    return data



def save_samples(samples, file_name):
    print("Saving samples...", samples.shape)
    with open(file_name, "wb") as f:
        np.save(f, samples)


def calc_CN(data, use_val=False):
    """
    Calc CNs for all node pairs
    """
    if use_val:
        adj = data.full_adj_t
    else:
        adj = data.adj_t

    cn_scores = adj @ adj

    return cn_scores


def calc_PA(data, batch_size=100000):
    # D. Liben-Nowell, J. Kleinberg. The Link Prediction Problem for Social Networks (2004). http://www.cs.cornell.edu/home/kleinber/link-pred.pdf
    
    G_degree = degree(data.edge_index[0], data.num_nodes)

    return G_degree


def calc_SP(src, G):
    """
    Calc SP distance
    """
    sp = nx.single_source_shortest_path_length(G, src)

    sp = [sp.get(i, 999) for i in range(G.number_of_nodes())]
    sp[src] = 999
    
    return 1 / torch.Tensor(sp)


def rank_score_matrix(row):
    """
    Rank from largest->smallest
    """
    num_greater_zero = (row > 0).sum().item()

    # Ignore 0s and -1s in ranking
    # Note: default is smallest-> largest so reverse
    if num_greater_zero > 0:
        ranks_row = rankdata(row[row > 0], method='min')
        ranks_row = ranks_row.max() - ranks_row + 1
        max_rank = ranks_row.max()
    else:
        ranks_row = []
        max_rank = 0

    # Overwrite row with ranks
    # Also overwrite 0s with max+1 and -1s with max+2
    row[row > 0] = ranks_row
    row[row == 0] = max_rank + 1
    row[row < 0] = max_rank + 2

    return row


def rank_and_merge_node(node_scores, true_pos_mask, data, args):
    """
    Do so for a single node
    """
    k = args.num_samples // 2 

    # Nodes that are 0 for all scores. Needed later when selecting top K
    zero_nodes_score_mask = (node_scores == 0).numpy()

    # Individual ranks
    node_ranks = rank_score_matrix(node_scores.numpy())

    # If enough non-zero scores we use just take top-k
    # Otherwise we have to randomly select from 0 scores        
    max_greater_zero = data['num_nodes'] - zero_nodes_score_mask.sum().item() - true_pos_mask.sum().item()

    # NOTE: Negate when using torch.topk since 1=highest
    if max_greater_zero >= k:
        node_topk = torch.topk(torch.from_numpy(-node_ranks), k).indices
        node_topk = node_topk.numpy()
    elif max_greater_zero <= 0:
        # All scores are either true_pos or 0
        # We just sample from 0s here
        node_zero_score_ids = zero_nodes_score_mask.nonzero()[0]
        node_topk = np.random.choice(node_zero_score_ids, k)
    else:
        # First just take whatever non-zeros there are
        node_greater_zero = torch.topk(torch.from_numpy(-node_ranks), max_greater_zero).indices
        node_greater_zero = node_greater_zero.numpy()

        # Then choose the rest randomly from 0 scores
        node_zero_score_ids = zero_nodes_score_mask.nonzero()[0]
        node_zero_rand = np.random.choice(node_zero_score_ids, k-max_greater_zero)
        node_topk = np.concatenate((node_greater_zero, node_zero_rand))

    return node_topk.reshape(-1, 1)


def rank_and_merge_edges(edges, cn_scores, pa_scores, data, train_nodes, args, test=False):
    """
    For each edge we get the rank for the types of scores for each node and merge them together to one rank

    Using that we get the nodes with the top k ranks
    """
    all_topk_edges = []
    k = args.num_samples // 2 

    # Used to determine positive samples to filter
    # For testing we also include val samples in addition to train
    if test:
        adj = data.full_adj
    else:
        adj = data.adj_t

    if args.metric.upper() == "SP":
        edge_index, edge_weight = data.full_edge_index, data.full_edge_weight
        A_ssp = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), shape=(data.num_nodes, data.num_nodes))
        G = nx.from_scipy_sparse_array(A_ssp)    

    ### Get nodes not in train
    all_nodes = set(list(range(data.num_nodes)))
    nodes_not_in_train = torch.Tensor(list(all_nodes - train_nodes)).long()
    
    for edge in tqdm(edges, "Ranking Scores"):
        source, target = edge[0].item(), edge[1].item()

        source_adj = adj[source].to_dense().squeeze(0).bool()
        target_adj = adj[target].to_dense().squeeze(0).bool()

        if args.metric.upper() == "CN":
            source_scores = cn_scores[source].to_dense().squeeze(0)
            target_scores = cn_scores[target].to_dense().squeeze(0)
        elif args.metric.upper() == "PA":
            source_scores = target_scores = pa_scores
        else:
            # raise NotImplementedError("SP metric!!!")
            source_scores = calc_SP(source, G)
            target_scores = calc_SP(target, G)

        source_true_pos_mask = source_adj
        target_true_pos_mask = target_adj

        # Don't remove true positive
        # So just set all to 0
        # if args.keep_train_val:
        #     source_true_pos_mask = torch.zeros_like(source_true_pos_mask)
        #     target_true_pos_mask = torch.zeros_like(target_true_pos_mask)

        # Mask nodes not in train
        source_true_pos_mask[nodes_not_in_train] = 1
        target_true_pos_mask[nodes_not_in_train] = 1

        # Include masking for self-loops
        source_true_pos_mask[source], source_true_pos_mask[target] = 1, 1
        target_true_pos_mask[target], target_true_pos_mask[source] = 1, 1

        # Filter samples by setting to -1
        source_scores[source_true_pos_mask], source_scores[source_true_pos_mask] = -1, -1 

        source_topk_nodes = rank_and_merge_node(source_scores, source_true_pos_mask, data, args)
        source_topk_edges = np.concatenate((np.repeat(source, k).reshape(-1, 1), source_topk_nodes), axis=-1)

        target_topk_nodes = rank_and_merge_node(target_scores, target_true_pos_mask, data, args)
        target_topk_edges = np.concatenate((target_topk_nodes, np.repeat(target, k).reshape(-1, 1)), axis=-1)
        
        edge_samples = np.concatenate((source_topk_edges, target_topk_edges))
        all_topk_edges.append(edge_samples)

    return np.stack(all_topk_edges)


def calc_all_heuristics(args, data, split_edge, dataset_name):
    """
    Calc and store top-k negative samples for each sample
    """
    # dataset = SynthDataset(f"{args.dataset}_seed1", root_dir=args.root_dir)
    # print("Getting Data...")
    # data = dataset.get()
    # print("Getting Edge Split...")
    # edge_split = dataset.get_edge_split()
    print("Prepping data...")
    data = prep_data(data, split_edge)

    # Get unique nodes in train
    train_nodes = set(split_edge['train']['edge'].flatten().tolist())

    print("Compute CNs...")
    cn_scores = calc_CN(data)
    print("Compute PA...")
    pa_scores = calc_PA(data)

    print("\n>>> Valid")
    val_neg_samples = rank_and_merge_edges(split_edge['valid']['edge'], cn_scores, pa_scores, data, train_nodes, args)
    with open(f"dataset/{dataset_name}Dataset/heart_valid_samples.npy", "wb") as f:
        np.save(f, val_neg_samples)

    print("\n>>> Test")
    test_neg_samples = rank_and_merge_edges(split_edge['test']['edge'], cn_scores, pa_scores, data, train_nodes, args, test=True)
    with open(f"dataset/{dataset_name}Dataset/heart_test_samples.npy", "wb") as f:
        np.save(f, test_neg_samples)

def set_seeds():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    os.environ["PYTHONHASHSEED"] = "42"

# def main():
#     parser = ArgumentParser(description="Create HeaRT negative samples")
#     parser.add_argument("--dataset", help="Dataset to create samples for", type=str, required=True)
#     parser.add_argument("--root-dir", help="Root directory of datasets", type=str, default="./dataset")
#     parser.add_argument("--use-val-in-test", help="Use validation in test. Only applies to Collab", action='store_true', default=False)
#     parser.add_argument("--keep-train-val", help="Keep train+valid samples instead of filtering", action='store_true', default=False)

#     parser.add_argument("--metric", help="Either 'CN', 'SP', 'PA'", type=str, default="CN")
#     parser.add_argument("--num-samples", help="Number of negative samples per sample", type=int, default=250)

#     # For PPR
#     parser.add_argument("--eps", help="Stopping criterion threshold", type=float, default=5e-5)
#     parser.add_argument("--alpha", help="Teleportation probability", type=float, default=0.15)

#     args = parser.parse_args()

#     set_seeds()

#     if args.dataset.startswith("ogbl"):
#         calc_all_heuristics(args, 1)
#     else:
#         for seed in range(1, 6):
#             print(f">>> Seed={seed}")
#             calc_all_heuristics(args, seed)



# if __name__ == "__main__":
#     main()
