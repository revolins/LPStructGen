

import sys
import math
from tqdm import tqdm
import random
import numpy as np
import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path
import torch
from torch_sparse import spspmm
import torch_geometric
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges, degree)
from scipy.sparse.linalg import expm
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics.pairwise import cosine_similarity

import pdb
import networkx as nx
import math

def PA(A, edge_index, batch_size=100000):
    # D. Liben-Nowell, J. Kleinberg. The Link Prediction Problem for Social Networks (2004). http://www.cs.cornell.edu/home/kleinber/link-pred.pdf
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    
    G = nx.from_scipy_sparse_array(A)
    G_degree = np.array(G.degree(np.array(G.nodes())))
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = G_degree[src][:, 1] * G_degree[dst][:, 1]
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores)

def CN(A, edge_index, batch_size=100000):
    # The Common Neighbor heuristic score.
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(cur_scores)
        # print('max cn: ', np.concatenate(scores, 0).max())

    return torch.FloatTensor(np.concatenate(scores, 0))


def AA(A, edge_index, batch_size=100000):
    # The Adamic-Adar heuristic score.
    multiplier = 1 / np.log(A.sum(axis=0))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    
    return torch.FloatTensor(scores)

def RA(A, edge_index, batch_size=100000):
    # The Adamic-Adar heuristic score.
    multiplier = 1 / (A.sum(axis=0))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores)

def PPR(A, edge_index):
    # The Personalized PageRank heuristic score.
    # Need install fast_pagerank by "pip install fast-pagerank"
    # Too slow for large datasets now.
    from fast_pagerank import pagerank_power
    num_nodes = A.shape[0]
    src_index, sort_indices = torch.sort(edge_index[0])
    dst_index = edge_index[1, sort_indices]
    edge_index = torch.stack([src_index, dst_index])
    #edge_index = edge_index[:, :50]
    scores = []
    visited = set([])
    j = 0
    for i in tqdm(range(edge_index.shape[1])):
        if i < j:
            continue
        src = edge_index[0, i]
        personalize = np.zeros(num_nodes)
        personalize[src] = 1
        ppr = pagerank_power(A, p=0.85, personalize=personalize, tol=1e-7)
        j = i
        while edge_index[0, j] == src:
            j += 1
            if j == edge_index.shape[1]:
                break
        all_dst = edge_index[1, i:j]
        cur_scores = ppr[all_dst]
        if cur_scores.ndim == 0:
            cur_scores = np.expand_dims(cur_scores, 0)
        scores.append(np.array(cur_scores))

    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores)


def SP(A, edge_index, remove=True):
    
    scores = []
    G = nx.from_scipy_sparse_array(A)
    print(len(G.edges()))
    add_flag1 = 0
    add_flag2 = 0
    count = 0
    count1 = count2 = 0
    print('remove: ', remove)
    for i in range(edge_index.size(1)):
        s = edge_index[0][i].item()
        t = edge_index[1][i].item()
        if s == t:
            count += 1
            scores.append(999)
            continue

        if remove:
            if (s,t) in G.edges: 
                G.remove_edge(s,t)
                add_flag1 = 1
                count1 += 1
            if (t,s) in G.edges: 
                G.remove_edge(t,s)
                add_flag2 = 1
                count2 += 1

        if nx.has_path(G, source=s, target=t):
            sp = nx.shortest_path_length(G, source=s, target=t)
        else:
            sp = 999
        
        if add_flag1 == 1: 
            G.add_edge(s,t)
            add_flag1 = 0

        if add_flag2 == 1: 
            G.add_edge(t, s)
            add_flag2 = 0
    
        scores.append(1/(sp))
    print('equal number: ', count)
    print('count1: ', count1)
    print('count2: ', count2)

    return torch.FloatTensor(scores)

def compute_normalization(S1, S2, g_func):
    return g_func(len(S1), len(S2))


def compute_TC(G, node, K, t_type, beta):
    beta = 0.5
    K = 3 
    node = 1  
    t_type = 'train'
    #Construct it to calculate for just a set of nodes in a given edge
    neighbors_t = [n for n in G.neighbors(node) if G.nodes[n]['type'] == t_type]
    total_score = 0
    for nj in neighbors_t:
        for k1 in range(1, K+1):
            for k2 in range(1, K+1):
                Hk1_i = set(nx.single_source_shortest_path_length(G, node, cutoff=k1).keys()) #Replace with BUDDY Hashes up to K=3
                Hk2_j = set(nx.single_source_shortest_path_length(G, nj, cutoff=k2).keys())
                intersection_size = len(Hk1_i & Hk2_j)
                normalization = compute_normalization(Hk1_i, Hk2_j, lambda a, b: a * b) 
                total_score += (beta ** (k1 + k2 - 2)) * (intersection_size / normalization)
    return total_score / len(neighbors_t) if neighbors_t else 0

def TC(A, edge_index, batch_size=100000):
    multiplier = 1 / (A.sum(axis=0))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores)

def katz_apro(A, edge_index, beta=0.005, path_len=3, remove=False):
    scores = []
    G = nx.from_scipy_sparse_matrix(A)
    path_len = int(path_len)
    count = 0
    add_flag1 = 0
    add_flag2 = 0
    count1 = count2 = 0
    betas = np.zeros(path_len)
    print('remove: ', remove)
    for i in range(len(betas)):
        betas[i] = np.power(beta, i+1)
    
    for i in range(edge_index.size(1)):
        s = edge_index[0][i].item()
        t = edge_index[1][i].item()

        if s == t:
            count += 1
            scores.append(0)
            continue
        
        if remove:
            if (s,t) in G.edges: 
                G.remove_edge(s,t)
                add_flag1 = 1
                count1 += 1
                
            if (t,s) in G.edges: 
                G.remove_edge(t,s)
                add_flag2 = 1
                count2 += 1


        paths = np.zeros(path_len)
        for path in nx.all_simple_paths(G, source=s, target=t, cutoff=path_len):
            paths[len(path)-2] += 1  
        
        kz = np.sum(betas * paths)

        scores.append(kz)
        
        if add_flag1 == 1: 
            G.add_edge(s,t)
            add_flag1 = 0

        if add_flag2 == 1: 
            G.add_edge(t, s)
            add_flag2 = 0
        
    print('equal number: ', count)
    print('count1: ', count1)
    print('count2: ', count2)

    return torch.FloatTensor(scores)


def katz_close(A, edge_index, beta=0.005):

    scores = []
    G = nx.from_scipy_sparse_matrix(A)

    adj = nx.adjacency_matrix(G, nodelist=range(len(G.nodes)))
    aux = adj.T.multiply(-beta).todense()
    np.fill_diagonal(aux, 1+aux.diagonal())
    sim = np.linalg.inv(aux)
    np.fill_diagonal(sim, sim.diagonal()-1)

    for i in range(edge_index.size(1)):
        s = edge_index[0][i].item()
        t = edge_index[1][i].item()

        scores.append(sim[s,t])

    
    return torch.FloatTensor(scores)

