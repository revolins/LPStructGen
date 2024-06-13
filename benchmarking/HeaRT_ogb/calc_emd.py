import argparse
#import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ssp
import scipy.stats as sst
from get_heuristic import *
from collections import defaultdict
import torch
import pickle
from synth_dataset import SynthDataset
from tqdm import tqdm
from torch_sparse import SparseTensor
import torch_sparse

# python calc_emd.py --use_heuristic CN --dataset ogbl-collab # For CN splits on ogbl-collab
# python calc_emd.py --use_heuristic CN --dataset ogbl-collab --eps # For EPS on collab
# # python calc_emd.py --use_heuristic CN --dataset ogbl-collab --edge_drop # For edge_drop on collab

def add_edges(dataset, edge_index, edge_weight, extra_edges, num_nodes):
    full_edge_index = torch.cat([edge_index.clone(), extra_edges], dim=-1)
    new_edge_weight = torch.ones(extra_edges.shape[1])
    full_edge_weights = torch.cat([edge_weight, new_edge_weight], 0) 
    adj_t = SparseTensor.from_edge_index(full_edge_index, full_edge_weights, sparse_sizes = [num_nodes,num_nodes])
    adj_t = adj_t.to_symmetric() 
    if dataset != "collab":
        adj_t = adj_t.fill_value(1.)
    return adj_t

def drop_edges(csr_mat, drop_probability):
    for i in range(csr_mat.shape[0]):
        row_start = csr_mat.indptr[i]
        row_end = csr_mat.indptr[i + 1]
        for j in range(row_start, row_end):
            if np.random.rand() < drop_probability:
                csr_mat.data[j] = 0

    csr_mat.eliminate_zeros()
    return csr_mat

def main():
    parser = argparse.ArgumentParser(description='subprocess script to save training and testing sample distributions for EMD analysis')
    parser.add_argument('--device', type=int, default='0')
    parser.add_argument('--use_heuristic', type=str, default='CN', help='CN, SP, PA')
    parser.add_argument('--dataset', type=str, help='ogbl-collab, ogbl-ppa', required=True)
    parser.add_argument('--eps', action='store_true', help='True = calculating EMD for EPS, False = default', default=False)
    parser.add_argument('--edge_drop', type=float, help='0.1, 0.3, 0.5', default=0.0)
    parser.add_argument('--eps_model', type=str, help='buddy, gcn', default='gcn')
    args = parser.parse_args() 

    dataset = [str(dataset) for dataset in args.dataset.split(',')]
    device = torch.device(args.device)

    # 1-to-1 for dataset split type
    if args.use_heuristic == 'CN':
        split_list = ["CN_0_1_2", "CN_0_2_4", "CN_0_3_5", "CN_2_1_0", "CN_4_2_0", "CN_5_3_0"]
    elif args.use_heuristic == 'SP':
        split_list = ["SP_00_017_026",  "SP_00_026_036","SP_036_026_00", "SP_026_017_00"]
    elif args.use_heuristic == 'PA':
        if 'collab' in args.dataset.lower():
            split_list = ["PA_0_50_100", "PA_0_100_200", "PA_0_150_250", "PA_250_150_0","PA_200_100_0", "PA_100_50_0"]
        elif 'ppa' in args.dataset.lower():
            split_list = ["PA_0_5000_10000", "PA_0_10000_20000", "PA_0_15000_25000", "PA_10000_5000_0", "PA_20000_10000_0", "PA_25000_15000_0"]
        else: "Bad Dataset Choice for PA, must be ogbl-collab or ogbl-ppa"
    else: raise "Bad Heuristic Choice, must choose - CN, SP, PA"
    

    for dn in dataset:
        dataset_dict = defaultdict(float)
        for spl_t in tqdm(split_list, desc="EMD Split Calculations"): 
            dataset_name = dn + '_' + spl_t + '_seed1' # Ex. "ogbl-collab_CN_0_1_2_seed1"
            data = SynthDataset(dataset_name=dataset_name).get()
            split_edge = SynthDataset(dataset_name=dataset_name).get_edge_split()

            node_num = data.num_nodes
            edge_index = data.edge_index
            edge_weight = torch.ones(data.edge_index.size(1), dtype=int)

            if args.eps and args.edge_drop > 0.0: raise "Cannot quantify edge_drop AND EPS, remove --eps or --edge_drop and re-enter"

            if args.eps:
                if 'collab' in args.dataset.lower():
                    index_end = 250000
                elif 'ppa' in args.dataset.lower():
                    index_end = 5000000
                else: raise "Invalid dataset type, must include 'ppa' or 'collab', (i.e. ogbl-collab)"

                data_name = args.dataset.split('-')[-1]
                data_splt_lst = spl_t.split('_')
                data_splt_name = ''
                for i in data_splt_lst:
                    data_splt_name = data_splt_name + i
                sort_prfx = data_name + data_splt_name
                sorted_edge_path = f'{sort_prfx}seed1{args.eps_model}_{sort_prfx}seed1{args.eps_model}_0_1_sorted_edges.pt'
                print(f"Using sorted edges: {sorted_edge_path}")
                sorted_test_edges = torch.load(f"filtered_edges/{sorted_edge_path}")
                extra_edges = sorted_test_edges[: int(index_end),:2].t().long()
                print('sorted test edges', sorted_test_edges.size())  
                adj_t = add_edges(args.dataset, edge_index, edge_weight, extra_edges, data.num_nodes).cpu() #.to(device)
                A = adj_t @ adj_t
                print("ENTERING ADJ_T REMOVE DIAG ", flush=True)
                A = torch_sparse.remove_diag(A)
                print("ENTERING ADJ_T CSR ", flush=True)
                A = A.to_scipy("csr")
                A[adj_t.to_scipy("csr")>0] = 0 #csc?
                #A = A.to_scipy("csr")
            else:
                A = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), 
                                shape=(node_num, node_num))
                
            if args.edge_drop > 0.0:
                print(f"Dropping edges with probability: {args.edge_drop}")
                A = drop_edges(A, args.edge_drop)
            
            rows, cols = A.nonzero()
            A[cols, rows] = A[rows, cols]

            pos_train_edge = split_edge['train']['edge']
            pos_test_edge = split_edge['test']['edge']
            pos_train_edge = pos_train_edge.transpose(1, 0)
            pos_test_edge = pos_test_edge.transpose(1, 0)
            
            train_scores = eval(args.use_heuristic)(A, pos_train_edge)
            test_scores = eval(args.use_heuristic)(A, pos_test_edge)
            print("Train Scores Size: ", train_scores.size())
            print("Test Scores Size: " ,test_scores.size())
            train_scores = np.nan_to_num(np.log(train_scores), posinf=0, neginf=0)
            test_scores = np.nan_to_num(np.log(test_scores), posinf=0, neginf=0)

            dataset_dict[spl_t] = sst.wasserstein_distance(train_scores, test_scores)

        print(dataset_dict)
        pickle_str = f'{args.use_heuristic}_{dn}'
        if args.eps:
            pickle_str = pickle_str + f'{args.eps_model}_eps'
        if args.edge_drop > 0.0:
            pickle_str = pickle_str + f'_edrp{args.edge_drop}'
            
        pickle.dump(dataset_dict, open('emd_samples/' + pickle_str + '_emd_dict.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        

if __name__ == "__main__":
    main()