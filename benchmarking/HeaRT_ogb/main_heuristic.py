import torch
import numpy as np
import argparse
import scipy.sparse as ssp

import sys, os
sys.path.append("..") 

from ogb.linkproppred import Evaluator

from utils import *
from get_heuristic import *
from evalutors import evaluate_mrr
from synth_dataset import SynthDataset


def get_prediction(A, full_A, use_heuristic, pos_val_edge, neg_val_edge, pos_test_edge, neg_test_edge, pos_train_edge, neg_train_edge, args):

    remove = args.remove

    if use_heuristic == 'SP':
        pos_train_pred = eval(use_heuristic)(A, pos_train_edge, remove)
        neg_train_pred = eval(use_heuristic)(A, neg_train_edge, remove)

        pos_val_pred = eval(use_heuristic)( A, pos_val_edge, remove)
        neg_val_pred = eval(use_heuristic)( A, neg_val_edge, remove)

        pos_test_pred = eval(use_heuristic)(full_A, pos_test_edge, remove)
        neg_test_pred = eval(use_heuristic)(full_A, neg_test_edge, remove)
    else:  
        pos_train_pred = eval(use_heuristic)(A, pos_train_edge)
        neg_train_pred = eval(use_heuristic)(A, neg_train_edge)
        pos_val_pred = eval(use_heuristic)(A, pos_val_edge)
        neg_val_pred = eval(use_heuristic)(A, neg_val_edge)
        pos_test_pred = eval(use_heuristic)(full_A, pos_test_edge)
        neg_test_pred = eval(use_heuristic)(full_A, neg_test_edge)

    return pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, pos_train_pred, neg_train_pred

def get_metric_score(evaluator_hit, evaluator_mrr, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, pos_train_pred, neg_train_pred):
    
    k_list = [1, 3, 10, 20, 50, 100]
    result = {}

    result_mrr_train = evaluate_mrr( evaluator_mrr,  pos_train_pred, neg_train_pred)
    result_mrr_val = evaluate_mrr( evaluator_mrr, pos_val_pred, neg_val_pred )
    result_mrr_test = evaluate_mrr( evaluator_mrr, pos_test_pred, neg_test_pred )
    
    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    for K in k_list:
        result[f'Hits@{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

    return result

def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='ogbl-collab_CN_0_1_2')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--use_heuristic', type=str, default='CN')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    
    parser.add_argument('--use_valedges_as_input', action='store_true', default=False)
    parser.add_argument('--use_mrr', action='store_true', default=False)

    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--remove', action='store_true', default=False)
    parser.add_argument('--input_dir', type=str, default=os.path.join(get_root_dir(), "dataset"))

    args = parser.parse_args()
    print(args)

    loggers = {
        'Hits@1': Logger(args.runs),
        'Hits@3': Logger(args.runs),
        'Hits@10': Logger(args.runs),
        'Hits@20': Logger(args.runs),
        'Hits@50': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs)
    }

    for run in range(args.runs):
        print('#################################          ', run, '          #################################')
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run + 1
        print('seed: ', seed)
        if 'ogbl' in args.data_name.lower():
            dataset_name = args.data_name + '_seed1'
            print("Dataset Name: ", dataset_name)
            data = SynthDataset(dataset_name=dataset_name).get()
            split_edge = SynthDataset(dataset_name=dataset_name).get_edge_split()
        else:
            dataset_name = args.data_name + f'_seed{seed}'
            print("Dataset Name: ", dataset_name)
            data = SynthDataset(dataset_name=dataset_name).get()
            split_edge = SynthDataset(dataset_name=dataset_name).get_edge_split()
        
        use_heuristic = args.use_heuristic
        node_num = data.num_nodes
        edge_index = data.edge_index
        edge_weight = torch.ones(data.edge_index.size(1), dtype=int)
        
        pos_train_edge = split_edge['train']['edge']
        pos_valid_edge = split_edge['valid']['edge']
        pos_test_edge = split_edge['test']['edge']
    
        print("************* Loading Heart Negatives ***************")
        with open(f'dataset/{dataset_name}Dataset/heart_valid_samples.npy', "rb") as f:
            neg_valid_edge = np.load(f)
            neg_valid_edge = torch.from_numpy(neg_valid_edge)
        with open(f'dataset/{dataset_name}Dataset/heart_test_samples.npy', "rb") as f:
            neg_test_edge = np.load(f)
            neg_test_edge = torch.from_numpy(neg_test_edge)

        A = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), 
                        shape=(node_num, node_num))
        
        #Enforce Symmetry
        rows, cols = A.nonzero()
        A[cols, rows] = A[rows, cols]

        if pos_valid_edge.size(0) < pos_train_edge.size(0):
            idx = torch.randperm(pos_train_edge.size(0))[:pos_valid_edge.size(0)] # For predictions, train shouldn't exceed valid
            train_val_edge = pos_train_edge[idx]
            print("*************** Permute by Valid *****************")
            idx = torch.randperm(neg_valid_edge.size(0)) # Randomly permute validation edges to make negative training edges
            neg_train_edge = neg_valid_edge[idx]
        else:
            train_val_edge = pos_train_edge
            print("*************** Slice by Train *****************")
            print(f"******** train_val_edge size in statement: {train_val_edge.size()} **************")
            print(f"******** neg_valid_edge size in statement: {neg_valid_edge.size()} **************")
            idx = torch.randperm(neg_valid_edge.size(0))[:train_val_edge.size(0)]
            assert idx.size(0) == train_val_edge.size(0), f"Somehow the randomly-permuted negative train edge index: {idx.size()}, is not equal to positive training: {train_val_edge.size()}"
            neg_train_edge = neg_valid_edge[idx]

        train_val_edge = train_val_edge.transpose(1, 0)
        pos_valid_edge = pos_valid_edge.transpose(1, 0)
        pos_test_edge = pos_test_edge.transpose(1, 0)

        neg_train_edge  = torch.permute(neg_train_edge, (2, 0, 1))
        neg_valid_edge = torch.permute(neg_valid_edge, (2, 0, 1))
        neg_test_edge = torch.permute(neg_test_edge, (2, 0 , 1))

        neg_train_edge = neg_train_edge.view(2, -1)
        neg_valid_edge = neg_valid_edge.view(2, -1)
        neg_test_edge = neg_test_edge.view(2, -1)
    
        print(f'edge sizes - pos_valid_edge: {pos_valid_edge.size()}, neg_valid_edge: {neg_valid_edge.size()}, pos_test_edge: {pos_test_edge.size()}, neg_test_edge: {neg_test_edge.size()}, pos_train_edge: {pos_train_edge.size()}, train_val_edge: {train_val_edge.size()}, neg_train_edge: {neg_train_edge.size()}')
        pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, train_val_pred, neg_train_pred = get_prediction(A, A, use_heuristic, pos_valid_edge, neg_valid_edge, pos_test_edge, neg_test_edge, train_val_edge, neg_train_edge, args)
       
        evaluator_hit = Evaluator(name='ogbl-collab')
        evaluator_mrr = Evaluator(name='ogbl-citation2')

        print(f'predictions before reshaping: pos_val_pred.size(): {pos_val_pred.size()}, neg_val_pred.size(): {neg_val_pred.size()}, pos_test_pred.size(): {pos_test_pred.size()}, neg_test_pred.size(): {neg_test_pred.size()}, train_val_pred: {train_val_pred.size()}, neg_train_pred: {neg_train_pred.size()}')

        # idx = torch.randperm(neg_val_pred.size(0))[:train_val_pred.size(0) * 500] # For evaluation, train randomly samples from negative validation samples
        # neg_train_val_pred = neg_val_pred[idx] 
    
        neg_val_pred = neg_val_pred.view(pos_val_pred.size(0), -1)
        neg_test_pred = neg_test_pred.view(pos_test_pred.size(0), -1)
        neg_train_pred = neg_train_pred.view(train_val_pred.size(0), -1)

        print(f'predictions after view: pos_val_pred.size(): {pos_val_pred.size()}, neg_val_pred.size(): {neg_val_pred.size()}, pos_test_pred.size(): {pos_test_pred.size()}, neg_test_pred.size(): {neg_test_pred.size()}, train_val_pred: {train_val_pred.size()}, neg_train_pred: {neg_train_pred.size()}')

        results = get_metric_score(evaluator_hit, evaluator_mrr, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, train_val_pred, neg_train_pred)
        print('heuristic: ', args.use_heuristic)    
    
        for key, result in results.items():
            train_hits, valid_hits, test_hits = result
            loggers[key].add_result(run, result)
            print(key)
            print( f'Train: {100 * train_hits:.2f}%, '
                                f'Valid: {100 * valid_hits:.2f}%, '
                                f'Test: {100 * test_hits:.2f}%')
            
        for key in loggers.keys():
            print("******** Logger Results ********")
            print(key)
            loggers[key].print_statistics(run)

    result_all_run = {}
    for key in loggers.keys():
        print(key)
        
        _,  _, mean_list, var_list = loggers[key].print_statistics()
        result_all_run[key] = [mean_list, var_list]

if __name__ == "__main__":
    main()