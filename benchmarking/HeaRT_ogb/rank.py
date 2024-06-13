import argparse
import random
import os

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected, add_self_loops, negative_sampling
import numpy as np

from ogb.linkproppred import PygLinkPropPredDataset
from synth_dataset import SynthDataset, find_root
from torch_geometric.data import Data, InMemoryDataset
from models import build_model, default_model_configs
from train_and_eval import train, test, hits, test_resource_allocation, buddy_train, buddy_test
from baseline_models.BUDDY.data import get_loaders_hard_neg
from baseline_models.BUDDY.utils import str2bool

from ogb.linkproppred import Evaluator
from torch_sparse import SparseTensor
from torch_sparse import sum as sparse_sum
from logger import Logger
from datetime import datetime
from tqdm import tqdm
from models import LinkPredictor, CommonNeighborsPredictor

from pathlib import Path

ROOT_DIR = find_root(__file__)

class TempDataset(InMemoryDataset):
    def __init__(self, data_name, root=''):
        '''
        - data_name (str): name of the given dataset
        - root (str): root directory to store the dataset folder
        Temporary Dataset Hack to allow ELPH pre-processing in OGB format
        '''
        self.data_name = data_name
        self.root = root

        super(TempDataset, self).__init__(self.root)

    def get_edge_split(self):
        return SynthDataset(dataset_name=self.data_name).get_edge_split()
    
    def get(self):
        return SynthDataset(dataset_name=self.data_name).get()


def add_edges(dataset, edge_index, edge_weight, extra_edges, num_nodes):
    full_edge_index = torch.cat([edge_index.clone(), extra_edges], dim=-1)
    new_edge_weight = torch.ones(extra_edges.shape[1])
    full_edge_weights = torch.cat([edge_weight, new_edge_weight], 0) 
    adj_t = SparseTensor.from_edge_index(full_edge_index, full_edge_weights, sparse_sizes = [num_nodes,num_nodes])
    adj_t = adj_t.to_symmetric() 
    if dataset != "collab":
        adj_t = adj_t.fill_value(1.)
    return adj_t

def get_dataset(dataset):

    if dataset == "ddi":
        dataset = PygLinkPropPredDataset(name='ogbl-ddi')
    elif dataset == "ppa":
        dataset = PygLinkPropPredDataset(name='ogbl-ppa')        
    elif dataset == "collab":
        dataset = PygLinkPropPredDataset(name='ogbl-collab')
    elif 'cn' in dataset.lower() or 'sp' in dataset.lower() or 'pa' in dataset.lower():
        dataset = SynthDataset(dataset_name=dataset) 
    else:
        raise NotImplemented
    return dataset

def make_obg_supervision_edges(split_edge, split, neg_edges=None):
    if neg_edges is not None:
        neg_edges = neg_edges
    else:
        if 'edge_neg' in split_edge[split]:
            neg_edges = split_edge[split]['edge_neg']
            neg_edges = neg_edges.view(-1,2)

    neg_edges = neg_edges.cpu()
    pos_edges = split_edge[split]['edge']
    n_pos, n_neg = pos_edges.shape[0], neg_edges.shape[0]
    edge_label = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)], dim=0)
    edge_label_index = torch.cat([pos_edges, neg_edges], dim=0).t()
    return edge_label, edge_label_index


def get_ogb_train_negs(split_edge, edge_index, num_nodes, num_negs=1, dataset_name=None):
    """
    for some inexplicable reason ogb datasets split_edge object stores edge indices as (n_edges, 2) tensors
    @param split_edge:

    @param edge_index: A [2, num_edges] tensor
    @param num_nodes:
    @param num_negs: the number of negatives to sample for each positive
    @return: A [num_edges * num_negs, 2] tensor of negative edges
    """
   
      # any source is fine
    pos_edge = split_edge['train']['edge'].t()
    new_edge_index, _ = add_self_loops(edge_index)
    neg_edge = negative_sampling(
        new_edge_index, num_nodes=num_nodes,
        num_neg_samples=pos_edge.size(1) * num_negs)
    return neg_edge.t()

def get_ogb_data(data, split_edge, dataset_name, args, num_negs=1):
    """
    ogb datasets come with fixed train-val-test splits and a fixed set of negatives against which to evaluate the test set
    The dataset.data object contains all of the nodes, but only the training edges
    @param dataset:
    @param use_valedges_as_input:
    @return:
    """
    if 'cn' in dataset_name.lower() or 'sp' in dataset_name.lower() or 'pa' in dataset_name.lower():
        read_data_name = dataset_name
    else:
        read_data_name = dataset_name.replace('-', '_')
    if not args.fnr:
        if num_negs == 1:
            # Replace with ROOT_DIR
            negs_name = f'dataset/{read_data_name}Dataset/negative_samples.pt'
        else:
            negs_name = f'dataset/{read_data_name}Dataset/negative_samples_{num_negs}.pt'
        print(f'looking for negative edges at {negs_name}')
        if os.path.exists(negs_name):
            print('loading negatives from disk')
            train_negs = torch.load(negs_name)
        else:
            print('negatives not found on disk. Generating negatives')
            train_negs = get_ogb_train_negs(split_edge, data.edge_index, data.num_nodes, num_negs, dataset_name)
            torch.save(train_negs, negs_name)
    else:
        print('filter and rank specified, generating negatives for this run')
        train_negs = get_ogb_train_negs(split_edge, data.edge_index, data.num_nodes, num_negs, dataset_name)

    splits = {}
    print('train neg number: ', train_negs.size(), flush=True)
    for key in split_edge.keys():
        print(f"Generating Negatives for {key} in get_ogb_data")
        # the ogb datasets come with test and valid negatives, but you have to cook your own train negs
        neg_edges = train_negs if key == 'train' else None
        edge_label, edge_label_index = make_obg_supervision_edges(split_edge, key, neg_edges)
        # use the validation edges for message passing at test time
        # according to the rules https://ogb.stanford.edu/docs/leader_rules/ only collab can use val edges at test time
        if key == 'test' and dataset_name == 'ogbl-collab':
            vei, vw = to_undirected(split_edge['valid']['edge'].t(), split_edge['valid']['weight'])
            edge_index = torch.cat([data.edge_index, vei], dim=1)
            edge_weight = torch.cat([data.edge_weight, vw.unsqueeze(-1)], dim=0)
        else:
            print("data.edge_index: ", data.edge_index)
            edge_index = data.edge_index
            if hasattr(data, "edge_weight"):
                edge_weight = data.edge_weight
            else:
                edge_weight = torch.ones(data.edge_index.shape[1])
        splits[key] = Data(x=data.x, edge_index=edge_index, edge_weight=edge_weight, edge_label=edge_label,
                           edge_label_index=edge_label_index)
    return splits

def get_data(args):
    data = SynthDataset(dataset_name=args.dataset).get()
    if 'ppa' in args.dataset.lower(): data.x = data.x.float()
    
    edge_index = data.edge_index
    edge_weight = torch.ones(data.edge_index.size(1))
    if "edge_weight" in data:
        edge_weight = data.edge_weight.view(-1)
    
    split_edge = SynthDataset(dataset_name=args.dataset).get_edge_split()

    with open(f'{ROOT_DIR}/benchmarking/HeaRT_ogb/dataset/{args.dataset}Dataset/heart_valid_samples.npy', "rb") as f:
        neg_valid_edge = np.load(f)
        neg_valid_edge = torch.from_numpy(neg_valid_edge)
    with open(f'{ROOT_DIR}/benchmarking/HeaRT_ogb/dataset/{args.dataset}Dataset/heart_test_samples.npy', "rb") as f:
        neg_test_edge = np.load(f)
        neg_test_edge = torch.from_numpy(neg_test_edge)

    split_edge['valid']['edge_neg'] = neg_valid_edge
    split_edge['test']['edge_neg'] = neg_test_edge

    if split_edge['valid']['edge'].size(0) < split_edge['train']['edge'].size(0):
        idx = torch.randperm(split_edge['train']['edge'].size(0))[:split_edge['valid']['edge'].size(0)] # For predictions, train shouldn't exceed valid
        split_edge['train']['edge'] = split_edge['train']['edge'][idx]
        print("*************** Permute by Valid *****************")
        idx = torch.randperm(split_edge['valid']['edge_neg'].size(0)) # Randomly permute validation edges to make negative training edges
        split_edge['train']['edge_neg'] = split_edge['valid']['edge_neg'][idx]
        assert split_edge['train']['edge_neg'].size(0) == split_edge['valid']['edge_neg'][idx].size(0), "Train Edge Negatives not equal to size of Valid Edge Negatives in Valid Permutation"
    else:
        print("*************** Slice by Train *****************")
        idx = torch.randperm(split_edge['valid']['edge_neg'].size(0))[:split_edge['train']['edge'].size(0)]
        assert idx.size(0) == split_edge['train']['edge'].size(0), f"Randomly-permuted negative train edge index: {idx.size()}, is not equal to positive training: {split_edge['train']['edge'].size()}, incorrect access time likely"
        split_edge['train']['edge_neg'] = split_edge['valid']['edge_neg'][idx]
 
    #data = T.ToSparseTensor()(data)
    data.adj_t = data.adj_t.to_symmetric()
    
    # features
    if not args.use_feature:
        data.x = None
        
    return edge_index, edge_weight, split_edge, data

def init_seed(seed=2020):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='General Experiment')
    # experiment configs
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--sorted_edge_path', type=str, default= "")
    parser.add_argument('--num_sorted_edge', type=int)
    parser.add_argument('--sweep_max', type=int)
    parser.add_argument('--sweep_min', type=int)
    parser.add_argument('--sweep_num', type=int)
    parser.add_argument('--only_supervision', action="store_true", default=False)
    parser.add_argument('--also_supervision', action="store_true", default=False)
    parser.add_argument('--gen_dataset_only', action="store_true", default=False)
    parser.add_argument('--valid_proposal', action="store_true", default=False)
    
    # save results
    parser.add_argument('--out_name', type=str)
    parser.add_argument('--save_models', action="store_true", default=False)
    
    # model configs; overwrite defaults if specified
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--hidden_channels', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--feature_dropout', type=float)
    parser.add_argument('--label_dropout', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int)
    #parser.add_argument('--use_feature', type=bool)
    parser.add_argument('--use_learnable_embedding', type=bool)
    parser.add_argument('--edge_drop', type=float, default=0.0)
    parser.add_argument('--input_dim', type=int)

    # BUDDY configs -- important for loading hashed datasets
    parser.add_argument('--max_hash_hops', type=int, default=2, help='the maximum number of hops to hash')
    parser.add_argument('--floor_sf', type=str2bool, default=0,
                        help='the subgraph features represent counts, so should not be negative. If --floor_sf the min is set to 0')
    parser.add_argument('--minhash_num_perm', type=int, default=128, help='the number of minhash perms')
    parser.add_argument('--hll_p', type=int, default=8, help='the hyperloglog p parameter')
    parser.add_argument('--use_zero_one', type=str2bool,
                        help="whether to use the counts of (0,1) and (1,0) neighbors")
    parser.add_argument('--load_features', action='store_true', help='load node features from disk')
    parser.add_argument('--load_hashes', action='store_true', help='load hashes from disk')
    parser.add_argument('--cache_subgraph_features', action='store_true',
                        help='write / read subgraph features from disk')
    parser.add_argument('--use_feature', type=str2bool, default=True,
                        help="whether to use raw node features as GNN input")
    parser.add_argument('--use_RA', type=str2bool, default=False, help='whether to add resource allocation features')
    parser.add_argument('--sign_k', type=int, default=0)
    parser.add_argument('--num_negs', type=int, default=1, help='number of negatives for each positive')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_node_embedding', action='store_true',
                        help="also train free-parameter node embeddings together with GNN")
    parser.add_argument('--pretrained_node_embedding', type=str, default=None,
                        help="load pretrained node embeddings as additional node features")
    parser.add_argument('--propagate_embeddings', action='store_true',
                        help='propagate the node embeddings using the GCN diffusion operator')
    parser.add_argument('--add_normed_features', dest='add_normed_features', type=str2bool,
                        help='Adds a set of features that are normalsied by sqrt(d_i*d_j) to calculate cosine sim')
    parser.add_argument('--train_samples', type=float, default=np.inf, help='the number of training edges or % if < 1')
    parser.add_argument('--use_struct_feature', type=str2bool, default=True,
                        help="whether to use structural graph features as GNN input")
    parser.add_argument('--loss', default='bce', type=str, help='bce or auc')

    parser.add_argument('--dynamic_train', action='store_true',
                        help="dynamically extract enclosing subgraphs on the fly")
    parser.add_argument('--dynamic_val', action='store_true')
    parser.add_argument('--dynamic_test', action='store_true')
    parser.add_argument('--eval_batch_size', type=int, default=1024*64,
                        help='eval batch size should be largest the GPU memory can take - the same is not necessarily true at training time')
    
    parser.add_argument('--year', type=int, default=0, help='filter training data from before this year')
    parser.add_argument('--sign_dropout', type=float, default=0.5)

    parser.add_argument('--eval_mrr_data_name', type=str, default='ogbl-citation2')
    parser.add_argument('--test_batch_size', type=int, default=4096)
        
    # other settings
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=20)   
    parser.add_argument('--eval_steps', type=int, default=20)
    parser.add_argument('--score_model', type=str, default='LinkPredictor')

    parser.add_argument('--fnr', action='store_true', default=False)
    
    args = parser.parse_args()
    
    args = default_model_configs(args)
    if args.model == 'buddy' and args.fnr: 
        args.cache_subgraph_features = False
    if args.model == 'resource_allocation': args.dropout = 0.0
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    Path("curves").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True) 
    
    assert not (args.only_supervision and args.also_supervision)

    ##############
    ## test setting
    ##############
    
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    K = hits['collab']
    print("Evaluating at hits: ", K)
    score_func = eval(args.score_model)(args.hidden_channels, args.hidden_channels,
                            1, 3, args.dropout).to(device)
    loggers = {
        f'Hits@{K[0]}': Logger(args.runs),
        f'Hits@{K[1]}': Logger(args.runs),
        f'MRR': Logger(args.runs)
    }

    #args.dataset = args.dataset + '_seed1'
    for run in range(args.runs):
        init_seed(run + 1)

        if args.out_name is None:
            args.out_name = args.dataset + "_" + args.model 

        ##############
        ## load data and model
        ##############
        
        edge_index, edge_weight, split_edge, data = get_data(args)

        data = data.cpu() # Sometimes data is loaded onto the wrong gpu, also required for numpy manipulations
        split_edge['train']['edge'] = split_edge['train']['edge'].cpu()
        split_edge['valid']['edge'] = split_edge['valid']['edge'].cpu()
        split_edge['test']['edge'] = split_edge['test']['edge'].cpu()
        split_edge['train']['edge_neg'] = split_edge['train']['edge_neg'].cpu()
        split_edge['valid']['edge_neg'] =  split_edge['valid']['edge_neg'].cpu()
        split_edge['test']['edge_neg'] = split_edge['test']['edge_neg'].cpu()

        if args.gen_dataset_only:
            return
        
        if args.model is None:
            raise "Model not specified"
        
        if args.model == 'buddy':
            dataset = TempDataset(data_name=args.dataset)
            splits = get_ogb_data(data, split_edge, args.dataset, args, 1)
            train_loader, _, valid_loader, test_loader = get_loaders_hard_neg(args, dataset, splits, False)
        
        data = data.to(device)
        split_edge['train']['edge'] = split_edge['train']['edge'].to(device)
        split_edge['valid']['edge'] = split_edge['valid']['edge'].to(device)
        split_edge['test']['edge'] = split_edge['test']['edge'].to(device)
        split_edge['train']['edge_neg'] = split_edge['train']['edge_neg'].to(device)
        split_edge['valid']['edge_neg'] =  split_edge['valid']['edge_neg'].to(device)
        split_edge['test']['edge_neg'] = split_edge['test']['edge_neg'].to(device)
        
      
        gnn, linkpred, emb, model = build_model(args, data, device)
        print(f'using model {model}')
        
        
        ##############
        ## prepare adding extra edges
        ##############
        
        if args.sorted_edge_path:
            # should be sorted of shape [E, 2] or [E, 3], where the 3rd index is possibly a score
            sorted_test_edges = torch.load(f"filtered_edges/{args.sorted_edge_path}")
            print('sorted test edges', sorted_test_edges.size())            
        else:
            # fake [E, 2]
            #sorted_test_edges = torch.zeros(42, 2)
            assert False, "EPS Failed To Load, does the filtered edges file exist or does BUDDY match the pre-processed features?"
            
        curve = []
        index_ends = []
        
        if args.sweep_num:
            if args.sweep_min is None:
                args.sweep_min = 0
            if args.sweep_max is None:
                args.sweep_max = (args.sweep_num -1)* 1000
            for i in range(args.sweep_num + 1):
                index_end = int(i * (args.sweep_max - args.sweep_min)/(args.sweep_num))
                index_ends.append(args.sweep_min + index_end)
        elif args.num_sorted_edge :
            index_ends.append(args.num_sorted_edge)
        else:
            index_ends.append(0)
        print(f"Scheduled extra edges sweep: {index_ends} x {args.runs}")

            
        ##############
        ## sweeps
        ##############

        for index_end in index_ends: 
            curve_point = []
            print('---------------------')
            print(f'Using {index_end} highest scoring edges')
            print('---------------------')
            
            ##############
            ## adding edges
            ##############
            
            extra_edges = sorted_test_edges[: int(index_end),:2].t().long()
            
            assert extra_edges.size(0) == 2
            assert extra_edges.size(1) == index_end
            
            # Add EPS Edges to training adjacency matrix
            data.adj_t = add_edges(args.dataset, edge_index, edge_weight, extra_edges, data.num_nodes).to(device)  

            model.reset_parameters()
            use_params = sum(p.numel() for p in model.parameters() if p.requires_grad) > 0
            print(sum(p.numel() for p in model.parameters() if p.requires_grad))
            if use_params:
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            else:
                optimizer = None
                
            if not use_params:
                args.epochs = 1

            highest_eval = 0
            for epoch in tqdm(range(1, 1 + args.epochs)):
                if use_params and args.model != 'buddy':
                    loss = train(gnn, score_func, data, args.dataset, split_edge, emb, optimizer,
                                args.batch_size, use_params, args.model, device)
                elif use_params and args.model == 'buddy':
                    loss = buddy_train(model, optimizer, train_loader, args, device)
                else:
                    loss = -1
                if epoch % args.eval_steps == 0:
                    if args.model not in ["adamic_ogb", "resource_allocation","katz", "buddy"]:
                        results = test(gnn, score_func, data, split_edge, emb, evaluator_hit,
                                evaluator_mrr, args.batch_size, device)
                    elif args.model == 'buddy':
                        results, _ = buddy_test(model, evaluator_hit, evaluator_mrr, train_loader,\
                                            valid_loader, test_loader, args, device)
                    elif args.model == "resource_allocation":
                        results = test_resource_allocation(model, data, split_edge, evaluator_hit,
                                args.batch_size, args, device)    
                    for key, result in results.items():
                        loggers[key].add_result(run, result)

                    if epoch % args.log_steps == 0:
                        for key, result in results.items():
                            train_hits, valid_hits, test_hits = result
                            if key == f"Hits@{K[1]}":
                                if valid_hits >= highest_eval:
                                    highest_eval = valid_hits
                                    filename = f'{args.out_name}|{args.sorted_edge_path.split(".")[0]}|{index_end}|{run}.pt'
                                    if args.save_models and use_params:
                                        torch.save(gnn.state_dict(), os.path.join('models', filename + '_gnncmpt'))
                                        torch.save(score_func.state_dict(), os.path.join('models', filename + '_sfncmpt'))
                                        torch.save(model.state_dict(), os.path.join('models', filename))

                            print(key)
                            print(f'Run: {run + 1:02d}, '
                                f'Epoch: {epoch:02d}, '
                                f'Loss: {loss:.4f}, '
                                f'Train: {100 * train_hits:.2f}%, '
                                f'Valid: {100 * valid_hits:.2f}%, '
                                f'Test: {100 * test_hits:.2f}%')
                        print('---', flush=True)

            for key in loggers.keys():
                loggers[key].print_statistics(run)

                if key == f"Hits@{K[1]}":
                    result = 100 * torch.tensor(loggers[key].results[run])
                    argmax = result[:, 1].argmax().item()
                    curve_point = [index_end, result[argmax, 1], result[argmax, 2]]

            time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            filename = f'{args.out_name}|{args.sorted_edge_path.split(".")[0]}|{index_end}|{time}.pt'    
            print(curve_point)
            print("Saving curve to ", filename)
            torch.save(curve_point, os.path.join('curves', filename))
        

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


if __name__ == "__main__":
    main()
