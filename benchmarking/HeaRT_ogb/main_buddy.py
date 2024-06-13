from math import inf
import sys

sys.path.insert(0, '..')

from utils import *
import numpy as np
import torch, os
from ogb.linkproppred import Evaluator
from torch_geometric.utils import to_undirected
from baseline_models.BUDDY.data import get_loaders_hard_neg
from baseline_models.BUDDY.utils import select_embedding, select_model, get_num_samples, get_loss, get_split_samples, str2bool
from torch.utils.data import DataLoader
from evalutors import evaluate_hits, evaluate_mrr
import argparse
from ogb.linkproppred import Evaluator
from baseline_models.BUDDY.utils import filter_by_year
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import (add_self_loops, negative_sampling,
                                   to_undirected)
from synth_dataset import SynthDataset

log_print		= get_logger('testrun', 'log', get_config_dir())

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

def make_obg_supervision_edges(split_edge, split, neg_edges=None):
    if neg_edges is not None:
        neg_edges = neg_edges
    else:
        if 'edge_neg' in split_edge[split]:
            neg_edges = split_edge[split]['edge_neg']
            neg_edges = neg_edges.view(-1,2)

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


def get_ogb_data(data, split_edge, dataset_name, num_negs=1):
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
    # else:
    #     train_negs = get_ogb_train_negs(split_edge, data.edge_index, data.num_nodes, num_negs, dataset_name)
    splits = {}
    print('train neg number: ', train_negs.size(), flush=True)
    for key in split_edge.keys():
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
            edge_index = data.edge_index
            if hasattr(data, "edge_weight"):
                edge_weight = data.edge_weight
            else:
                edge_weight = torch.ones(data.edge_index.shape[1])
        splits[key] = Data(x=data.x, edge_index=edge_index, edge_weight=edge_weight, edge_label=edge_label,
                           edge_label_index=edge_label_index)
    return splits

def get_data(args):
    directed = False

    dataset = TempDataset(data_name=args.dataset)
    data = SynthDataset(dataset_name=args.dataset).get()
    split_edge = SynthDataset(dataset_name=args.dataset).get_edge_split()

    with open(f'dataset/{args.dataset}Dataset/heart_valid_{args.filename}', "rb") as f:
        neg_valid_edge = np.load(f)
        neg_valid_edge = torch.from_numpy(neg_valid_edge)
    with open(f'dataset/{args.dataset}Dataset/heart_test_{args.filename}', "rb") as f:
        neg_test_edge = np.load(f)
        neg_test_edge = torch.from_numpy(neg_test_edge)
        
    split_edge['valid']['edge_neg'] = neg_valid_edge
    split_edge['test']['edge_neg'] = neg_test_edge

    while neg_valid_edge.size(0) <= args.batch_size: 
        print("Negative Validation Edges are smaller than batch, reducing batch size")
        args.batch_size = args.batch_size // 2
        if args.batch_size == 0:
            raise Exception("Batch Size Reached 0 in Neg. Val. Edges")
            
    while neg_test_edge.size(0) <= args.eval_batch_size:
        print("Negative Testing Edges are smaller than batch, reducing batch size")
        args.eval_batch_size = args.eval_batch_size // 2
        if args.eval_batch_size == 0:
            raise Exception("Batch Size Reached 0 in Neg. Test Edges")

    pos_train_edge = split_edge['train']['edge']
    pos_valid_edge = split_edge['valid']['edge']

    while split_edge['train']['edge'].size(0) <= args.batch_size:
                print("Positive Training Edges are smaller than batch, reducing batch size")
                args.batch_size = args.batch_size // 2
                if args.batch_size <= 0:
                    raise Exception("Batch Size Reached 0 in Pos. Train Edges")
            
    while split_edge['valid']['edge'].size(0) <= args.batch_size:
        print("Positive Validation Edges are smaller than batch, reducing batch size")
        args.batch_size = args.batch_size // 2
        if args.batch_size <= 0:
            raise Exception("Batch Size Reached 0 in Pos. Val. Edges")
        
    while split_edge['test']['edge'].size(0) <= args.eval_batch_size:
        print("Positive Testing Edges are smaller than testing batch, reducing testing batch size")
        args.eval_batch_size = args.eval_batch_size // 2
        if args.eval_batch_size <= 0:
            raise Exception("Batch Size reached 0 in Pos. Testing Edges")

    if pos_valid_edge.size(0) < pos_train_edge.size(0):
        idx = torch.randperm(pos_train_edge.size(0))[:pos_valid_edge.size(0)] # For predictions, train shouldn't exceed valid
        train_val_edge = pos_train_edge[idx]
        print("*************** Permute by Valid *****************")
        idx = torch.randperm(neg_valid_edge.size(0)) # Randomly permute validation edges to make negative training edges
        neg_train_edge = neg_valid_edge[idx]
    else:
        train_val_edge = pos_train_edge
        print("*************** Slice by Train *****************")
        idx = torch.randperm(neg_valid_edge.size(0))[:train_val_edge.size(0)]
        assert idx.size(0) == train_val_edge.size(0), f"Somehow the randomly-permuted negative train edge index: {idx.size()}, is not equal to positive training: {train_val_edge.size()}"
        neg_train_edge = neg_valid_edge[idx]
    
    print("*************** NEGATIVE TRAIN EDGE SIZE IN BUDDY LOADING **************: ", neg_train_edge.size())
    print("*************** NEGATIVE VALID EDGE SIZE IN BUDDY LOADING **************: ", neg_valid_edge.size())
    assert neg_train_edge != None, "Negative Training Edges are not assigned, loading failed."
    split_edge['train']['edge_neg'] = neg_train_edge

    print('train_pos train_neg val val_neg test test_neg: ', split_edge['train']['edge'].size(), split_edge['train']['edge_neg'].size(), split_edge['valid']['edge'].size(), split_edge['valid']['edge_neg'].size(), split_edge['test']['edge'].size(), split_edge['test']['edge_neg'].size())

    if args.dataset == 'ogbl-collab' and args.year > 0:  # filter out training edges before args.year
        data, split_edge = filter_by_year(data, split_edge, args.year)
    print("###########")
    print("args.dataset in get data")
    print(args.dataset, flush=True)
    splits = get_ogb_data(data, split_edge, args.dataset, args.num_negs)
    
    return dataset, splits, directed

def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    
    k_list = [10, 20]
    result = {}

    result_mrr_train = evaluate_mrr( evaluator_mrr,  pos_train_pred, neg_train_pred)
    result_mrr_val = evaluate_mrr( evaluator_mrr, pos_val_pred, neg_val_pred )
    result_mrr_test = evaluate_mrr( evaluator_mrr, pos_test_pred, neg_test_pred )
    
    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    for K in k_list:
        result[f'Hits@{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

    return result

def train(model, optimizer, train_loader, args, device, emb=None):

    model.train()
    total_loss = 0
    data = train_loader.dataset
    # hydrate edges
    links = data.links
    labels = torch.tensor(data.labels)
    # sampling
    train_samples = get_num_samples(args.train_samples, len(labels))
    sample_indices = torch.randperm(len(labels))[:train_samples]
    links = links[sample_indices]
    labels = labels[sample_indices]

    batch_processing_times = []
    loader = DataLoader(range(len(links)), args.batch_size, shuffle=True)
    for batch_count, indices in enumerate(loader):
        # do node level things
        if model.node_embedding is not None:
            if args.propagate_embeddings:
                emb = model.propagate_embeddings_func(data.edge_index.to(device))
            else:
                emb = model.node_embedding.weight
        else:
            emb = None
        curr_links = links[indices]
        batch_emb = None if emb is None else emb[curr_links].to(device)

        if args.use_struct_feature:
            sf_indices = sample_indices[indices]  # need the original link indices as these correspond to sf
            subgraph_features = data.subgraph_features[sf_indices].to(device)
        else:
            subgraph_features = torch.zeros(data.subgraph_features[indices].shape).to(device)
        node_features = data.x[curr_links].to(device)
        degrees = data.degrees[curr_links].to(device)
        if args.use_RA:
            ra_indices = sample_indices[indices]
            RA = data.RA[ra_indices].to(device)
        else:
            RA = None
        
        optimizer.zero_grad()
        logits = model(subgraph_features, node_features, degrees[:, 0], degrees[:, 1], RA, batch_emb)
        loss = get_loss(args.loss)(logits, labels[indices].squeeze(0).to(device))

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * args.batch_size
    
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test_edge(model, loader, device, args, split=None):

    n_samples = get_split_samples(split, args, len(loader.dataset))
    preds = []
    data = loader.dataset
    # hydrate edges
    links = data.links
    labels = torch.tensor(data.labels)
    loader = DataLoader(range(len(links)), args.eval_batch_size,
                        shuffle=False)  # eval batch size should be the largest that fits on GPU
    if model.node_embedding is not None:
        if args.propagate_embeddings:
            emb = model.propagate_embeddings_func(data.edge_index.to(device))
        else:
            emb = model.node_embedding.weight
    else:
        emb = None
    for batch_count, indices in enumerate(loader):
        curr_links = links[indices]
        batch_emb = None if emb is None else emb[curr_links].to(device)
        if args.use_struct_feature:
            subgraph_features = data.subgraph_features[indices].to(device)
        else:
            subgraph_features = torch.zeros(data.subgraph_features[indices].shape).to(device)
        node_features = data.x[curr_links].to(device)
        degrees = data.degrees[curr_links].to(device)
        if args.use_RA:
            RA = data.RA[indices].to(device)
        else:
            RA = None
        logits = model(subgraph_features, node_features, degrees[:, 0], degrees[:, 1], RA, batch_emb)
        preds.append(logits.view(-1).cpu())
        if (batch_count + 1) * args.eval_batch_size > n_samples:
            break

    pred = torch.cat(preds)
    labels = labels[:len(pred)]
    pos_pred = pred[labels == 1]
    neg_pred = pred[labels == 0]
    return pos_pred, neg_pred

@torch.no_grad()
def test(model, evaluator_hit, evaluator_mrr, train_loader, val_loader, test_loader, args, device):
   
    model.eval()

    pos_train_pred, neg_train_pred = test_edge(model, train_loader, device, args, split='train')

    pos_valid_pred, neg_valid_pred = test_edge(model, val_loader, device, args, split='val')
    
    pos_test_pred, neg_test_pred  = test_edge(model, test_loader, device, args, split='test')

    pos_train_pred = torch.flatten(pos_train_pred)
    pos_valid_pred = torch.flatten(pos_valid_pred)
    pos_test_pred =  torch.flatten(pos_test_pred)
    
    neg_train_pred = neg_train_pred.view(pos_train_pred.size(0), -1)
    neg_valid_pred = neg_valid_pred.view(pos_valid_pred.size(0), -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.size(0), -1)

    #pos_train_pred = pos_valid_pred

    print('train_pos train_neg valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), neg_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, neg_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)

    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu()]

    return result, score_emb

def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--dataset', type=str, default='ogbl-collab')
    
    ##gnn setting
    parser.add_argument('--hidden_channels', type=int, default=256)
    
    ### train setting
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=20,    type=int,       help='early stopping')
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--input_dir', type=str, default=os.path.join(get_root_dir(), "dataset"))
    parser.add_argument('--filename', type=str, default='samples.npy')
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=1)
    
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--metric', type=str, default='MRR')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)

    ##
    parser.add_argument('--model', type=str, default='BUDDY')
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
    parser.add_argument('--label_dropout', type=float, default=0.5)
    parser.add_argument('--feature_dropout', type=float, default=0.5)
    parser.add_argument('--propagate_embeddings', action='store_true',
                        help='propagate the node embeddings using the GCN diffusion operator')
    parser.add_argument('--add_normed_features', dest='add_normed_features', type=str2bool,
                        help='Adds a set of features that are normalsied by sqrt(d_i*d_j) to calculate cosine sim')
    parser.add_argument('--train_samples', type=float, default=inf, help='the number of training edges or % if < 1')
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
    parser.add_argument('--edge_drop', type=float, default=0.0)

    parser.add_argument('--eval_mrr_data_name', type=str, default='ogbl-citation2')
    parser.add_argument('--test_batch_size', type=int, default=4096)
    parser.add_argument('--fnr', action='store_true', default=False)
    parser.add_argument('--tc_buddy', action='store_true', default=False)
    
    args = parser.parse_args()
    args.test_batch_size = args.eval_batch_size
    print(args)

    init_seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name=args.eval_mrr_data_name)

    loggers = {
        'Hits@10': Logger(args.runs),
        'Hits@20': Logger(args.runs),
        'MRR': Logger(args.runs)
    }

    eval_metric = 'MRR'
    
    temp_data_name = args.dataset

    for run in range(args.runs):
        print('#################################          ', run, '          #################################')
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run + 1
        print('seed: ', seed)

        init_seed(seed)
        if 'cn' in args.dataset.lower() or 'pa' in args.dataset.lower() or 'sp' in args.dataset.lower() or 'sbm' in args.dataset.lower():
            print('args data name in run: ', str(args.dataset))
            if 'ogbl' in args.dataset.lower():
                args.dataset = temp_data_name + '_seed1'
            else:
                args.dataset = temp_data_name + f'_seed{seed}'
            print('args data name after update: ', str(args.dataset))
            data = SynthDataset(dataset_name=args.dataset).get() #Quick fix for handling synthetic data
            dataset, splits, directed = get_data(args) 
            train_loader, train_eval_loader, val_loader, test_loader = get_loaders_hard_neg(args, dataset, splits, directed)
        save_path = args.output_dir+'/lr'+str(args.lr) + '_l2'+ str(args.l2) + '_dp' + str(args.feature_dropout) +'_dim'+str(args.hidden_channels) + '_edgedrop' + str(args.edge_drop) + '_'+ 'best_run_'+str(seed)+'_'+str(args.dataset)+ '_BUDDY_'
        emb = select_embedding(args, data.num_nodes, device)
        model, optimizer = select_model(args, dataset, emb, device)

        best_valid = 0
        kill_cnt = 0
        best_test = 0

        for epoch in range(1, 1 + args.epochs):
           
            loss = train(model, optimizer, train_loader, args, device)

            if epoch % args.eval_steps == 0:
                results_rank, score_emb = test(model, evaluator_hit, evaluator_mrr, train_loader, val_loader, test_loader, args, device)

                for key, result in results_rank.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results_rank.items():
                        
                        print(key)
                        
                        train_hits, valid_hits, test_hits = result
                        log_print.info(
                            f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')

                    r = torch.tensor(loggers[eval_metric].results[run])
                    best_valid_current = round(r[:, 1].max().item(),4)
                    best_test = round(r[r[:, 1].argmax(), 2].item(), 4)

                    print(eval_metric)
                    log_print.info(f'best valid: {100*best_valid_current:.2f}%, '
                                   f'best test: {100*best_test:.2f}%')
                    print('---')
                
                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0
                    if args.save: 
                        torch.save(model.state_dict(), save_path+'_model.pt')
                        # torch.save(optimizer.state_dict(),save_path+'_op.pt')
                        # torch.save(emb,save_path+'_emb.pt')
                        # torch.save(score_emb, save_path+'_scoremb.pt')
                else:
                    kill_cnt += 1
                    if kill_cnt > args.kill_cnt: 
                        print("Early Stopping!!")
                        break
        
        for key in loggers.keys():
            if len(loggers[key].results[run]) > 0:
                print(key)
                loggers[key].print_statistics(run)
    
    result_all_run = {}
    for key in loggers.keys():
        if len(loggers[key].results[run]) > 0:
            print(key)
            
            _,  _, mean_list, var_list = loggers[key].print_statistics()
            result_all_run[key] = [mean_list, var_list]

if __name__ == "__main__":
    main()
