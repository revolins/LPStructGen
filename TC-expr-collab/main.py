from tqdm import tqdm
from parse import parse_args
import torch
import os
from ogb.linkproppred import Evaluator

from utils import *
from model import GCN, MLP
from BUDDY.model import BUDDY
from torch_geometric.data import Data, InMemoryDataset
from synth_dataset import SynthDataset
from BUDDY.data import get_loaders_hard_neg
from dataprocess import load_data
from torch_geometric.utils import to_undirected, add_self_loops, negative_sampling
from learn import train, eval, test, update_adj, eval_comprehensive, buddy_train, buddy_test
import math
import pickle as pkl

from utils import find_root
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

def make_obg_supervision_edges(split_edge, split, neg_edges=None):
    if neg_edges is not None:
        neg_edges = neg_edges
    else:
        if 'edge_neg' in split_edge[split]:
            neg_edges = split_edge[split]['edge_neg']
            neg_edges = neg_edges.view(-1,2)

    pos_edges = split_edge[split]['edge']
    n_pos, n_neg = pos_edges.shape[0], neg_edges.shape[0]
    neg_edges = neg_edges.to(str(pos_edges.device))
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
    # if num_negs == 1:
    #     # Replace with ROOT_DIR
    #     negs_name = f'{ROOT_DIR}/benchmarking/HeaRT_ogb/dataset/{read_data_name}Dataset/negative_samples.pt'
    # else:
    #     negs_name = f'{ROOT_DIR}/benchmarking/HeaRT_ogb/dataset/{read_data_name}Dataset/negative_samples_{num_negs}.pt'
    # print(f'looking for negative edges at {negs_name}')
    # if os.path.exists(negs_name):
    #     print('loading negatives from disk')
    #     train_negs = torch.load(negs_name)
    # else:
    #     print('negatives not found on disk. Generating negatives')
    train_negs = get_ogb_train_negs(split_edge, data.edge_index, data.num_nodes, num_negs, dataset_name)
    # torch.save(train_negs, negs_name)

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

def run(encoder, predictor, split_edge, data, optimizer, adj_list_dict, args, tcs, loader_dict):
    best_val_hit = -math.inf

    data.train_adj_aug = data.train_adj.to(data.x.device)
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    log_res = []

    if args.train:
        for epoch in range(1, 1 + args.epochs):
            if args.model == 'GCN' or args.model == 'GCN-aug':
                loss = train(encoder, predictor, optimizer, data, split_edge['train']['edge'], args)
            if args.model == 'BUDDY' or args.model == 'BUDDY-aug':
                loss = buddy_train(encoder, optimizer, loader_dict['train'], args, True)

            if epoch % args.eval_steps == 0:
                if args.model == 'GCN' or args.model == 'GCN-aug':
                    ress = test(encoder, predictor, data, split_edge, None,\
                                evaluator_hit, evaluator_mrr, args)
                if args.model == 'BUDDY' or args.model == 'BUDDY-aug':
                    ress = buddy_test(encoder, evaluator_hit, evaluator_mrr, loader_dict['train'],\
                                            loader_dict['valid'], loader_dict['test'], args, args.device)

                print('\n\n**********Evaluation Result@{}**********'.format(epoch))
                for key, res in ress.items():
                    print('**********{}**********'.format(key))
                    print(res)
                print(f'Loss: {loss:.4f}')
                
                if ress['MRR'][1] > best_val_hit:
                    best_val_hit = ress['MRR'][1]

                    ress_final = ress
                    
                    if args.save:
                        torch.save(encoder.state_dict(), os.path.join(args.path, 'model', args.dataset, args.model, 'encoder_{}.pt'.format(args.run)))
                        torch.save(predictor.state_dict(), os.path.join(args.path, 'model', args.dataset, args.model, 'predictor_{}.pt'.format(args.run)))
                        torch.save(data.train_adj_aug, os.path.join(args.path, 'model', args.dataset, args.model, 'train_adj_aug_{}.pt'.format(args.run)))

                        for key in tcs:
                            torch.save(tcs[key], os.path.join(args.path, 'model', args.dataset, args.model, '{}_tc_update_{}.pt'.format(key, args.run)))
            
            if args.aug and epoch >= args.warm_up and epoch%args.update_interval == 0 and 'buddy' not in args.model.lower():
                data.train_adj_aug, tcs = update_adj(encoder, predictor, data, epoch, args)
           
    else:
        encoder.load_state_dict(torch.load(os.path.join(args.path, 'model', args.dataset, args.model, 'encoder_{}.pt'.format(args.run))))
        predictor.load_state_dict(torch.load(os.path.join(args.path, 'model', args.dataset, args.model, 'predictor_{}.pt'.format(args.run))))
        data.train_adj_aug = torch.load(os.path.join(args.path, 'model', args.dataset, args.model, 'train_adj_aug_{}.pt'.format(args.run)))

        if args.model == 'GCN-aug' or args.model == 'BUDDY-aug':
            data.train_adj_aug = torch.load(os.path.join(args.path, 'model', args.dataset, args.model, 'train_adj_aug_{}.pt'.format(args.run)))

        ress_final = eval_comprehensive(encoder, predictor, data, evaluator, split_edge, adj_list_dict, 'test', args)
    return ress_final, log_res


if __name__ == '__main__':
    args = parse_args()

    if args.model == 'GCN' or args.model == 'GCN-aug': args.gcn_denoise = True
    if 'aug' in args.model: args.aug = True

    args.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    args.path = os.getcwd()
    temp_dataset_name = args.dataset

    seed_everything(args.seed)
    args.track_idx = 2 #H@20

    evaluator = Evaluator(name='ogbl-collab')
    loader_dict = {'train': None, 'valid': None, 'test': None}
    train_hits, val_hits, test_hits = [], [], []
    train_h20, val_h20, test_h20 = [], [],[]
    pbar = tqdm(range(args.runs), unit='run')

    for args.run in pbar:
        args.dataset = temp_dataset_name + '_seed1'
        print(f"***** Experiment Run on Dataset: ", args.dataset)
        path_everything(args.dataset, args.model)
        seed_everything(args.seed + args.run)

        """build dataset"""
        data, split_edge, adj_list_dict, tcs = load_data(args)


        for key1 in split_edge:
            for key2 in split_edge[key1]:
                split_edge[key1][key2] = split_edge[key1][key2].to(args.device)

        if args.model == 'BUDDY' or args.model == 'BUDDY-aug':
            dataset = TempDataset(data_name=args.dataset)
            splits = get_ogb_data(data, split_edge, args.dataset, 1)
            train_loader, _, valid_loader, test_loader = get_loaders_hard_neg(args, dataset, splits, False)
            loader_dict['train'], loader_dict['valid'], loader_dict['test'] = train_loader, valid_loader, test_loader
        data = data.to(args.device)

        """build encoder"""
        if args.encoder == 'GCN':
            encoder = GCN(data.x.shape[1], args.n_hidden, args.n_hidden, args.n_layers, args.en_dp).to(args.device)
        elif args.encoder == 'BUDDY':
            encoder = BUDDY(args, data.x.size(1), node_embedding=None).to(args.device)
            
        """build link predictor"""
        if args.predictor == 'MLP':
            predictor = MLP(args.n_hidden, args.n_hidden, 1, args.n_layers, args.lp_dp).to(args.device)

        if args.encoder in ['GCN', 'BUDDY']:
            optimizer = torch.optim.Adam([{'params': encoder.parameters(), "lr": args.encoder_lr},
                                          {'params': predictor.parameters(), 'lr': args.predictor_lr}])

        ress_final, log_res = run(encoder, predictor, split_edge, data, optimizer, adj_list_dict, args, tcs, loader_dict)
        
        train_hits.append(ress_final['MRR'][0])
        val_hits.append(ress_final['MRR'][1])
        test_hits.append(ress_final['MRR'][2])
        train_h20.append(ress_final['Hits@20'][0])
        val_h20.append(ress_final['Hits@20'][1])
        test_h20.append(ress_final['Hits@20'][2])

        pkl.dump(log_res, open(os.path.join(args.path, 'model', args.dataset, args.model, 'log_res_{}.pkl'.format(args.run)), 'wb'))

    print('Train_MRR:', np.mean(train_hits, axis = 0), np.std(train_hits, axis = 0))          
    print('Val_MRR:', np.mean(val_hits, axis = 0), np.std(val_hits, axis = 0))
    print('Test_MRR:', np.mean(test_hits, axis = 0), np.std(test_hits, axis = 0))
    print('Train_H@20:', np.mean(train_h20, axis = 0), np.std(train_h20, axis = 0))          
    print('Val_H@20:', np.mean(val_h20, axis = 0), np.std(val_h20, axis = 0))
    print('Test_H@20:', np.mean(test_h20, axis = 0), np.std(test_h20, axis = 0))



