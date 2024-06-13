import os 
import torch
import numpy as np
from torch_sparse import SparseTensor
from torch.nn.init import xavier_uniform_

import torch_geometric.transforms as T
from torch_geometric.data import Dataset
from torch_geometric.utils import to_undirected, degree

from util.calc_ppr_scores import get_ppr

FILE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),)
NEGS_DIR = os.path.join(FILE_DIR, "..", "..", "..", "negatives")



class SynthDataset(Dataset):
    def __init__(self, dataset_name, root_dir="dataset"):
        super().__init__(dataset_name)
        self.root_dir = root_dir
        self.dataset_name = dataset_name

    def get(self):
        data = torch.load(f"{self.root_dir}/{self.dataset_name}Dataset/{self.dataset_name}Dataset.pt")
        return data

    def get_edge_split(self):
        split_edge = torch.load(f"{self.root_dir}/{self.dataset_name}Dataset/{self.dataset_name}Dataset_split.pt")
        return split_edge

    def len(self):
        return 0


def read_data(args, seed, device):
    """
    Read data for OGB and synth splits
    """
    data_obj = {
        "dataset": args.data_name,
    }

    print("Loading all data...")

    dataset = SynthDataset(f"{args.data_name}_seed{seed}", root_dir=args.root_dir)
    data = dataset.get()
    split_edge = dataset.get_edge_split()

    data.adj_t = data.adj_t.coalesce().bool().float()
    data_obj['adj_t'] = data.adj_t.to_symmetric().to(device)

    data_obj['num_nodes'] = data.num_nodes
    data_obj['edge_index'] = data.edge_index.to(device)

    data_obj['train_pos'] = split_edge['train']['edge'].to(device)
    data_obj['valid_pos'] = split_edge['valid']['edge'].to(device)
    data_obj['test_pos'] = split_edge['test']['edge'].to(device)

    print("Train/Valid/Test Samples:", len(data_obj['train_pos']), len(data_obj['valid_pos']), len(data_obj['test_pos']))

    # Import Negatives
    path=f"{args.root_dir}/{dataset.dataset_name}Dataset"
    with open(f'{path}/heart_valid_samples.npy', "rb") as f:
        neg_valid_edge = np.load(f)
        data_obj['valid_neg'] = torch.from_numpy(neg_valid_edge).to(device)
    with open(f'{path}/heart_test_samples.npy', "rb") as f:
        neg_test_edge = np.load(f)
        data_obj['test_neg'] = torch.from_numpy(neg_test_edge).to(device)

    # Test train performance without evaluating all test samples
    idx = torch.randperm(data_obj['train_pos'].size(0))[:data_obj['valid_pos'].size(0)]
    data_obj['train_pos_val'] = data_obj['train_pos'][idx]

    if "ogbl" in args.data_name:
        data_obj['x'] = data.x.to(device).to(torch.float)
    else:
        data_obj['x'] =  torch.nn.Parameter(torch.zeros(data_obj['num_nodes'], args.dim).to(device))
        xavier_uniform_(data_obj['x'])
        
    data_obj['adj_mask'] = data_obj['adj_t'].to_symmetric().to_torch_sparse_coo_tensor().to(device)        
    data_obj['adj_mask'] = data_obj['adj_mask'].coalesce().bool().int()

    ### Keep '*_test' or 'full_*' versions below so rest of code will run correctly....
    data_obj['full_adj_t'] = data_obj['adj_t']
    data_obj['full_adj_mask'] = data_obj['adj_mask']

    data_obj['degree'] = degree(data['edge_index'][0], num_nodes=data_obj['num_nodes']).to(device)
    data_obj['degree_test'] = data_obj['degree']

    ### Load PPR matrix
    data_obj['ppr'] = get_ppr(data_obj, alpha=0.15, eps=args.eps).to(device)
    data_obj['ppr_test'] = data_obj['ppr'].to(device)

    return data_obj

