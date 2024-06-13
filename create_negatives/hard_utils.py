import os 
import random
import torch
import numpy as np

from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.data import Dataset
from torch_geometric.utils import to_undirected

# import joblib  # Make ogb loads faster...idk
# from ogb.linkproppred import PygLinkPropPredDataset


def set_seeds():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    os.environ["PYTHONHASHSEED"] = "42"


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

