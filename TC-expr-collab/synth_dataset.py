import torch
from torch_geometric.data import Dataset
from utils import find_root

ROOT_DIR = find_root(__file__)

class SynthDataset(Dataset):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        self.dataset_name = dataset_name

    def get(self):
        data = torch.load(f"{ROOT_DIR}/benchmarking/HeaRT_ogb/dataset/{self.dataset_name}Dataset/{self.dataset_name}Dataset.pt")
        return data

    def get_edge_split(self):
        split_edge = torch.load(f"{ROOT_DIR}/benchmarking/HeaRT_ogb/dataset/{self.dataset_name}Dataset/{self.dataset_name}Dataset_split.pt")
        return split_edge

    def len(self):
        return 0