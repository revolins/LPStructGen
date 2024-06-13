import torch
from torch_geometric.data import Dataset
import os

def find_root(current_dir, marker=".git"):
    current_dir = os.path.abspath(current_dir)
    while not os.path.exists(os.path.join(current_dir, marker)):
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError(f"Root at {marker} not found, file deleted or repository structure changed?")
        current_dir = parent_dir
    return current_dir

class SynthDataset(Dataset):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        self.dataset_name = dataset_name

    def get(self):
        data = torch.load(f"dataset/{self.dataset_name}Dataset/{self.dataset_name}Dataset.pt")
        return data

    def get_edge_split(self):
        split_edge = torch.load(f"dataset/{self.dataset_name}Dataset/{self.dataset_name}Dataset_split.pt")
        return split_edge

    def len(self):
        return 0