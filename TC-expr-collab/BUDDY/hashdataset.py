
import os
from time import time

import torch
from torch_geometric.data import Dataset
from BUDDY.hashing import ElphHashes
from torch_geometric.utils import to_undirected
from torch_sparse import coalesce
import scipy.sparse as ssp
from BUDDY.heuristic import RA

import torch_sparse
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
from BUDDY.utils import get_same_source_negs

ROOT_DIR = '../ogb_results/'

def find_root(current_dir, marker=".git"):
    current_dir = os.path.abspath(current_dir)
    while not os.path.exists(os.path.join(current_dir, marker)):
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError(f"Root at {marker} not found, file deleted or repository structure changed?")
        current_dir = parent_dir
    return current_dir

DATASET_DIR = find_root(__file__)

class HashDataset(Dataset):
    """
    A class that combines propagated node features x, and subgraph features that are encoded as sketches of
    nodes k-hop neighbors
    """

    def __init__(
            self, root, split, data, pos_edges, neg_edges, args, use_coalesce=False,
            directed=False, **kwargs):
        if args.model != 'ELPH':  # elph stores the hashes directly in the model class for message passing
            self.elph_hashes = ElphHashes(args)  # object for hash and subgraph feature operations
        self.split = split  # string: train, valid or test
        self.root = root
        self.pos_edges = pos_edges
        self.neg_edges = neg_edges
        self.use_coalesce = use_coalesce
        self.directed = directed
        self.args = args
        self.load_features = False # Hard-code for simplicity in argparse
        self.load_hashes = False
        self.use_zero_one = None
        self.cache_subgraph_features = True
        self.max_hash_hops = 2
        self.use_feature = True
        self.use_RA = False
        self.hll_p = 8
        self.subgraph_features = None
        self.hashes = None
        super(HashDataset, self).__init__(root)

        self.links = torch.cat([self.pos_edges, self.neg_edges], 0)  # [n_edges, 2]
        self.labels = [1] * self.pos_edges.size(0) + [0] * self.neg_edges.size(0)

        if self.use_coalesce:  # compress multi-edge into edge with weight
            data.edge_index, data.edge_weight = coalesce(
                data.edge_index, data.edge_weight,
                data.num_nodes, data.num_nodes)

        if 'edge_weight' in data:
            self.edge_weight = data.edge_weight.view(-1)
        else:
            self.edge_weight = torch.ones(data.edge_index.size(1), dtype=int)
        if self.directed:  # make undirected graphs like citation2 directed
            print(
                f'this is a directed graph. Making the adjacency matrix undirected to propagate features and calculate subgraph features')
            self.edge_index, self.edge_weight = to_undirected(data.edge_index, self.edge_weight)
        else:
            self.edge_index = data.edge_index
        self.A = ssp.csr_matrix(
            (self.edge_weight, (self.edge_index[0], self.edge_index[1])),
            shape=(data.num_nodes, data.num_nodes)
        )

        self.degrees = torch.tensor(self.A.sum(axis=0, dtype=float), dtype=torch.float).flatten()

        if self.use_RA:
            self.RA = RA(self.A, self.links, batch_size=2000000)[0]

        if args.model == 'ELPH':  # features propagated in the model instead of preprocessed
            self.x = data.x
        else:
            self.x = self._preprocess_node_features(data, self.edge_index, self.edge_weight, args.sign_k)
        if args.model != 'ELPH':  # ELPH does hashing and feature prop on the fly
            # either set self.hashes or self.subgraph_features depending on cmd args
            self._preprocess_subgraph_features(self.edge_index.device, data.num_nodes, args.dataset, args.num_negs)

    def _generate_sign_features(self, data, edge_index, edge_weight, sign_k):
        """
        Generate features by preprocessing using the Scalable Inception Graph Neural Networks (SIGN) method
         https://arxiv.org/abs/2004.11198
        @param data: A pyg data object
        @param sign_k: the maximum number of times to apply the propagation operator
        @return:
        """
        try:
            num_nodes = data.x.size(0)
        except AttributeError:
            num_nodes = data.num_nodes
        edge_index, edge_weight = gcn_norm(  # yapf: disable
            edge_index, edge_weight.float(), num_nodes)
        if sign_k == 0:
            # for most datasets it works best do one step of propagation
            xs = torch_sparse.spmm(edge_index, edge_weight, data.x.shape[0], data.x.shape[0], data.x)
        else:
            xs = [data.x]
            for _ in range(sign_k):
                x = torch_sparse.spmm(edge_index, edge_weight, data.x.shape[0], data.x.shape[0], data.x)
                xs.append(x)
            xs = torch.cat(xs, dim=-1)
        return xs

    def _preprocess_node_features(self, data, edge_index, edge_weight, sign_k=0):
        """
        preprocess the node features
        @param data: pyg Data object
        @param edge_weight: pyg edge index Int Tensor [edges, 2]
        @param sign_k: the number of propagation steps used by SIGN
        @return: Float Tensor [num_nodes, hidden_dim]
        """
        if sign_k == 0:
            feature_name = f'{self.root}_{self.split}_featurecache.pt'
        else:
            feature_name = f'{self.root}_{self.split}_k{sign_k}_featurecache.pt'
        if self.load_features and os.path.exists(feature_name):
            print('loading node features from disk')
            x = torch.load(feature_name).to(edge_index.device)
        else:
            print('constructing node features')
            start_time = time()
            x = self._generate_sign_features(data, edge_index, edge_weight, sign_k)
            print("Preprocessed features in: {:.2f} seconds".format(time() - start_time))
            if self.load_features:
                torch.save(x.cpu(), feature_name)
        return x

    def _read_subgraph_features(self, name, device):
        """
        return True if the subgraph features can be read off disk, otherwise returns False
        @param name:
        @param device:
        @return:
        """
        retval = False
        # look on disk
        if self.cache_subgraph_features and os.path.exists(name):
            print(f'looking for subgraph features in {name}')
            self.subgraph_features = torch.load(name).to(device)
            print(f"cached subgraph features found at: {name}")
            assert self.subgraph_features.shape[0] == len(
                self.links), 'subgraph features are inconsistent with the link object. Delete subgraph features file and regenerate'
            retval = True
        return retval

    def _generate_file_names(self, num_negs, data_name):
        """
        get the subgraph feature file name and the stubs needed to make a new one if necessary
        :param num_negs: Int negative samples / positive sample
        :return:
        """
        if self.max_hash_hops != 2:
            hop_str = f'{self.max_hash_hops}hop_'
        else:
            hop_str = ''
        if self.args.fnr == True:
            end_str = f'_{hop_str}_eps_subgraph_featurecache.pt'
        elif self.args.tc_buddy == True:
            end_str = f'_{hop_str}_topcon_subgraph_featurecache.pt'
        else:
            end_str = f'_{hop_str}subgraph_featurecache.pt'
        if self.args.dataset == 'ogbl-collab' and self.args.year > 0:
            year_str = f'year_{self.args.year}'
        else:
            year_str = ''
        if num_negs == 1 or self.split != 'train':
            subgraph_cache_name = f'{DATASET_DIR}/benchmarking/HeaRT_ogb/dataset/{data_name}Dataset/{self.root}{self.split}{year_str}{end_str}'
        else:
            subgraph_cache_name = f'{DATASET_DIR}/benchmarking/HeaRT_ogb/dataset/{data_name}Dataset/{self.root}{self.split}_negs{num_negs}{year_str}{end_str}'
        return subgraph_cache_name, year_str, hop_str

    def _preprocess_subgraph_features(self, device, num_nodes, data_name, num_negs=1):
        """
        Handles caching of hashes and subgraph features where each edge is fully hydrated as a preprocessing step
        Sets self.subgraph_features
        @return:
        """
        subgraph_cache_name, year_str, hop_str = self._generate_file_names(num_negs, data_name)
        found_subgraph_features = self._read_subgraph_features(subgraph_cache_name, device)
        if not found_subgraph_features:
            if self.cache_subgraph_features:
                print(f'no subgraph features found at {subgraph_cache_name}')
            print('generating subgraph features')
            hash_name = f'{self.root}{self.split}{year_str}_{hop_str}hashcache.pt'
            cards_name = f'{self.root}{self.split}{year_str}_{hop_str}cardcache.pt'
            if self.load_hashes and os.path.exists(hash_name):
                print('loading hashes from disk')
                hashes = torch.load(hash_name)
                if os.path.exists(cards_name):
                    print('loading cards from disk')
                    cards = torch.load(cards_name)
                else:
                    print(f'hashes found at {hash_name}, but cards not found. Delete hashes and run again')
            else:
                print('no hashes found on disk, constructing hashes...')
                start_time = time()
                hashes, cards = self.elph_hashes.build_hash_tables(num_nodes, self.edge_index)
                print("Preprocessed hashes in: {:.2f} seconds".format(time() - start_time))
                if self.load_hashes:
                    torch.save(cards, cards_name)
                    torch.save(hashes, hash_name)
            print('constructing subgraph features')
            start_time = time()
            print(f"********** SPLIT = {self.split} **************", flush=True)
            print(f"********** POS EDGE SIZE = {self.pos_edges.size()} **************", flush=True)
            print(f"********** NEG EDGE SIZE = {self.neg_edges.size()} **************", flush=True)
            self.subgraph_features = self.elph_hashes.get_subgraph_features(self.links, hashes, cards)
            print("Preprocessed subgraph features in: {:.2f} seconds".format(time() - start_time))
            assert self.subgraph_features.shape[0] == len(
                self.links), 'subgraph features are a different shape link object. Delete subgraph features file and regenerate'
            if self.cache_subgraph_features:
                torch.save(self.subgraph_features, subgraph_cache_name)
        if self.args.floor_sf and self.subgraph_features is not None:
            self.subgraph_features[self.subgraph_features < 0] = 0
            print(
                f'setting {torch.sum(self.subgraph_features[self.subgraph_features < 0]).item()} negative values to zero')
        if not self.use_zero_one and self.subgraph_features is not None:  # knock out the zero_one features (0,1) and (1,0)
            if self.max_hash_hops > 1:
                self.subgraph_features[:, [4, 5]] = 0
            if self.max_hash_hops == 3:
                self.subgraph_features[:, [11, 12]] = 0  # also need to get rid of (0, 2) and (2, 0)

    def len(self):
        return len(self.links)

    def get(self, idx):
        src, dst = self.links[idx]
        if self.args.use_struct_feature:
            subgraph_features = self.subgraph_features[idx]
        else:
            subgraph_features = torch.zeros(self.max_hash_hops * (2 + self.max_hash_hops))

        y = self.labels[idx]
        if self.use_RA:
            RA = self.A[src].dot(self.A_RA[dst].T)[0, 0]
            RA = torch.tensor([RA], dtype=torch.float)
        else:
            RA = -1
        src_degree, dst_degree = get_src_dst_degree(src, dst, self.A, None)
        node_features = torch.cat([self.x[src].unsqueeze(dim=0), self.x[dst].unsqueeze(dim=0)], dim=0)
        return subgraph_features, node_features, src_degree, dst_degree, RA, y

class HashedTrainEvalDataset(Dataset):
    """
    Subset of the full training dataset used to get unbiased estimate of training performance for large datasets
    where otherwise training eval is a significant % of runtime
    """

    def __init__(
            self, links, labels, subgraph_features, RA, dataset):
        self.links = links
        self.labels = labels
        self.edge_index = dataset.edge_index
        self.subgraph_features = subgraph_features
        self.x = dataset.x
        self.degrees = dataset.degrees
        self.RA = RA

    def len(self):
        return len(self.links)

    def get(self, idx):
        return self.links[idx]
    
def get_hashed_train_val_test_datasets(dataset, train_data, val_data, test_data, args, directed=False):
    root = f'{dataset.root}/elph_'
    print(f'data path: {root}')
    use_coalesce = True if args.dataset == 'ogbl-collab' else False
    pos_train_edge, neg_train_edge = get_pos_neg_edges(train_data)
    pos_val_edge, neg_val_edge = get_pos_neg_edges(val_data)
    pos_test_edge, neg_test_edge = get_pos_neg_edges(test_data)
    print(
        f'before sampling, considering a superset of {pos_train_edge.shape[0]} pos, {neg_train_edge.shape[0]} neg train edges '
        f'{pos_val_edge.shape[0]} pos, {neg_val_edge.shape[0]} neg val edges '
        f'and {pos_test_edge.shape[0]} pos, {neg_test_edge.shape[0]} neg test edges for supervision')
    print('constructing training dataset object')
    train_dataset = HashDataset(root, 'train', train_data, pos_train_edge, neg_train_edge, args,
                                use_coalesce=use_coalesce, directed=directed)
    print('constructing validation dataset object')
    val_dataset = HashDataset(root, 'valid', val_data, pos_val_edge, neg_val_edge, args,
                              use_coalesce=use_coalesce, directed=directed)
    print('constructing test dataset object')
    test_dataset = HashDataset(root, 'test', test_data, pos_test_edge, neg_test_edge, args,
                               use_coalesce=use_coalesce, directed=directed)
    return train_dataset, val_dataset, test_dataset

def get_pos_neg_edges(data, sample_frac=1):
    """
    extract the positive and negative supervision edges (as opposed to message passing edges) from data that has been
     transformed by RandomLinkSplit
    :param data: A train, val or test split returned by RandomLinkSplit
    :return: positive edge_index, negative edge_index.
    """
    device = data.edge_index.device # rank.py does not send edge_index to device
    edge_index = data['edge_label_index'].to(device)
    labels = data['edge_label'].to(device)
    pos_edges = edge_index[:, labels == 1].t()
    neg_edges = edge_index[:, labels == 0].t()
    if sample_frac != 1:
        n_pos = pos_edges.shape[0]
        np.random.seed(123)
        perm = np.random.permutation(n_pos)
        perm = perm[:int(sample_frac * n_pos)]
        pos_edges = pos_edges[perm, :]
        neg_edges = neg_edges[perm, :]
    return pos_edges.to(device), neg_edges.to(device)

def get_src_dst_degree(src, dst, A, max_nodes):
    """
    Assumes undirected, unweighted graph
    :param src: Int Tensor[edges]
    :param dst: Int Tensor[edges]
    :param A: scipy CSR adjacency matrix
    :param max_nodes: cap on max node degree
    :return:
    """
    src_degree = A[src].sum() if (max_nodes is None or A[src].sum() <= max_nodes) else max_nodes
    dst_degree = A[dst].sum() if (max_nodes is None or A[src].sum() <= max_nodes) else max_nodes
    return src_degree, dst_degree



def make_train_eval_data(args, train_dataset, num_nodes, n_pos_samples=5000, negs_per_pos=1000):
    """
    A much smaller subset of the training data to get a comparable (with test and val) measure of training performance
    to diagnose overfitting
    @param args: Namespace object of cmd args
    @param train_dataset: pyG Dataset object
    @param n_pos_samples: The number of positive samples to evaluate the training set on
    @return: HashedTrainEvalDataset
    """
    # ideally the negatives and the subgraph features are cached and just read from disk
    # need to save train_eval_negs_5000 and train_eval_subgraph_features_5000 files
    # and ensure that the order is always the same just as with the other datasets
    print('constructing dataset to evaluate training performance')
    dataset_name = args.dataset
    pos_sample = train_dataset.pos_edges[:n_pos_samples]  # [num_edges, 2]

    item1, item2 = dataset_name[:4], dataset_name[5:]
    dataset_name_save = item1 + '_' + item2
    #dataset_name_save = item1 + '-' + item2

    # negs_name = f'{ROOT_DIR}/dataset/{dataset_name_save}/train_eval_negative_samples_{negs_per_pos}.pt'
    # print(f'looking for negative edges at {negs_name}')
    # if os.path.exists(negs_name):
    #     print('loading negatives from disk')
    #     neg_sample = torch.load(negs_name)
    # else:
    print('negatives not found on disk. Generating negatives')
    neg_sample = get_same_source_negs(num_nodes, negs_per_pos, pos_sample.t()).t()  # [num_neg_edges, 2]
    # torch.save(neg_sample, negs_name)
    # make sure these are the correct negative samples with source nodes corresponding to the positive samples
    assert torch.all(torch.eq(pos_sample[:, 0].repeat_interleave(negs_per_pos), neg_sample[:,
                                                                                0])), 'negatives have different source nodes to positives. Delete train_eval_negative_samples_* and subgraph features and regenerate'
    links = torch.cat([pos_sample, neg_sample], 0)  # [n_edges, 2]
    labels = [1] * pos_sample.size(0) + [0] * neg_sample.size(0)
    if train_dataset.use_RA:
        pos_RA = train_dataset.RA[:n_pos_samples]
        neg_RA = RA(train_dataset.A, neg_sample, batch_size=2000000)[0]
        RA_links = torch.cat([pos_RA, neg_RA], dim=0)
    else:
        RA_links = None
    pos_sf = train_dataset.subgraph_features[:n_pos_samples]
    # try to read negative subgraph features from disk or generate them
    if args.tc_buddy == True:
        subgraph_cache_name = f'{DATASET_DIR}/benchmarking/HeaRT_ogb/dataset/{dataset_name_save}/train_eval_negative_samples_{negs_per_pos}_topcon_subgraph_featurecache.pt'
    elif args.fnr == True:
        subgraph_cache_name = f'{DATASET_DIR}/benchmarking/HeaRT_ogb/dataset/{dataset_name_save}/train_eval_negative_samples_{negs_per_pos}_edgeproposalset_subgraph_featurecache.pt'
    else:
        subgraph_cache_name = f'{DATASET_DIR}/benchmarking/HeaRT_ogb/dataset/{dataset_name_save}/train_eval_negative_samples_{negs_per_pos}_subgraph_featurecache.pt'
    print(f'looking for subgraph features at {subgraph_cache_name}')
    if os.path.exists(subgraph_cache_name):
        neg_sf = torch.load(subgraph_cache_name).to(pos_sf.device)
        print(f"cached subgraph features found at: {subgraph_cache_name}")
        assert neg_sf.shape[0] == len(
            neg_sample * negs_per_pos), 'subgraph features are a different shape link object. Delete subgraph features file and regenerate'
    else:  # generate negative subgraph features
        #  we're going to need the hashes
        file_stub = dataset_name.replace('-', '_')  # pyg likes to add -
        if args.max_hash_hops == 3:
            hash_name = f'{DATASET_DIR}/benchmarking/HeaRT_ogb/dataset/{dataset_name_save}/{file_stub}_elph__train_3hop_hashcache.pt'
        else:
            hash_name = f'{DATASET_DIR}/benchmarking/HeaRT_ogb/dataset/{dataset_name_save}/{file_stub}_elph__train_hashcache.pt'
        print(f'looking for hashes at {hash_name}')
        eh = ElphHashes(args)
        if os.path.exists(hash_name):
            hashes = torch.load(hash_name)
            print(f"cached hashes found at: {hash_name}")
        else:  # need to generate the hashes, but this is a corner case as they should have been generated to make the training dataset
            hashes, cards = eh.build_hash_tables(num_nodes, train_dataset.edge_index)
            torch.save(hashes, hash_name)
        print('caching subgraph features for negative samples to evaluate training performance')
        neg_sf = eh.get_subgraph_features(neg_sample, hashes, cards)
        torch.save(neg_sf, subgraph_cache_name)
    subgraph_features = torch.cat([pos_sf, neg_sf], dim=0)
    train_eval_dataset = HashedTrainEvalDataset(links, labels, subgraph_features, RA_links, train_dataset)
    return train_eval_dataset
