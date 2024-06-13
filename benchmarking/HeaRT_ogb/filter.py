import argparse

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import normalize
from torch_geometric.data import Data

import scipy.sparse as ssp
import numpy as np
from train_and_eval import resource_allocation
import torch_sparse
from adamic_utils import get_A
from models import LinkGNN
from baseline_models.BUDDY.data import get_loaders_hard_neg
from baseline_models.BUDDY.utils import get_split_samples, str2bool
from baseline_models.BUDDY.hashdataset import HashDataset

from rank import get_data, add_edges, TempDataset
from synth_dataset import find_root

from models import build_model, default_model_configs
from pathlib import Path
from tqdm import tqdm

ROOT_DIR = find_root(__file__)

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

def main():
    parser = argparse.ArgumentParser(description='ddi (GNN)')
    # experiment configs
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--run', type=int, required=True)
    parser.add_argument('--checkpoint', type=str) # Should automatically update node based on HeaRT settings
    
    # model configs; overwrite defaults if specified
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--hidden_channels', type=int)
    parser.add_argument('--input_dim', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--feature_dropout', type=float)
    parser.add_argument('--label_dropout', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int)
    #parser.add_argument('--use_feature', type=bool)
    parser.add_argument('--use_learnable_embedding', type=bool)
    parser.add_argument('--edge_drop', type=float, default=0.0)
#     parser.add_argument('--use_node_embedding', action="store_true", default=False)
    
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

    parser.add_argument('--eval_mrr_data_name', type=str, default='ogbl-citation2')
    parser.add_argument('--test_batch_size', type=int, default=4096)

    parser.add_argument('--fnr', action='store_true', default=False)
    
    # other settings
    parser.add_argument('--device', type=int, default=0)
    
    args = parser.parse_args()
    args = default_model_configs(args)
    if args.model == 'buddy': 
        args.fnr = True
        args.cache_subgraph_features = False
    print(args)
    
    Path("filtered_edges").mkdir(exist_ok=True)
   
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    ##############
    ## load data and model
    ##############
    edge_index, edge_weight, split_edge, data = get_data(args)
    split_edge['train']['edge'] = split_edge['train']['edge'].to(device)
    split_edge['valid']['edge'] = split_edge['valid']['edge'].to(device)
    split_edge['test']['edge'] = split_edge['test']['edge'].to(device)
    split_edge['train']['edge_neg'] = split_edge['train']['edge_neg'].to(device)
    split_edge['valid']['edge_neg'] = split_edge['valid']['edge_neg'].to(device)
    split_edge['test']['edge_neg'] = split_edge['test']['edge_neg'].to(device)
    data = data.to(device)
    gnn, linkpred, emb, model = build_model(args, data, device)
    
    print(f'using gnn {gnn}')
    print(f'using linkpred {linkpred}')
    print(f'using emb {emb}')
    print(f'using model {model}')

    if args.model == 'buddy':
        args.checkpoint = f'lr{args.lr}_l20.0_dp{args.dropout}_dim256_best_run_{args.run}_{args.dataset}_BUDDY__model.pt'
        
    elif args.model == 'gcn':
        temp_dataset = args.dataset.split('_')
        temp_dataset.pop(-1)
        gcn_read_data = ''
        for i in range(len(temp_dataset)):
            if i != len(temp_dataset) - 1:
                gcn_read_data += temp_dataset[i] + '_'
            else:
                gcn_read_data += temp_dataset[i]
        args.checkpoint = f'lr{args.lr}_drop{args.dropout}_l20.0_numlayer3_numPredlay3_numGinMlplayer2_dim128_edrop0.0_best_run_{args.run}_{gcn_read_data}_GCN'
    
    use_params = sum(p.numel() for p in model.parameters() if p.requires_grad) > 0
    print('using params?', use_params)
    if use_params and args.model == 'gcn':
        gnn_state_dict = torch.load(f'{ROOT_DIR}/benchmarking/HeaRT_ogb/models/{args.checkpoint}__model.pt', map_location=device)
        gnn.load_state_dict(gnn_state_dict, strict=False)
        linkpred.load_state_dict(torch.load(f'{ROOT_DIR}/benchmarking/HeaRT_ogb/models/{args.checkpoint}__predictor.pt', map_location=device))
        emb = data.x
        model = LinkGNN(emb, gnn, linkpred)
  
    if args.model == 'buddy' or args.model == 'gcn':
        parts = args.checkpoint.split("_")
        print("split checkpoint: ", parts)
    
    dataname_save = args.dataset.split('_')
    for i in dataname_save:
        if 'ogbl-' in i:
            cleanse = i.split('-')
            cleanse.pop(0)
            dataname_save.pop(0)
            dataname_save.extend(cleanse)
    print("current split checkpoint: ", dataname_save)
    split_name = dataname_save[-1]
    for i in range(len(dataname_save) - 1):
        split_name  += dataname_save[i]
    spec = split_name + args.model
    sorted_edge_path = split_name + args.model
    
    data.adj_t = add_edges(args.dataset, edge_index, edge_weight, torch.zeros([2,0], dtype=int) , data.num_nodes).to(device)
        
    model.eval()
    
    all_scores = []
    # restrict to edges that have at least one common neighbor for relevant models
    if True:
        print("ENTERING ADJ_T MANIPULATIONS ", flush=True)
        adj_t = data.adj_t.cpu() #.to(device)
        print("ENTERING ADJ_T EDGEWISE ", flush=True)
        A2 = adj_t @ adj_t
        print("ENTERING ADJ_T REMOVE DIAG ", flush=True)
        A2 = torch_sparse.remove_diag(A2)
        print("ENTERING ADJ_T CSC ", flush=True)
        A2 = A2.to_scipy("csc")
        # dont compute for edges that we are know positive
        A2[adj_t.to_scipy("csc")>0] = 0
        print("ENTERING ADJ_T FROM SCIPY ", flush=True)
        indices, values = torch_sparse.from_scipy(A2)
        print("ENTERING VALUES SQUEEZE NONZERO ", flush=True)
        selected = values.nonzero().squeeze(1)
        print("ENTERING ADJ_T UNSQUEEZE M ", flush=True)
        m = torch.cat([indices[:, selected].t(), values[selected].unsqueeze(1)], 1).long()
        print("ENTERING M SLICE ", flush=True)
        all_edges = m[:,:2]
        
        print(f'using {all_edges.size()} edges')
            
        if args.model not in ["adamic_ogb", "resource_allocation"]:
            with torch.no_grad():
                if args.model.lower() == 'buddy':
                    all_edges_index = all_edges.t() #HashDataset pre-processing to load just the Edge Proposal Set.
                    assert all_edges_index.size(0) == 2, "CN Edge Proposal Set improper shape in BUDDY loader"
                    dataset = TempDataset(data_name=args.dataset)
                    full_edge_data = Data(edge_index=all_edges_index, x=data.x, adj_t=adj_t, num_nodes=data.num_nodes,\
                                          edge_label=torch.cat([torch.ones(all_edges_index.size(1) - 1), torch.zeros(1)], dim=0) ,\
                                        edge_label_index=torch.cat([all_edges[:-1, :], all_edges[-1:, :]], dim=0).t()).cpu()
                    pos_train_edge, neg_train_edge = get_pos_neg_edges(full_edge_data)
                    root = f'{dataset.root}/elph_'
                    train_dataset = HashDataset(root, 'train', full_edge_data, pos_train_edge, neg_train_edge, args,
                                        use_coalesce=False, directed=False) #use_coalesce=False since SynthDataset is coalesced, undirected
                    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers)
                    
                    n_samples = len(train_loader.dataset)
                    data = train_loader.dataset
                    links = data.links
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
                        batch_emb = None if emb is None else emb[curr_links].cpu() #.to(device)
                        subgraph_features = data.subgraph_features[indices].cpu() #.to(device)
                        node_features = data.x[curr_links].cpu() #.to(device)
                        degrees = data.degrees[curr_links].cpu() #.to(device)
                        RA = None
                        logits = model(subgraph_features, node_features, degrees[:, 0], degrees[:, 1], RA, batch_emb).squeeze()
                        edge_score = torch.cat([curr_links, logits.unsqueeze(1)], dim=1).cpu() # curr_links.t()
                        all_scores.append(edge_score)
                        if (batch_count + 1) * args.eval_batch_size > n_samples: break
                elif args.model.lower() == 'gcn':
                    all_edges = all_edges.t()
                    for perm in tqdm(DataLoader(range(all_edges.size(1)), args.batch_size)):
                        edges = all_edges[:, perm].to(device)
                        score = model(data.x, edges, data.adj_t).squeeze()
                        edge_score = torch.cat([edges.t(), score.unsqueeze(1)], dim=1).cpu()
                        all_scores.append(edge_score)
                else: assert False, f"Can't load edges for {args.model}, should be resource_allocation, GCN, BUDDY"                    
            all_scores = torch.cat(all_scores, 0)
            #print("all_scores in model function: ", all_scores)
        elif args.model == "adamic_ogb":
            all_edges = all_edges.t()
            A = get_A(data.adj_t, data.num_nodes)
            pred, edge = eval('AA')(A, all_edges.cpu())
            all_scores = torch.cat((edge, pred.unsqueeze(0)), 0).T
        else:
            print("ENTERING RESOURCE ALLOCATION ", flush=True)
            assert args.model == "resource_allocation"
            train_edges_raw = np.array(split_edge['train']['edge'].cpu())
            train_edges_reverse = np.array(
                [train_edges_raw[:, 1], train_edges_raw[:, 0]]).transpose()
            train_edges = np.concatenate(
                [train_edges_raw, train_edges_reverse], axis=0)
            edge_weight = torch.ones(train_edges.shape[0], dtype=int)
            A = ssp.csr_matrix(
                (edge_weight, (train_edges[:, 0], train_edges[:, 1])), shape=(
                    data.num_nodes, data.num_nodes)
            )
            print("FINSIHED ADJ MATRX AND NOW PREDICTING IN RESOURCE ALLOCATION ", flush=True)
            pred = resource_allocation(A, all_edges.cpu(), batch_size=1024*8)
            all_scores = torch.cat((all_edges.t(), pred.unsqueeze(0)), 0).T
    
    _, indices = all_scores[:,2].sort(descending=True)
    sorted_edges = all_scores[indices].cpu()
    
    #print(sorted_edges)
    filename = f'filtered_edges/{spec}_{sorted_edge_path}_0_{args.run}_sorted_edges.pt' #using our models so num_sorted_edge not tracked, before args.run
    torch.save(sorted_edges, filename)
    print("Saving to ", filename)
    
if __name__ == "__main__":
    main()
