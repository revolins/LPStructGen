import sys
sys.path.append("..") 
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from baseline_models.NCN.model import predictor_dict, convdict, GCN, DropEdge
from functools import partial
from sklearn.metrics import roc_auc_score, average_precision_score
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils import train_test_split_edges, negative_sampling, to_undirected
from torch_geometric.datasets import Planetoid
from synth_dataset import SynthDataset
from torch.utils.tensorboard import SummaryWriter
from baseline_models.NCN.util import PermIterator
from utils import init_seed, Logger, save_emb, get_logger, get_config_dir, get_root_dir
from tqdm import tqdm
import time
from typing import Iterable
import random
import os
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc

log_print = get_logger('testrun', 'log', get_config_dir())

# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)
    
def set_seed(seed=2020):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

def randomsplit(dataset, data_name, dir_path, filename, val_ratio=0.1, test_ratio=0.2):

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    train_pos, valid_pos, test_pos = [], [], []
    train_neg, valid_neg, test_neg = [], [], []
    node_set = set()
   
    data_name = data_name.lower()
    for split in ['train', 'test', 'valid']:

        path = dir_path+ '/{}/{}_pos.txt'.format(data_name, split)
       
        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            
            node_set.add(sub)
            node_set.add(obj)
            
            if sub == obj: continue
            if split == 'train': train_pos.append((sub, obj))
            if split == 'valid': valid_pos.append((sub, obj))  
            if split == 'test': test_pos.append((sub, obj))
    
    num_nodes = len(node_set)
    print('the number of nodes in ' + data_name + ' is: ', num_nodes)

    with open(f'{dir_path}/{data_name}/heart_valid_{filename}', "rb") as f:
        valid_neg = np.load(f)
        valid_neg = torch.from_numpy(valid_neg)
    with open(f'{dir_path}/{data_name}/heart_test_{filename}', "rb") as f:
        test_neg = np.load(f)
        test_neg = torch.from_numpy(test_neg)

    train_pos_tensor = torch.tensor(train_pos)

    valid_pos = torch.tensor(valid_pos)

    test_pos =  torch.tensor(test_pos)

    idx = torch.randperm(train_pos_tensor.size(0))
    idx = idx[:valid_pos.size(0)]
    train_val = train_pos_tensor[idx]

    split_edge['train']['edge'] = train_pos_tensor
    # data['train_val'] = train_val

    split_edge['valid']['edge']= valid_pos
    split_edge['valid']['edge_neg'] = valid_neg
    split_edge['test']['edge']  = test_pos
    split_edge['test']['edge_neg']  = test_neg

    return split_edge

def loaddataset(name, use_valedges_as_input,  dir_path, filename, args, load=None):

    if 'pa' in name.lower() or 'sp' in name.lower() or 'cn' in name.lower():
        data = SynthDataset(dataset_name=name).get()
        split_edge = SynthDataset(dataset_name=name).get_edge_split()

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
            
        while split_edge['test']['edge'].size(0) <= args.testbs:
            print("Positive Testing Edges are smaller than batch, reducing batch size")
            args.testbs = args.testbs // 2
            if args.testbs <= 0:
                raise Exception("Batch Size reached 0 in Pos. Testing Edges")

        #name = name.replace('-', '_')
        with open(f'dataset/{name}Dataset/heart_valid_{args.filename}', "rb") as f:
            neg_valid_edge = np.load(f)
            neg_valid_edge = torch.from_numpy(neg_valid_edge)
        with open(f'dataset/{name}Dataset/heart_test_{args.filename}', "rb") as f:
            neg_test_edge = np.load(f)
            neg_test_edge = torch.from_numpy(neg_test_edge)
        
        while neg_valid_edge.size(0) <= args.batch_size: 
            print("Negative Validation Edges are smaller than batch, reducing batch size")
            args.batch_size = args.batch_size // 2
            if args.batch_size == 0:
                raise Exception("Batch Size Reached 0 in Neg. Val. Edges")
            
        while neg_test_edge.size(0) <= args.testbs:
            print("Negative Testing Edges are smaller than batch, reducing batch size")
            args.testbs = args.testbs // 2
            if args.testbs == 0:
                raise Exception("Batch Size Reached 0 in Neg. Test Edges")
            
        split_edge['valid']['edge_neg'] = neg_valid_edge
        split_edge['test']['edge_neg'] = neg_test_edge

        edge_index = data.edge_index

        print('train val val_neg test test_neg: ', split_edge['train']['edge'].size(), split_edge['valid']['edge'].size(), split_edge['valid']['edge_neg'].size(), split_edge['test']['edge'].size(), split_edge['test']['edge_neg'].size())
        edge_index = data.edge_index
    data.edge_weight = None 
    print(data.num_nodes, edge_index.max())
    data.adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce().bool().float() # Clamp edge_weights
    data.adj_t = data.adj_t.to_symmetric() # Enforce Symmetry

    data.max_x = -1
    if "ppa" in name.lower():
        data.x = torch.argmax(data.x, dim=-1)
        data.max_x = torch.max(data.x).item()
    if "ppa" not in name.lower() and "collab" not in name.lower():
        print("Loading Embeddings", flush=True)
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)
        data.x = torch.nn.Embedding(data.num_nodes, args.hiddim).to(device).weight
  
    if load is not None:
        data.x = torch.load(load, map_location="cpu")
        data.max_x = -1

    print("dataset split ")
    for key1 in split_edge:
        for key2  in split_edge[key1]:
            print(key1, key2, split_edge[key1][key2].shape[0])

    # Use training + validation edges for inference on test set.
    if use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
        data.full_adj_t = data.full_adj_t.to_symmetric()
    else:
        data.full_adj_t = data.adj_t
    return data, split_edge

def get_metric_score( evaluator_mrr, pos_train_pred, neg_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    
    k_list = [10, 20]
    result = {}

    result_mrr_train = evaluate_mrr( evaluator_mrr,  pos_train_pred, neg_train_pred)
    result_mrr_val = evaluate_mrr( evaluator_mrr, pos_val_pred, neg_val_pred )
    result_mrr_test = evaluate_mrr( evaluator_mrr, pos_test_pred, neg_test_pred )
    
    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    for K in k_list:
        result[f'Hits@{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

    return result

def train(model,
          predictor,
          data,
          split_edge,
          optimizer,
          batch_size,
          maskinput: bool = True,
          cnprobs: Iterable[float]=[],
          alpha: float=None):
    def penalty(posout, negout):
        scale = torch.ones_like(posout[[0]]).requires_grad_()
        loss = -F.logsigmoid(posout*scale).mean()-F.logsigmoid(-negout*scale).mean()
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(torch.square(grad))
    
    if alpha is not None:
        predictor.setalpha(alpha)
    
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    #print("pos_train_edge size before transpose: ", pos_train_edge.size())
    pos_train_edge = pos_train_edge.t()

    total_loss = []
    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)
    
    negedge = negative_sampling(data.edge_index.to(pos_train_edge.device), data.adj_t.sizes()[0])
    for perm in PermIterator(
            adjmask.device, adjmask.shape[0], batch_size
    ):
        optimizer.zero_grad()
        if maskinput:
            adjmask[perm] = 0
            tei = pos_train_edge[:, adjmask]
            adj = SparseTensor.from_edge_index(tei,
                               sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(
                                   pos_train_edge.device, non_blocking=True)
            adjmask[perm] = 1
            adj = adj.to_symmetric()
        else:
            adj = data.adj_t
        h = model(data.x, adj)
        edge = pos_train_edge[:, perm]
        pos_outs = predictor.multidomainforward(h,
                                                    adj,
                                                    edge,
                                                    cndropprobs=cnprobs)

        pos_losss = -F.logsigmoid(pos_outs).mean()
        
        edge = negedge[:, perm]
        neg_outs = predictor.multidomainforward(h, adj, edge, cndropprobs=cnprobs)
        neg_losss = -F.logsigmoid(-neg_outs).mean()
        
        loss = neg_losss + pos_losss
        loss.backward()
        optimizer.step()

        total_loss.append(loss)
    total_loss = np.average([_.item() for _ in total_loss])
    return total_loss

def evaluate_auc(val_pred, val_true):
    valid_auc = roc_auc_score(val_true, val_pred)
    # test_auc = roc_auc_score(test_true, test_pred)
    results = {}
    
    valid_auc = round(valid_auc, 4)
    # test_auc = round(test_auc, 4)

    results['AUC'] = valid_auc

    valid_ap = average_precision_score(val_true, val_pred)
    # test_ap = average_precision_score(test_true, test_pred)
    
    valid_ap = round(valid_ap, 4)
    # test_ap = round(test_ap, 4)
    
    results['AP'] = valid_ap

    return results

@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator_mrr, batch_size,
         use_valedges_as_input):
    model.eval()
    predictor.eval()

    #pos_train_edge = split_edge['train']['edge'].to(data.adj_t.device())
    pos_valid_edge = split_edge['valid']['edge'].to(data.adj_t.device())
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.adj_t.device())
    pos_test_edge = split_edge['test']['edge'].to(data.adj_t.device())
    neg_test_edge = split_edge['test']['edge_neg'].to(data.adj_t.device())

    if pos_valid_edge.size(0) < split_edge['train']['edge'].size(0):
        idx = torch.randperm(split_edge['train']['edge'].size(0))[:pos_valid_edge.size(0)] # For predictions, train shouldn't exceed valid
        split_edge['train']['edge'] = split_edge['train']['edge'][idx]
        print("*************** Permute by Valid *****************", flush=True)
        idx = torch.randperm(neg_valid_edge.size(0)) # Randomly permute validation edges to make negative training edges
        neg_train_edge = neg_valid_edge[idx]
    else:
        print("*************** Slice by Train *****************", flush=True)
        idx = torch.randperm(neg_valid_edge.size(0))[:split_edge['train']['edge'].size(0)]
        assert idx.size(0) == split_edge['train']['edge'].size(0), f"Randomly-permuted negative train edge index: {idx.size()}, is not equal to positive training: {pos_train_edge.size()}, incorrect access time likely"
        neg_train_edge = neg_valid_edge[idx]
    
    #assert train_val_edge.size() == pos_train_edge.size(), "Positive Training Edges not the same size as Edge used for splitting"
    neg_train_edge = neg_train_edge[:, 0:1, :]
    neg_train_edge = neg_train_edge.to(data.adj_t.device())
    pos_train_edge = split_edge['train']['edge'].to(data.adj_t.device())
    print("neg_valid_edge.size(): ", neg_valid_edge.size())
    print("neg_test_edge.size(): ", neg_test_edge.size())
    print("neg_train_edge.size(): ", neg_train_edge.size())

   
    adj = data.adj_t
    h = model(data.x, adj)
    neg_num = neg_test_edge.size(1)
    print("neg_num: ", neg_num, flush=True)
    perm_count = 0 

    '''pos_train_pred = torch.cat([
        predictor(h, adj, train_valid_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(train_valid_edge.device,
                                 train_valid_edge.shape[0], batch_size, False)
    ],
                               dim=0)''' #perm crash and burn, need to update logic to handle empty perm
    pos_preds = []
    neg_preds = []
    for perm in PermIterator(pos_train_edge.device,
                                 pos_train_edge.shape[0], batch_size, False):
        if len(perm) == 0:
            perm_count += 1
            continue

        pos_preds += [predictor(h, adj, pos_train_edge[perm].t()).squeeze().cpu()]
        
        neg_edges = torch.permute(neg_train_edge[perm], (2, 0, 1))
        neg_edges = neg_edges.view(2,-1)
        neg_preds += [predictor(h, adj, neg_edges).squeeze().cpu()]

    pos_train_pred = torch.cat(pos_preds, dim=0)
    neg_train_pred = torch.cat(neg_preds, dim=0)


    pos_preds = []
    neg_preds = []
    for perm in PermIterator(pos_valid_edge.device,
                                 pos_valid_edge.shape[0], batch_size, False):
        if len(perm) == 0:
            perm_count += 1
            continue
        
        pos_preds += [predictor(h, adj, pos_valid_edge[perm].t()).squeeze().cpu()]
        
        neg_edges = torch.permute(neg_valid_edge[perm], (2, 0, 1))
        neg_edges = neg_edges.view(2,-1)
        neg_preds += [predictor(h, adj, neg_edges).squeeze().cpu()]
    
    pos_valid_pred = torch.cat(pos_preds, dim=0)
    neg_valid_pred = torch.cat(neg_preds, dim=0)
    
    if use_valedges_as_input:
        adj = data.full_adj_t
        h = model(data.x, adj)

    pos_preds = []
    neg_preds = []
    for perm in PermIterator(pos_test_edge.device, pos_test_edge.shape[0],
                                 batch_size, False):
        if len(perm) == 0:
            perm_count += 1
            continue

        pos_preds += [predictor(h, adj, pos_test_edge[perm].t()).squeeze().cpu()] 
        
        neg_edges = torch.permute(neg_test_edge[perm], (2, 0, 1))
        neg_edges = neg_edges.view(2,-1)
       
        neg_preds += [predictor(h, adj, neg_edges).squeeze().cpu()] 
    
    pos_test_pred = torch.cat(pos_preds, dim=0)
    neg_test_pred = torch.cat(neg_preds, dim=0)
    
    neg_valid_pred = neg_valid_pred.view(-1, neg_num)
    neg_test_pred = neg_test_pred.view(-1, neg_num)
    #print("NEG NUM TRAIN PRED SIZE: ", neg_train_pred.view(-1, neg_num).size())
    print("NEG TRAIN PRED SIZE: ", neg_train_pred.view(-1, 1).size(), flush=True)
    neg_train_pred = neg_train_pred.view(-1, 1) # DEBUG -- when neg_num = 1
    
    pos_valid_pred = torch.flatten(pos_valid_pred)
    pos_test_pred = torch.flatten(pos_test_pred)
    #pos_train_pred = pos_valid_pred
    pos_train_pred = torch.flatten(pos_train_pred)

    print('train_pos train_neg valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), neg_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size(), flush=True)
    
    result = get_metric_score( evaluator_mrr, pos_train_pred, neg_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)

    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), h.cpu()]

    print("perm_count: ", perm_count)

    return result, h.cpu(), score_emb

def parseargs():
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (GNN)')
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--mplayers', type=int, default=1)
    parser.add_argument('--nnlayers', type=int, default=3)
    parser.add_argument('--ln', action="store_true")
    parser.add_argument('--lnnn', action="store_true")
    parser.add_argument('--res', action="store_true")
    parser.add_argument('--jk', action="store_true")
    parser.add_argument('--maskinput', action="store_true")
    parser.add_argument('--hiddim', type=int, default=32)
    parser.add_argument('--gnndp', type=float, default=0.3)
    parser.add_argument('--xdp', type=float, default=0.3)
    parser.add_argument('--tdp', type=float, default=0.3)
    parser.add_argument('--gnnedp', type=float, default=0.3)
    parser.add_argument('--predp', type=float, default=0.3)
    parser.add_argument('--preedp', type=float, default=0.3)
    parser.add_argument('--splitsize', type=int, default=-1)
    parser.add_argument('--gnnlr', type=float, default=0.0003)
    parser.add_argument('--prelr', type=float, default=0.0003)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--testbs', type=int, default=8192)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--probscale', type=float, default=5)
    parser.add_argument('--proboffset', type=float, default=3)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--trndeg', type=int, default=-1)
    parser.add_argument('--tstdeg', type=int, default=-1)
    parser.add_argument('--dataset', type=str, default="collab")
    parser.add_argument('--predictor', choices=predictor_dict.keys())
    parser.add_argument('--model', choices=convdict.keys())
    parser.add_argument('--cndeg', type=int, default=-1)
    parser.add_argument('--save_gemb', action="store_true")
    parser.add_argument('--load', type=str)
    parser.add_argument('--cnprob', type=float, default=0)
    parser.add_argument('--pt', type=float, default=0.5)
    parser.add_argument("--learnpt", action="store_true")
    parser.add_argument("--use_xlin", action="store_true")
    parser.add_argument("--tailact", action="store_true")
    parser.add_argument("--twolayerlin", action="store_true")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--increasealpha", action="store_true")
    parser.add_argument("--savex", action="store_true")
    parser.add_argument("--loadx", action="store_true")
    parser.add_argument("--loadmod", action="store_true")
    parser.add_argument("--savemod", action="store_true")

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=30,    type=int,       help='early stopping')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--input_dir', type=str, default=os.path.join(get_root_dir(), "dataset"))
    parser.add_argument('--filename', type=str, default='samples.npy')
    parser.add_argument('--eval_mrr_data_name', type=str, default='ogbl-citation2')
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    
    args = parser.parse_args()
    return args

def main():
    args = parseargs()
   
    print(args, flush=True)
    hpstr = str(args).replace(" ", "").replace("Namespace(", "").replace(
        ")", "").replace("True", "1").replace("False", "0").replace("=", "").replace("epochs", "").replace("runs", "").replace("save_gemb", "")
    # writer = SummaryWriter(f"./rec/{args.model}_{args.predictor}")
    # writer.add_text("hyperparams", hpstr)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    predfn = predictor_dict[args.predictor]

    if args.predictor != "cn0":
        predfn = partial(predfn, cndeg=args.cndeg)
    if args.predictor in ["cn1", "incn1cn1", "scn1", "catscn1", "sincn1cn1"]:
        predfn = partial(predfn, use_xlin=args.use_xlin, tailact=args.tailact, twolayerlin=args.twolayerlin, beta=args.beta)
    if args.predictor == "incn1cn1":
        predfn = partial(predfn, depth=args.depth, splitsize=args.splitsize, scale=args.probscale, offset=args.proboffset, trainresdeg=args.trndeg, testresdeg=args.tstdeg, pt=args.pt, learnablept=args.learnpt, alpha=args.alpha)
    ret = []
    ret_auc = []
    eval_metric = 'MRR'

    evaluator_mrr = Evaluator(name=args.eval_mrr_data_name)
    loggers = {
        'Hits@10': Logger(args.runs),
        'Hits@20': Logger(args.runs),
        'MRR': Logger(args.runs)
    }

    for run in range(0, args.runs):
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run + 1
        print("###########")
        print('seed: ', seed, flush=True)
        print("###########")
        set_seed(seed)

        if 'cn' in args.dataset.lower() or 'pa' in args.dataset.lower() or 'sp' in args.dataset.lower():
            if 'ogbl' in args.dataset.lower():
                dataset_name = args.dataset + '_seed1'
            else:
                dataset_name = args.dataset + f'_seed{seed}'
            data, split_edge = loaddataset(dataset_name, args.use_valedges_as_input, args.input_dir, args.filename, args,args.load)
            data = data.to(device)
        print("dataset_name: ", dataset_name, flush=True)
        
        save_path = args.output_dir+'/lr'+str(args.gnnlr) + '_drop' + str(args.gnndp) +'_l2'+str(args.l2) +  '_numlayer' + str(args.mplayers)+ '_Prednum'+str(args.nnlayers) +'_dim'+str(args.hiddim) + '_'+ 'best_run_'+str(seed) + '_'+ str(args.dataset) + '_edgedrop' + str(args.gnnedp) +'_NCNC_' 
        
        bestscore = None
        model = GCN(data.num_features, args.hiddim, args.hiddim, args.mplayers,
                    args.gnndp, args.ln, args.res, data.max_x,
                    args.model, args.jk, args.gnnedp,  xdropout=args.xdp, taildropout=args.tdp, noinputlin=args.loadx).to(device)
        if args.loadx:
            with torch.no_grad():
                model.xemb[0].weight.copy_(torch.load(f"gemb/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pt", map_location="cpu"))
            model.xemb[0].weight.requires_grad_(False)
        predictor = predfn(args.hiddim, args.hiddim, 1, args.nnlayers,
                           args.predp, args.preedp, args.lnnn).to(device)
        if args.loadmod:
            keys = model.load_state_dict(torch.load(f"gmodel/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pt", map_location="cpu"), strict=False)
            print("unmatched params", keys, flush=True)
            keys = predictor.load_state_dict(torch.load(f"gmodel/{args.dataset}_{args.model}_cn1_{args.hiddim}_{run}.pre.pt", map_location="cpu"), strict=False)
            print("unmatched params", keys, flush=True)
        
        optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": args.gnnlr}, 
           {'params': predictor.parameters(), 'lr': args.prelr}], weight_decay=args.l2)
        
        kill_cnt = 0
        best_valid = 0

        for epoch in tqdm(range(1, 1 + args.epochs)):
            alpha = max(0, min((epoch-5)*0.1, 1)) if args.increasealpha else None
            t1 = time.time()
            loss = train(model, predictor, data, split_edge, optimizer,
                         args.batch_size, args.maskinput, [], alpha)
            print(f"*************** trn time {time.time()-t1:.2f} s ****************", flush=True)
            if epoch % args.eval_steps == 0:
                t1 = time.time()
                print("***********************************")
                print("SAVING NCN MODEL BEFORE EVALUATION")
                print("***********************************", flush=True)
                torch.save(model.state_dict(), save_path+'_model.pt')
                torch.save(predictor.state_dict(), save_path+'_predictor.pt')
                results, h, score_emb = test(model, predictor, data, split_edge, evaluator_mrr,
                               args.testbs, args.use_valedges_as_input)
                print(f"**************** test time {time.time()-t1:.2f} s ****************")
                if bestscore is None:
                    bestscore = {key: list(results[key]) for key in results}

                if True:
                    for key, result in results.items():
                        loggers[key].add_result(run, result)

                        train_hits, valid_hits, test_hits = result
                        if key == eval_metric:
                            current_valid_eval = valid_hits

                        if valid_hits > bestscore[key][1]:
                            bestscore[key] = list(result)
                        
                        print(key)
                        log_print.info(f'Run: {run + 1:02d}, '
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
                    print('---', flush=True)

                if bestscore[eval_metric][1] >   best_valid:
                    kill_cnt = 0
                    best_valid =  bestscore[eval_metric][1]
                    if args.save:
                        torch.save(model.state_dict(), save_path+'_model.pt')
                        torch.save(predictor.state_dict(), save_path+'_predictor.pt')
                else:
                    kill_cnt += 1
                   
                    if kill_cnt > args.kill_cnt: 
                        print("Early Stopping!!")
                        break

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)
                
    result_all_run = {}
    for key in loggers.keys():
        print(key)
        
        _,  _, mean_list, var_list = loggers[key].print_statistics()
        result_all_run[key] = [mean_list, var_list]

if __name__ == "__main__":
   
    main()
