
import sys
sys.path.append("..") 

import torch
import numpy as np
import argparse
from gnn_model import *
from utils import *
from scoring import mlp_score

from torch.utils.data import DataLoader

from ogb.linkproppred import Evaluator
from evalutors import evaluate_mrr
from synth_dataset import SynthDataset
import os

dir_path = get_root_dir()
log_print = get_logger('testrun', 'log', get_config_dir())

def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, neg_train_pred):
    
    k_list = [10, 20]
    result = {}

    result_mrr_train = evaluate_mrr( evaluator_mrr,  pos_train_pred, neg_train_pred)
    result_mrr_val = evaluate_mrr( evaluator_mrr, pos_val_pred, neg_val_pred )
    result_mrr_test = evaluate_mrr( evaluator_mrr, pos_test_pred, neg_test_pred )
    
    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    for K in k_list:
        result[f'Hits@{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

    return result

def train(model, score_func, split_edge, train_pos, data, emb, optimizer, batch_size, pos_train_weight, data_name, remove_edge_aggre):
    model.train()
    score_func.train()

    # train_pos = train_pos.transpose(1, 0)
    total_loss = total_examples = 0
    # pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    
    if emb == None: 
        x = data.x
        emb_update = 0
    else: 
        x = emb.weight
        emb_update = 1

    for perm in DataLoader(range(train_pos.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()
        num_nodes = x.size(0)
        adj = data.adj_t 

        h = model(x, adj)

        edge = train_pos[perm].t()

        pos_out = score_func(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = torch.randint(0, num_nodes, edge.size(), dtype=torch.long,
                                device=h.device)
            
        neg_out = score_func(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        if emb_update == 1: torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples

@torch.no_grad()
def test_edge(score_func, input_data, h, batch_size,  negative_data=None):

    pos_preds = []
    neg_preds = []

    if negative_data is not None:
        
        for perm in DataLoader(range(input_data.size(0)),  batch_size):
            #print(perm)
            pos_edges = input_data[perm].t()
            neg_edges = torch.permute(negative_data[perm], (2, 0, 1))

            pos_scores = score_func(h[pos_edges[0]], h[pos_edges[1]]).cpu()
            neg_scores = score_func(h[neg_edges[0]], h[neg_edges[1]]).cpu()

            pos_preds += [pos_scores]
            neg_preds += [neg_scores]
        
        neg_preds = torch.cat(neg_preds, dim=0)
    else:
        neg_preds = None
        for perm  in DataLoader(range(input_data.size(0)), batch_size):
            edge = input_data[perm].t()
            pos_preds += [score_func(h[edge[0]], h[edge[1]]).cpu()]
            
    pos_preds = torch.cat(pos_preds, dim=0)

    return pos_preds, neg_preds

@torch.no_grad()
def test(model, score_func, data, evaluation_edges, emb, evaluator_hit, evaluator_mrr, batch_size):
    model.eval()
    score_func.eval()

    train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge, neg_train_edge = evaluation_edges

    if emb == None: x = data.x
    else: x = emb.weight
    
    h = model(x, data.adj_t.to(x.device))
    x1 = h
    x2 = torch.tensor(1)
 
    train_val_edge = train_val_edge.to(x.device)
    pos_valid_edge = pos_valid_edge.to(x.device) 
    neg_valid_edge = neg_valid_edge.to(x.device)
    pos_test_edge = pos_test_edge.to(x.device) 
    neg_test_edge = neg_test_edge.to(x.device)
    neg_train_edge = neg_train_edge.to(x.device)

    pos_valid_pred, neg_valid_pred = test_edge(score_func, pos_valid_edge, h, batch_size, negative_data=neg_valid_edge)
    pos_train_pred, neg_train_pred = test_edge(score_func, train_val_edge, h, batch_size, negative_data=neg_train_edge)
    pos_test_pred, neg_test_pred = test_edge(score_func, pos_test_edge, h, batch_size, negative_data=neg_test_edge)
   
    pos_valid_pred = torch.flatten(pos_valid_pred)
    pos_test_pred = torch.flatten(pos_test_pred)
    pos_train_pred = torch.flatten(pos_train_pred)
    #pos_train_pred = pos_valid_pred

    neg_valid_pred = neg_valid_pred.squeeze(-1)
    neg_test_pred = neg_test_pred.squeeze(-1)
    neg_train_pred = neg_train_pred.squeeze(-1)
    print("neg_train_pred size before predictions: ", neg_train_pred.size(), flush=True)
    # neg_train_pred = neg_train_pred[:, 0:1] # DEBUG -- reduce torch.Size() to 1
   
    print('train_pos train_neg valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), neg_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred, neg_train_pred)
    
    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), x1.cpu(), x2.cpu()]

    return result, score_emb

def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='ogbl-collab')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='GCN')
    parser.add_argument('--score_model', type=str, default='mlp_score')

    ##gnn setting
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_layers_predictor', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--gnnout_hidden_channels', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--edge_drop', type=float, default=0.0)
    parser.add_argument('--use_valedges_as_input', action='store_true', default=False)
    parser.add_argument('--remove_edge_aggre', action='store_true', default=False)

    ##gin setting
    parser.add_argument('--gin_mlp_layer', type=int, default=2)

    ######gat
    parser.add_argument('--gat_head', type=int, default=1)

    ######mf
    parser.add_argument('--cat_node_feat_mf', default=False, action='store_true')

    ##### n2v
    parser.add_argument('--cat_n2v_feat', default=False, action='store_true')

    ### train setting
    parser.add_argument('--batch_size', type=int, default=16384)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=20,    type=int,       help='early stopping')
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--input_dir', type=str, default=os.path.join(get_root_dir(), "dataset"))
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=1)
    
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--save_test', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--log_steps', type=int, default=1)

    parser.add_argument('--eval_mrr_data_name', type=str, default='ogbl-citation2')
    parser.add_argument('--test_batch_size', type=int, default=4096)
    parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()
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

    for run in range(args.runs):

        print('#################################          ', run, '          #################################')
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run + 1
        print('seed: ', seed)
        init_seed(seed)
        if 'cn' in args.data_name.lower() or 'pa' in args.data_name.lower() or 'sp' in args.data_name.lower():
            if 'ogbl' in args.data_name:
                dataset_name = args.data_name + '_seed1'
            else:
                dataset_name = args.data_name + f'_seed{seed}'
            print("################################")
            print(f'Loading Dataset: {dataset_name}')
            print("################################")
            data = SynthDataset(dataset_name=dataset_name).get()
            if 'ppa' in dataset_name: data.x = data.x.float()
            split_edge = SynthDataset(dataset_name=dataset_name).get_edge_split()
            edge_index = data.edge_index
            node_num = data.num_nodes

            while split_edge['train']['edge'].size(0) <= args.batch_size:
                print("Positive Training Edges are smaller than batch, reducing batch size")
                args.batch_size = args.batch_size // 2
                if args.batch_size <= 0:
                    raise Exception("Batch Size Reached 0 in Pos. Train Edges")
            
            while split_edge['valid']['edge'].size(0) <= args.batch_size:
                print(f"Positive Validation Edges, {split_edge['valid']['edge'].size()} are smaller than batch, reducing batch size from {args.batch_size}")
                args.batch_size = args.batch_size // 2
                print("New Batch Size: ", args.batch_size)
                if args.batch_size <= 0:
                    raise Exception("Batch Size Reached 0 in Pos. Val. Edges")
                
            while split_edge['test']['edge'].size(0) <= args.test_batch_size:
                print("Positive Testing Edges are smaller than test batch, reducing test batch size")
                args.test_batch_size = args.test_batch_size // 2
                if args.test_batch_size <= 0:
                    raise Exception("Batch Size reached 0 in Pos. Testing Edges")

            emb = None
            input_channel = data.x.size(1)

            if hasattr(data, 'edge_weight'):
                if data.edge_weight != None:
                    edge_weight = data.edge_weight.to(torch.float)
                    data.edge_weight = data.edge_weight.view(-1).to(torch.float)
                    train_edge_weight = split_edge['train']['weight'].to(device)
                    train_edge_weight = train_edge_weight.to(torch.float)
                else:
                    train_edge_weight = None
            else:
                train_edge_weight = None

            data = T.ToSparseTensor()(data) 
            data.adj_t = data.adj_t.coalesce().bool().float() # Clamp edge_weights
            data.adj_t = data.adj_t.to_symmetric() # Enforce Symmetry
            data = data.to(device)
            model = eval(args.gnn_model)(input_channel, args.hidden_channels,
                            args.hidden_channels, args.num_layers, args.dropout, args.edge_drop, mlp_layer=args.gin_mlp_layer, head=args.gat_head, node_num=node_num, cat_node_feat_mf=args.cat_node_feat_mf,  data_name=args.data_name).to(device)

            score_func = eval(args.score_model)(args.hidden_channels, args.hidden_channels,
                            1, args.num_layers_predictor, args.dropout).to(device)
            
            pos_train_edge = split_edge['train']['edge']

            pos_valid_edge = split_edge['valid']['edge']
            
            pos_test_edge = split_edge['test']['edge']
            
            with open(f'dataset/{dataset_name}Dataset/heart_valid_samples.npy', "rb") as f:
                neg_valid_edge = np.load(f)
                neg_valid_edge = torch.from_numpy(neg_valid_edge)
            with open(f'dataset/{dataset_name}Dataset/heart_test_samples.npy', "rb") as f:
                neg_test_edge = np.load(f)
                neg_test_edge = torch.from_numpy(neg_test_edge)
                
            while neg_valid_edge.size(0) <= args.batch_size: 
                print("Negative Validation Edges are smaller than batch, reducing batch size")
                args.batch_size = args.batch_size // 2
                if args.batch_size == 0:
                    raise Exception("Batch Size Reached 0 in Neg. Val. Edges")
            
            while neg_test_edge.size(0) <= args.test_batch_size:
                print("Negative Testing Edges are smaller than testing batch, reducing testing batch size")
                args.test_batch_size = args.test_batch_size // 2
                if args.test_batch_size == 0:
                    raise Exception("Batch Size Reached 0 in Neg. Test Edges")

            if pos_valid_edge.size(0) < pos_train_edge.size(0):
                idx = torch.randperm(pos_train_edge.size(0))[:pos_valid_edge.size(0)] # For predictions, train shouldn't exceed valid
                pos_train_edge = pos_train_edge[idx]
                print("*************** Permute by Valid *****************")
                idx = torch.randperm(neg_valid_edge.size(0)) # Randomly permute validation edges to make negative training edges
                neg_train_edge = neg_valid_edge[idx]
            else:
                print("*************** Slice by Train *****************")
                idx = torch.randperm(neg_valid_edge.size(0))[:pos_train_edge.size(0)]
                assert idx.size(0) == pos_train_edge.size(0), f"Randomly-permuted negative train edge index: {idx.size()}, is not equal to positive training: {pos_train_edge.size()}, incorrect access time likely"
                neg_train_edge = neg_valid_edge[idx]

            evaluation_edges = [pos_train_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge, neg_train_edge]
            print('train train_neg val val_neg test test_neg: ', pos_train_edge.size(), neg_train_edge.size(), pos_valid_edge.size(), neg_valid_edge.size(), pos_test_edge.size(), neg_test_edge.size(),  flush=True)
        
        save_path = args.output_dir +  '/lr'+str(args.lr) + '_drop' + str(args.dropout) + '_l2'+ str(args.l2) + '_numlayer' + str(args.num_layers)+ '_numPredlay' + str(args.num_layers_predictor) + '_numGinMlplayer' + str(args.gin_mlp_layer)+'_dim'+str(args.hidden_channels)+ '_edrop'+ str(args.edge_drop)+ '_'+ 'best_run_'+str(seed)+ '_' + str(args.data_name)+ '_GCN_'

        save_path_test = args.output_dir +  '/lr'+str(args.lr) + '_drop' + str(args.dropout) + '_l2'+ str(args.l2) + '_numlayer' + str(args.num_layers)+ '_numPredlay' + str(args.num_layers_predictor) + '_numGinMlplayer' + str(args.gin_mlp_layer)+'_dim'+str(args.hidden_channels) + '_'+ 'best_run_'+str(seed)+ '_' + str(args.data_name)+ '_GCN_'

        if args.use_saved_model:
            model.load_state_dict(torch.load(save_path_test+'_model.pt'))
            score_func.load_state_dict(torch.load(save_path_test+'_predictor.pt'))
        if emb != None:
            torch.nn.init.xavier_uniform_(emb.weight)

        model.reset_parameters()
        score_func.reset_parameters()

        if emb != None:
            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(score_func.parameters()) + list(emb.parameters() ),lr=args.lr, weight_decay=args.l2)
        else:
            optimizer = torch.optim.Adam(
                    list(model.parameters()) + list(score_func.parameters()),lr=args.lr, weight_decay=args.l2)

        best_valid = 0
        kill_cnt = 0
        best_test = 0
        
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, score_func, split_edge, pos_train_edge, data, emb, optimizer, args.batch_size, train_edge_weight, args.data_name, args.remove_edge_aggre)
           
            if epoch % args.eval_steps == 0:
                results_rank, score_emb= test(model, score_func, data, evaluation_edges, emb, evaluator_hit, evaluator_mrr, args.test_batch_size)

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
                        torch.save(score_func.state_dict(), save_path+'_predictor.pt')
                        #
                        # 
                    if args.save_test:
                        torch.save(emb,save_path_test+'_emb.pt')
                        torch.save(score_emb, save_path_test+'_scemb.pt')
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


    
