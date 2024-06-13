from utils import *
import torch
import os
from torch_sparse import SparseTensor
import torch_sparse
from torch.nn import BCEWithLogitsLoss
from torch_geometric.utils import softmax, negative_sampling
import torch.nn.functional as F
from dataprocess import PermIterator, update_tc
from torch.utils.data import DataLoader
from scipy.stats import rankdata
from itertools import product
from torch_scatter import segment_csr

def eval_mrr(y_pred_pos, y_pred_neg):
    '''
        compute mrr
        y_pred_neg is an array with shape (batch size, num_entities_neg).
        y_pred_pos is an array with shape (batch size, )
    '''


    # calculate ranks
    y_pred_pos = y_pred_pos.view(-1, 1)
    # optimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked first among those with equal score
    optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
    # pessimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked last among those with equal score
    pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1

    hits1_list = (ranking_list <= 1).to(torch.float)
    hits3_list = (ranking_list <= 3).to(torch.float)

    hits20_list = (ranking_list <= 20).to(torch.float)
    hits50_list = (ranking_list <= 50).to(torch.float)
    hits10_list = (ranking_list <= 10).to(torch.float)
    hits100_list = (ranking_list <= 100).to(torch.float)
    mrr_list = 1./ranking_list.to(torch.float)

    return { 'hits@1_list': hits1_list,
                'hits@3_list': hits3_list,
                'hits@20_list': hits20_list,
                'hits@50_list': hits50_list,
                'hits@10_list': hits10_list,
                'hits@100_list': hits100_list,
                'mrr_list': mrr_list}

def evaluate_mrr(evaluator, pos_val_pred, neg_val_pred):
    
    # print("**********************************************")
    # print("pos_val_pred.shape[0]: ", pos_val_pred.shape[0])
    # print("neg_val_pred.shape[0]: ", neg_val_pred.shape[0])
    # print("**********************************************")
    # exit()

    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    # neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    
    mrr_output =  eval_mrr(pos_val_pred, neg_val_pred)


    valid_mrr =mrr_output['mrr_list'].mean().item()
    valid_mrr_hit1 = mrr_output['hits@1_list'].mean().item()
    valid_mrr_hit3 = mrr_output['hits@3_list'].mean().item()
    valid_mrr_hit10 = mrr_output['hits@10_list'].mean().item()

    valid_mrr_hit20 = mrr_output['hits@20_list'].mean().item()
    valid_mrr_hit50 = mrr_output['hits@50_list'].mean().item()
    valid_mrr_hit100 = mrr_output['hits@100_list'].mean().item()


    valid_mrr = round(valid_mrr, 4)
    # test_mrr = round(test_mrr, 4)
    valid_mrr_hit1 = round(valid_mrr_hit1, 4)
    valid_mrr_hit3 = round(valid_mrr_hit3, 4)
    valid_mrr_hit10 = round(valid_mrr_hit10, 4)

    valid_mrr_hit20 = round(valid_mrr_hit20, 4)
    valid_mrr_hit50 = round(valid_mrr_hit50, 4)
    valid_mrr_hit100 = round(valid_mrr_hit100, 4)
    
    results = {}
    results['mrr_hit1'] = valid_mrr_hit1
    results['mrr_hit3'] = valid_mrr_hit3
    results['mrr_hit10'] = valid_mrr_hit10

    results['MRR'] = valid_mrr

    results['mrr_hit20'] = valid_mrr_hit20
    results['mrr_hit50'] = valid_mrr_hit50
    results['mrr_hit100'] = valid_mrr_hit100

    
    return results

def get_num_samples(sample_arg, dataset_len):
    """
    convert a sample arg that can be a number of % into a number of samples
    :param sample_arg: float interpreted as % if < 1 or count if >= 1
    :param dataset_len: the number of data points before sampling
    :return:
    """
    if sample_arg < 1:
        samples = int(sample_arg * dataset_len)
    else:
        samples = int(min(sample_arg, dataset_len))
    return samples

def bce_loss(logits, y, tc_scaling, num_neg=1):
    return F.binary_cross_entropy_with_logits(logits.view(-1), y.to(torch.float), tc_scaling)

def get_loss(loss_str):
    if loss_str == 'bce':
        loss = bce_loss
    else:
        raise NotImplementedError
    return loss

def buddy_train(model, optimizer, train_loader, args, emb):

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

    loader = DataLoader(range(len(links)), args.train_bsz, shuffle=True)
    for batch_count, indices in enumerate(loader):
        # do node level things
        emb = None
        curr_links = links[indices]
        batch_emb = None if emb is None else emb[curr_links].to(args.device)

        sf_indices = sample_indices[indices]  # need the original link indices as these correspond to sf
        subgraph_features = data.subgraph_features[sf_indices].to(args.device)
        node_features = data.x[curr_links].to(args.device)
        degrees = data.degrees[curr_links].to(args.device)
        RA = None
        
        optimizer.zero_grad()
        logits, n_emb = model(subgraph_features, node_features, degrees[:, 0], degrees[:, 1], RA, batch_emb)
        n_emb = n_emb.detach()
        loss = get_loss(args.loss)(logits, labels[indices].squeeze(0).to(args.device), torch.sigmoid(n_emb))
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * args.train_bsz
    
    return total_loss / len(train_loader.dataset)

def train(encoder, predictor, optimizer, data, train_edge, args):
    encoder.train()
    predictor.train()

    neg_edge = negative_sampling(data.train_edge_index, num_neg_samples = train_edge.shape[0]).t()

    total_loss, count = 0, 0
    for batch in PermIterator(train_edge.device, train_edge.shape[0], args.train_bsz):
        h = encoder(data.x, data.train_adj_aug)

        pos_score = predictor(h[train_edge[batch, 0]]*h[train_edge[batch, 1]])
    
        pos_loss = -F.logsigmoid(pos_score).mean()

        neg_score = predictor(h[neg_edge[batch, 0]]*h[neg_edge[batch, 1]])
        neg_loss = -F.logsigmoid(-neg_score).mean()

        loss = (pos_loss + neg_loss) / 2

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()
        total_loss += loss.item() * train_edge.shape[0]
        count += train_edge.shape[0]

    return total_loss / count

@torch.no_grad()
def test_edge(score_func, input_data, h, batch_size,  negative_data=None):

    pos_preds = []
    neg_preds = []

    if negative_data is not None:
        
        for perm in DataLoader(range(input_data.size(0)),  batch_size):
            #print(perm)
            pos_edges = input_data[perm].t()
            neg_edges = torch.permute(negative_data[perm], (2, 0, 1))

            pos_scores = score_func(h[pos_edges[0]]* h[pos_edges[1]]).to(h.device)
            neg_scores = score_func(h[neg_edges[0]]* h[neg_edges[1]]).to(h.device)

            pos_preds += [pos_scores]
            neg_preds += [neg_scores]
        
        neg_preds = torch.cat(neg_preds, dim=0)
    else:
        neg_preds = None
        for perm  in DataLoader(range(input_data.size(0)), batch_size):
            edge = input_data[perm].t()
            pos_preds += [score_func(h[edge[0]]* h[edge[1]]).to(h.device)]
            
    pos_preds = torch.cat(pos_preds, dim=0)

    return pos_preds, neg_preds

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

def get_split_samples(split, args, dataset_len):
    """
    get the
    :param split: train, val, test
    :param args: Namespace object
    :param dataset_len: total size of dataset
    :return:
    """
    samples = dataset_len
    if split == 'train':
        if args.dynamic_train:
            samples = get_num_samples(args.train_samples, dataset_len)
    elif split in {'val', 'valid'}:
        if args.dynamic_val:
            samples = get_num_samples(args.val_samples, dataset_len)
    elif split == 'test':
        if args.dynamic_test:
            samples = get_num_samples(args.test_samples, dataset_len)
    else:
        raise NotImplementedError(f'split: {split} is not a valid split')
    return samples

@torch.no_grad()
def buddy_test_edge(model, loader, device, args, split=None):

    n_samples = get_split_samples(split, args, len(loader.dataset))
    preds = []
    data = loader.dataset
    # hydrate edges
    links = data.links
    labels = torch.tensor(data.labels)
    loader = DataLoader(range(len(links)), args.eval_bsz,
                        shuffle=False)  # eval batch size should be the largest that fits on GPU
    emb = None
    for batch_count, indices in enumerate(loader):
        curr_links = links[indices]
        batch_emb = None if emb is None else emb[curr_links].to(args.device)
        if args.use_struct_feature:
            subgraph_features = data.subgraph_features[indices].to(args.device)
        else:
            subgraph_features = torch.zeros(data.subgraph_features[indices].shape).to(args.device)
        node_features = data.x[curr_links].to(args.device)
        degrees = data.degrees[curr_links].to(args.device)
        if args.use_RA:
            RA = data.RA[indices].to(args.device)
        else:
            RA = None
        logits, _ = model(subgraph_features, node_features, degrees[:, 0], degrees[:, 1], RA, batch_emb)
        preds.append(logits.view(-1).to(args.device))
        if (batch_count + 1) * args.eval_bsz > n_samples:
            break

    pred = torch.cat(preds)
    labels = labels[:len(pred)]
    pos_pred = pred[labels == 1]
    neg_pred = pred[labels == 0]
    return pos_pred, neg_pred


@torch.no_grad()
def buddy_test(model, evaluator_hit, evaluator_mrr, train_loader, val_loader, test_loader, args, device):
   
    model.eval()

    pos_train_pred, neg_train_pred = buddy_test_edge(model, train_loader, device, args, split='train')

    pos_valid_pred, neg_valid_pred = buddy_test_edge(model, val_loader, device, args, split='val')
    
    pos_test_pred, neg_test_pred  = buddy_test_edge(model, test_loader, device, args, split='test')

    pos_train_pred = torch.flatten(pos_train_pred)
    pos_valid_pred = torch.flatten(pos_valid_pred)
    pos_test_pred =  torch.flatten(pos_test_pred)
    
    neg_valid_pred = neg_valid_pred.view(pos_valid_pred.size(0), -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.size(0), -1)
    neg_train_pred = neg_train_pred.view(pos_train_pred.size(0), -1)

    print('train_pos train_neg valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), neg_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred, neg_train_pred)

    return result

@torch.no_grad()
def test(model, score_func, data, split_edge, emb, evaluator_hit, evaluator_mrr, args):
    model.eval()
    score_func.eval()

    train_val_edge = split_edge['train']['edge']
    neg_train_edge = split_edge['train']['edge_neg']
    pos_valid_edge = split_edge['valid']['edge']
    neg_valid_edge = split_edge['valid']['edge_neg']
    pos_test_edge = split_edge['test']['edge']
    neg_test_edge = split_edge['test']['edge_neg']

    if emb == None: x = data.x
    else: x = emb.weight
    
    h = model(x, data.train_adj_aug.to(x.device))
 
    train_val_edge = train_val_edge.to(x.device)
    pos_valid_edge = pos_valid_edge.to(x.device) 
    neg_valid_edge = neg_valid_edge.to(x.device)
    pos_test_edge = pos_test_edge.to(x.device) 
    neg_test_edge = neg_test_edge.to(x.device)
    neg_train_edge = neg_train_edge.to(x.device)

    pos_valid_pred, neg_valid_pred = test_edge(score_func, pos_valid_edge, h, args.eval_bsz, negative_data=neg_valid_edge)
    pos_train_pred, neg_train_pred = test_edge(score_func, train_val_edge, h, args.eval_bsz, negative_data=neg_train_edge)
    pos_test_pred, neg_test_pred = test_edge(score_func, pos_test_edge, h, args.eval_bsz, negative_data=neg_test_edge)
   
    pos_valid_pred = torch.flatten(pos_valid_pred)
    pos_test_pred = torch.flatten(pos_test_pred)
    pos_train_pred = torch.flatten(pos_train_pred)

    neg_valid_pred = neg_valid_pred.squeeze(-1)
    neg_test_pred = neg_test_pred.squeeze(-1)
    neg_train_pred = neg_train_pred.squeeze(-1)
    #print("neg_train_pred size before predictions: ", neg_train_pred.size(), flush=True)
    #neg_train_pred = neg_train_pred[:, 0:1] # DEBUG -- reduce torch.Size() to 1
   
    print('train_pos train_neg valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), neg_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred, neg_train_pred)

    return result


@torch.no_grad()
def eval(encoder, predictor, data, evaluator, split_edge, args):
    encoder.eval()
    predictor.eval()

    ress = {'train': [],
            'valid': [],
            'test': []}

    h = encoder(data.x, data.train_adj_aug)

    for key in split_edge:

        edge, neg_edge = split_edge[key]['edge'], split_edge[key]['edge_neg']

        pos_preds = torch.cat([predictor(h[edge[batch, 0]]*h[edge[batch, 1]]).squeeze().cpu() \
            for batch in PermIterator(edge.device, edge.shape[0], args.eval_bsz, training = False)])
        neg_preds = torch.cat([predictor(h[neg_edge[batch, 0]]*h[neg_edge[batch, 1]]).squeeze().cpu() \
            for batch in PermIterator(neg_edge.device, neg_edge.shape[0], args.eval_bsz, training = False)])
        
        for K in args.topks:
            evaluator.K = K
            hits = evaluator.eval({'y_pred_pos': pos_preds, 'y_pred_neg': neg_preds})[f'hits@{K}']

            ress[key].append(hits)

    return ress

@torch.no_grad()
def update_adj(encoder, predictor, data, epoch, args):
    encoder.eval()
    predictor.eval()

    h_train = encoder(data.x, data.train_adj_aug.to(data.x.device))
    h_node = data.train_adj_gcn.matmul(h_train)[data.train_edge_index[1]]

    h_pos_node = h_train[data.train_edge_index[0]]

    score = predictor.score(h_node*h_pos_node).squeeze()
    score = softmax(score, ptr = data.train_ptr, dim = 0, num_nodes = data.x.shape[0])

    norm_origin = segment_csr(data.train_edge_weight, data.train_ptr, reduce='sum')[data.train_edge_index[1]]

    weight = (score * norm_origin).detach()

    new_weight = args.alpha*weight + data.train_edge_weight

    adj_t = SparseTensor(
        row=data.train_edge_index[0], col=data.train_edge_index[1], value=new_weight, is_sorted=False)
    
    train_tc, valid_tc, test_tc = update_tc(data, new_weight)

    torch.save(train_tc, os.path.join(args.path, 'model', args.dataset, args.model, 'train_tc_update.pt'))
    torch.save(valid_tc, os.path.join(args.path, 'model', args.dataset, args.model, 'valid_tc_update.pt'))
    torch.save(test_tc, os.path.join(args.path, 'model', args.dataset, args.model, 'test_tc_update.pt'))
    
    data.train_tc = train_tc[(data.deg['train'] != 0) & (data.one_hot_dict['test'])].mean().item()
    data.valid_tc = valid_tc[(data.deg['valid'] != 0) & (data.one_hot_dict['test'])].mean().item()
    data.test_tc = test_tc[(data.deg['test'] != 0) & (data.one_hot_dict['test'])].mean().item()


    return adj_t, {'train': train_tc.cpu(), 'valid': valid_tc.cpu(), 'test': test_tc.cpu()}

@torch.no_grad()
def eval_comprehensive(encoder, predictor, data, evaluator, split_edge, adj_list_dict, eval_type, args):
    encoder.eval()
    predictor.eval()

    ress = {'train': [],
            'valid': [],
            'test': []}

    h = encoder(data.x, data.train_adj_aug)

    #Eval edge
    for key in split_edge:
        edge, neg_edge = split_edge[key]['edge'], split_edge[key]['edge_neg']

        pos_preds, neg_preds = test_edge(predictor, edge, h, args.eval_bsz, negative_data=neg_edge)

        for K in args.topks:
            evaluator.K = K
            hits = evaluator.eval({'y_pred_pos': pos_preds, 'y_pred_neg': neg_preds})[f'hits@{K}']

            ress[key].append(hits)

    #Eval node
    ratings_list = []
    groundTruth_nodes_list = []
    nodes_list = []
    mrr_list= []

    node = data.eval_node[eval_type]
    print(node.shape[0] // args.eval_node_bsz)

    for count, batch in enumerate(PermIterator(data.x.device, node.shape[0], args.eval_node_bsz, training = False)):
        print(count)
        batch_node = node[batch]

        score = (h[batch_node].unsqueeze(1) * h.unsqueeze(0)).view(batch.shape[0]*h.shape[0], -1)

        score = predictor(score).detach().cpu().squeeze().view(batch_node.shape[0], h.shape[0])

        if eval_type == 'train':
            clicked_nodes = [np.array([])]
            groundTruth_nodes = [list(adj_list_dict['train'][node.item()]) for node in batch_node]

        elif eval_type == 'valid':
            clicked_nodes = [np.array(list(adj_list_dict['train'][node.item()]), dtype = int) for node in batch_node]
            groundTruth_nodes = [list(adj_list_dict['valid'][node.item()]) for node in batch_node]

        elif eval_type == 'test':
            clicked_nodes = [np.array((list(adj_list_dict['train'][node.item()]) + list(adj_list_dict['valid'][node.item()]))) for node in batch_node]
            groundTruth_nodes = [list(adj_list_dict['test'][node.item()]) for node in batch_node]

        exclude_index, exclude_nodes = [], []
        for range_i, nodes in enumerate(clicked_nodes):
            exclude_index.extend([range_i] * len(nodes))
            exclude_nodes.extend(nodes)

        if args.dataset not in ['collab']:
            score[exclude_index, exclude_nodes] = -(1 << 10)
        rating_K = torch.topk(score, k = max(args.topks))[1]


        rating_ranking = rankdata(-np.array(score), method = 'ordinal', axis = 1)
        for i in range(len(groundTruth_nodes)):
            rank = min(rating_ranking[i][list(groundTruth_nodes[i])])
            mrr_list.append(1/rank)

        ratings_list.append(rating_K)
        groundTruth_nodes_list.append(groundTruth_nodes)
        nodes_list.append(batch_node.tolist())

    recall_list, ndcg_list, hit_ratio_list, precision_list, F1_list = [], [], [], [], []

    for nodes, X in zip(nodes_list, zip(ratings_list, groundTruth_nodes_list)):
        recalls, ndcgs, hit_ratios, precisions, F1s = test_one_batch_group(X, args.topks)

        recall_list.append(recalls)
        ndcg_list.append(ndcgs)
        hit_ratio_list.append(hit_ratios)
        precision_list.append(precisions)
        F1_list.append(F1s)

    recall_list = np.concatenate(recall_list)
    ndcg_list = np.concatenate(ndcg_list)
    hit_ratio_list = np.concatenate(hit_ratio_list)
    precision_list = np.concatenate(precision_list)
    F1_list = np.concatenate(F1_list)
    mrr_list = np.array(mrr_list)

    np.save(os.getcwd() + '/res/' + args.dataset + '/' + args.model + '/' + str(args.run) + '_recall_list_' + eval_type + '.npy', recall_list)
    np.save(os.getcwd() + '/res/' + args.dataset + '/' + args.model + '/' + str(args.run) + '_ndcg_list_' + eval_type + '.npy', ndcg_list)
    np.save(os.getcwd() + '/res/' + args.dataset + '/' + args.model + '/' + str(args.run) + '_hit_ratio_list_' + eval_type + '.npy', hit_ratio_list)
    np.save(os.getcwd() + '/res/' + args.dataset + '/' + args.model + '/' + str(args.run) + '_precision_list_' + eval_type + '.npy', precision_list)
    np.save(os.getcwd() + '/res/' + args.dataset + '/' + args.model + '/' + str(args.run) + '_F1_list_' + eval_type + '.npy', F1_list)
    np.save(os.getcwd() + '/res/' + args.dataset + '/' + args.model + '/' + str(args.run) + '_mrr_list_' + eval_type + '.npy', mrr_list)

    return ress







# @torch.no_grad()
# def update_adj_aug(encoder, predictor, data):
#     encoder.eval()
#     predictor.eval()

#     h_train = encoder(data.x, data.train_val_adj)

#     h_node = data.train_val_adj.matmul(h_train)[data.train_val_edge_index[1]]

#     h_pos_node = h_train[data.train_val_edge_index[0]]

#     score = predictor.score(h_node*h_pos_node).squeeze()
#     score = softmax(score, ptr = data.train_val_ptr, dim = 0, num_nodes = data.x.shape[0])

#     norm_origin = segment_csr(data.train_val_edge_weight, data.train_val_ptr, reduce='sum')[data.train_val_edge_index[1]]

#     weight = (score * norm_origin).detach()

#     adj_t = SparseTensor(
#         row=data.train_val_edge_index[0], col=data.train_val_edge_index[1], value=1 * weight + 1 * data.train_val_edge_weight, is_sorted=False)

#     return adj_t
