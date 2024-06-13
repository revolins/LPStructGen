import torch
from ogb.linkproppred import Evaluator
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling, to_undirected
from adamic_utils import get_A, AA, get_pos_neg_edges
import scipy.sparse as ssp
from scipy.sparse.linalg import inv
from scipy.sparse import eye
import numpy as np
from tqdm import tqdm

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

evaluators = {
    "collab": Evaluator(name='ogbl-collab'),
    "reddit": Evaluator(name='ogbl-collab'),
    "ddi": Evaluator(name='ogbl-ddi'),
    "ppa": Evaluator(name='ogbl-ppa'),
    "email": Evaluator(name='ogbl-ddi'),
    "twitch": Evaluator(name='ogbl-ddi'),
    "fb": Evaluator(name='ogbl-collab'),
}
hits = {
    "collab": [10,20],
    "ppa":[10, 20],
    "ogbl-collab_CN_0_1_2_seed1": [10,50,100],
}

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

def bce_loss(logits, y, num_neg=1):
    return BCEWithLogitsLoss()(logits.view(-1), y.to(torch.float))

def get_loss(loss_str):
    if loss_str == 'bce':
        loss = bce_loss
    else:
        raise NotImplementedError
    return loss

def buddy_train(model, optimizer, train_loader, args, device, emb=None):

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

    batch_processing_times = []
    loader = DataLoader(range(len(links)), args.batch_size, shuffle=True)
    for batch_count, indices in enumerate(loader):
        # do node level things
        if model.node_embedding is not None:
            emb = model.node_embedding.weight
        else:
            emb = None
        curr_links = links[indices]
        batch_emb = None if emb is None else emb[curr_links].cpu() #.to(device)

        sf_indices = sample_indices[indices]  # need the original link indices as these correspond to sf
        subgraph_features = data.subgraph_features[sf_indices].cpu() #.to(device)
        node_features = data.x[curr_links].cpu() #.to(device)
        degrees = data.degrees[curr_links].cpu() #.to(device)
        RA = None
        
        optimizer.zero_grad()
        logits = model(subgraph_features, node_features, degrees[:, 0], degrees[:, 1], RA, batch_emb)
        loss = get_loss(args.loss)(logits, labels[indices].squeeze(0).cpu()) #.to(device))
        #loss = BCEWithLogitsLoss(logits.view(-1), labels[indices].squeeze(0).to(torch.float).to(device))

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * args.batch_size
    
    return total_loss / len(train_loader.dataset)

def train(model, score_func, data, dataset_name, split_edge, emb, optimizer, batch_size, use_params, model_str, device):
    model.train()
    score_func.train()
    pos_train_edge = split_edge['train']['edge'].to(device)
    
    row, col, _ = data.adj_t.coo()
    edge_index = torch.stack([col, row], dim=0)

    if emb == None: 
        x = data.x
        emb_update = 0
    else: 
        x = emb.weight
        emb_update = 1
    
    total_loss = total_examples = 0
    running_loss = None
    alpha = 0.99
    running_acc = None
    for idx, perm in enumerate(DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True)):

        if use_params:
            optimizer.zero_grad()
        adj = data.adj_t 

        h = model(x, adj)
        pos_edge = to_undirected(pos_train_edge[perm].t(), num_nodes=data.num_nodes)

        pos_out = score_func(h[pos_edge[0]], h[pos_edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        
        if model_str in ['gcn', 'buddy']:
            neg_edge = torch.randint(0, data.num_nodes, pos_edge.size(),dtype=torch.long, device=pos_edge.device)
        else:
            neg_edge = torch.randint(0, data.num_nodes, (1,pos_edge.size(1)),dtype=torch.long, device=pos_edge.device)
            neg_edge = torch.stack([pos_edge[0],neg_edge[0]])

        neg_out = score_func(h[neg_edge[0]], h[neg_edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean() 
        
        loss = pos_loss + neg_loss
            
        if use_params:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples

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
    loader = DataLoader(range(len(links)), args.eval_batch_size,
                        shuffle=False)  # eval batch size should be the largest that fits on GPU
    if model.node_embedding is not None:
        if args.propagate_embeddings:
            emb = model.propagate_embeddings_func(data.edge_index.cpu()) #.to(device)
        else:
            emb = model.node_embedding.weight
    else:
        emb = None
    for batch_count, indices in enumerate(loader):
        curr_links = links[indices]
        batch_emb = None if emb is None else emb[curr_links].cpu() #.to(device)
        if args.use_struct_feature:
            subgraph_features = data.subgraph_features[indices].cpu() #.to(device)
        else:
            subgraph_features = torch.zeros(data.subgraph_features[indices].shape).cpu() #.to(device)
        node_features = data.x[curr_links].cpu() #.to(device)
        degrees = data.degrees[curr_links].cpu() #.to(device)
        if args.use_RA:
            RA = data.RA[indices].cpu() #.to(device)
        else:
            RA = None
        logits = model(subgraph_features, node_features, degrees[:, 0], degrees[:, 1], RA, batch_emb)
        preds.append(logits.view(-1).cpu())
        if (batch_count + 1) * args.eval_batch_size > n_samples:
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
    
    #pos_train_pred = pos_valid_pred

    print('train_pos train_neg valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), neg_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred, neg_train_pred)

    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu()]

    return result, score_emb

@torch.no_grad()
def test_edge(score_func, input_data, h, batch_size,  negative_data=None):

    pos_preds = []
    neg_preds = []

    if negative_data is not None:
        
        for perm in DataLoader(range(input_data.size(0)),  batch_size):
            #print(perm)
            pos_edges = input_data[perm].t()
            neg_edges = torch.permute(negative_data[perm], (2, 0, 1))
            #print("Positive edges in rank pos_edges: ", pos_edges)
            #print("Negative edges in rank pos_edges: ", neg_edges)
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
def test(model, score_func, data, split_edge, emb, evaluator_hit, evaluator_mrr, batch_size, device):
    model.eval()

    train_val_edge = split_edge['train']['edge']
    neg_train_edge = split_edge['train']['edge_neg']
    pos_valid_edge = split_edge['valid']['edge']
    neg_valid_edge = split_edge['valid']['edge_neg']
    pos_test_edge = split_edge['test']['edge']
    neg_test_edge = split_edge['test']['edge_neg']

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

   
    pos_train_pred, neg_train_pred = test_edge(score_func, train_val_edge, h, batch_size, negative_data=neg_train_edge)
    pos_valid_pred, neg_valid_pred = test_edge(score_func, pos_valid_edge, h, batch_size, negative_data=neg_valid_edge)
    pos_test_pred, neg_test_pred = test_edge(score_func, pos_test_edge, h, batch_size, negative_data=neg_test_edge)
    
    pos_valid_pred = torch.flatten(pos_valid_pred)
    pos_test_pred = torch.flatten(pos_test_pred)
    pos_train_pred = torch.flatten(pos_train_pred)
    #pos_train_pred = pos_valid_pred

    neg_valid_pred = neg_valid_pred.squeeze(-1)
    neg_test_pred = neg_test_pred.squeeze(-1)
    neg_train_pred = neg_train_pred.squeeze(-1)
    print("neg_train_pred size before predictions: ", neg_train_pred.size(), flush=True)
    neg_train_pred = neg_train_pred[:, 0:1] # DEBUG -- reduce torch.Size() to 1
   
    print('train_pos train_neg valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), neg_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred, neg_train_pred)
    
    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), x1.cpu(), x2.cpu()]

    return result

def resource_allocation(adj_matrix, link_list, batch_size=32768):
    '''
    cite: [Predicting missing links via local information](https://arxiv.org/pdf/0901.0553.pdf)
    :param adj_matrix: Compressed Sparse Row matrix
    :param link_list: torch tensor list of links, shape[m, 2]
    :return: RA similarity for each link
    '''
    A = adj_matrix  # e[i, j]
    w = 1 / A.sum(axis=0)
    w[np.isinf(w)] = 0
    D = A.multiply(w).tocsr()  # e[i,j] / log(d_j)

    print("link_list: ", link_list)
    print("link_list.shape: ", link_list.shape)
    link_index = link_list.t()
    link_loader = DataLoader(range(link_index.size(1)), batch_size)
    scores = []
    for idx in tqdm(link_loader):
        src, dst = link_index[0, idx], link_index[1, idx]
        batch_scores = np.array(np.sum(A[src].multiply(D[dst]), 1)).flatten()
        scores.append(batch_scores)
    scores = np.concatenate(scores, 0)

    return torch.FloatTensor(scores)

def test_resource_allocation(model, data, split_edge, evaluator, batch_size, args, device):
    print("RA: Constructing graph.")
#     train_edges_raw = np.array(split_edge['train']['edge'])
#     train_edges_reverse = np.array(
#         [train_edges_raw[:, 1], train_edges_raw[:, 0]]).transpose()
#     train_edges = np.concatenate(
#         [train_edges_raw, train_edges_reverse], axis=0)
#     edge_weight = torch.ones(train_edges.shape[0], dtype=int)
#     A = ssp.csr_matrix(
#         (edge_weight, (train_edges[:, 0], train_edges[:, 1])), shape=(
#             data.num_nodes, data.num_nodes)
#     )
    A_eval = get_A(data.adj_t, data.num_nodes)
    A = get_A(data.full_adj_t, data.num_nodes)

    # test
    print("Benchmark test.", flush=True)
    batch_size = 1024
    pos_valid_edge = split_edge['valid']['edge']
    neg_valid_edge = split_edge['valid']['edge_neg']

    pos_test_edge = split_edge['test']['edge']
    neg_test_edge = split_edge['test']['edge_neg']

    model_predictor = resource_allocation  # use model_name as function
    pos_valid_pred = model_predictor(A_eval, pos_valid_edge, batch_size=batch_size)
    neg_valid_pred = model_predictor(A_eval, neg_valid_edge, batch_size=batch_size)

    pos_test_pred = model_predictor(A, pos_test_edge)
    neg_test_pred = model_predictor(A, neg_test_edge)

    pos_train_edge = split_edge['train']['edge'].to(device)
    pos_train_pred = torch.ones(pos_train_edge.size(0))

    results = {}
    evaluator = Evaluator(name='ogbl-collab')
    for K in hits['collab']:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results

def test_katz(model, data, split_edge, evaluator, batch_size, args, device):
    assert args.model == "katz"
    A = get_A(data.full_adj_t, data.num_nodes)
    A_train = get_A(data.adj_t, data.num_nodes)
    pos_train_edge = split_edge['eval_train']['edge']
    pos_valid_edge = split_edge['valid']['edge']
    neg_valid_edge = split_edge['valid']['edge_neg']
    pos_test_edge = split_edge['test']['edge']
    neg_test_edge = split_edge['test']['edge_neg']

    beta = 0.05
    if args.dataset == "collab":
        H_train = beta * A_train
        for _ in range(2):
            H_train += beta * (A_train @ H_train)

        H = beta * A
        for _ in range(2):
            H += beta * (A @ H)
    else:
        H_train = inv(eye(data.num_nodes) - beta * A_train) - eye(data.num_nodes) 
        H = inv(eye(data.num_nodes) - beta * A) - eye(data.num_nodes) 
    
    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), 100):
        edge = pos_train_edge[perm].t()
        pos_train_preds.append(torch.tensor(H_train[np.array(edge[0]), np.array(edge[1])]).squeeze(0))
    pos_train_pred = torch.cat(pos_train_preds, dim=0)    
    
    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds.append(torch.tensor(H_train[np.array(edge[0]), np.array(edge[1])]).squeeze(0))
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds.append(torch.tensor(H_train[np.array(edge[0]), np.array(edge[1])]).squeeze(0))
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds.append(torch.tensor(H[np.array(edge[0]), np.array(edge[1])]).squeeze(0))
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds.append(torch.tensor(H[np.array(edge[0]), np.array(edge[1])]).squeeze(0))
    neg_test_pred = torch.cat(neg_test_preds, dim=0)
    
    results = {}
    for K in hits['collab']:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results

if __name__ == "__main__":
    main()