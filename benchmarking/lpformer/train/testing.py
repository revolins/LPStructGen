import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from util.utils import *

import sys
sys.path.append("../..")

from benchmarking.evalutors import evaluate_mrr


def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, neg_train_pred):
    
    k_list = [10, 20]
    result = {}

    result_mrr_train = evaluate_mrr(evaluator_mrr, pos_train_pred, neg_train_pred)
    result_mrr_val = evaluate_mrr(evaluator_mrr, pos_val_pred, neg_val_pred)
    result_mrr_test = evaluate_mrr(evaluator_mrr, pos_test_pred, neg_test_pred)
    
    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    for K in k_list:
        result[f'Hits@{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

    return result



@torch.no_grad()
def test_edge(model, score_func, input_data, batch_size, test_set=False):
    """
    Evaluate performance on val/test set
    """
    preds = []

    # for perm in tqdm(DataLoader(range(input_data.size(0)), batch_size), "Evaluating"):
    for perm in DataLoader(range(input_data.size(0)), batch_size):
        edge = input_data[perm].t()
        h = model(edge, test_set=test_set)
        preds += [score_func(h).cpu()]

    pred_all = torch.cat(preds, dim=0)

    return pred_all


@torch.no_grad()
def test_heart_negatives(negative_data, model, score_func, batch_size=32768, test_set=False):
    """
    For HeaRT setting
    """    
    neg_preds = []
    num_negative = negative_data.size(1)
    negative_data = torch.permute(negative_data, (2, 0, 1)).reshape(2, -1).t()

    # TODO: Move to parent function so only run once
    h = model.propagate()

    qqq = DataLoader(range(negative_data.size(0)),  batch_size)
    #qqq = tqdm(qqq, "Testing Negatives")

    for perm in qqq:
        neg_edges = negative_data[perm].t().to(h.device)

        elementwise_feats = model.elementwise_lin(h[neg_edges[0]] * h[neg_edges[1]])
        pairwise_feats, _ = model.calc_pairwise(neg_edges, h, test_set=test_set)
        combined_feats = torch.cat((elementwise_feats, pairwise_feats), dim=-1)

        neg_preds += [score_func(combined_feats).squeeze().cpu()]

    neg_preds = torch.cat(neg_preds, dim=0).view(-1, num_negative)

    return neg_preds


def test(
        model, 
        score_func, 
        data, 
        evaluator_hit, 
        evaluator_mrr, 
        batch_size, 
        **kwargs,
    ):
    model.eval()
    score_func.eval()

    with torch.no_grad():
        pos_train_pred = test_edge(model, score_func, data['train_pos_val'], batch_size)
        pos_valid_pred = test_edge(model, score_func, data['valid_pos'], batch_size)
        pos_test_pred = test_edge(model, score_func, data['test_pos'], batch_size, test_set=True)

        neg_valid_pred = test_heart_negatives(data['valid_neg'], model, score_func, batch_size=batch_size)
        neg_test_pred = test_heart_negatives(data['test_neg'], model, score_func, batch_size=batch_size, test_set=True)

        pos_valid_pred = pos_valid_pred.view(-1)
        pos_test_pred = pos_test_pred.view(-1)
        pos_train_pred = pos_train_pred.view(-1)
        
        result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred, neg_valid_pred)

    return result

