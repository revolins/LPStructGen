import torch
import argparse

from util.utils import *
from train.train_model import train_data


def run_model(cmd_args):
    """
    Run model using args
    """
    device = torch.device(f'cuda:{cmd_args.device}' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"  # DEBUG

    args = {
        'gcn_cache': True,
        'gnn_layers': cmd_args.gnn_layers,
        'trans_layers': cmd_args.tlayers,
        'dim': cmd_args.dim,
        'num_heads': cmd_args.num_heads,
        'lr': cmd_args.lr,
        'weight_decay': cmd_args.l2,
        'decay': cmd_args.decay,
        'dropout': cmd_args.all_drop,
        'gnn_drop': cmd_args.all_drop,
        'pred_dropout': cmd_args.all_drop,
        'att_drop': cmd_args.all_drop,
        "feat_drop": cmd_args.all_drop,
        "edge_drop": cmd_args.edge_drop,
        "residual": cmd_args.residual,
        "layer_norm": not cmd_args.no_layer_norm,
        "relu": not cmd_args.no_relu
    }

    # Important shit from argparse
    fields = ['thresh_1hop', "mask_input", "count_ra",  
              "filter_1hop", "filter_cn", "thresh_cn", "thresh_non1hop", 
              "ablate_att", "ablate_pe", "ablate_feats", "ablate_ppr", 
              "ablate_counts", "ablate_ppr_type"
              ]
    for f in fields:
        args[f] = getattr(cmd_args, f)

    train_data(cmd_args, args, device, verbose = not cmd_args.non_verbose)

def find_root(current_dir, marker=".git"):
    current_dir = os.path.abspath(current_dir)
    while not os.path.exists(os.path.join(current_dir, marker)):
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError(f"Root at {marker} not found, file deleted or repository structure changed?")
        current_dir = parent_dir
    return current_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='ogbl-collab')
    parser.add_argument('--root-dir', type=str, default=find_root())
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument("--mask-input", action='store_true', default=False)
    parser.add_argument("--non-verbose", action='store_true', default=False)

    # Model Settings
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--tlayers', type=int, default=1)
    parser.add_argument('--num-heads', type=int, default=1)
    parser.add_argument('--gnn-layers', type=int, default=3)
    parser.add_argument('--pred-layers', type=int, default=2)
    parser.add_argument('--all-drop', type=float, default=0.1)
    parser.add_argument("--edge_drop", type=float, default=0.0)
    parser.add_argument("--residual", action='store_true', default=False)
    parser.add_argument("--no-layer-norm", action='store_true', default=False)
    parser.add_argument("--no-relu", action='store_true', default=False)

    # Train Settings
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--decay', type=float, default=1)
    parser.add_argument('--batch-size', type=int, default=32768)
    parser.add_argument('--test-batch-size', type=int, default=32768)
    parser.add_argument('--num-negative', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--kill_cnt', dest='kill_cnt', default=100, type=int, help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--use-val-in-test", action='store_true', default=False)
    parser.add_argument("--remove-pos-edges", action='store_true', default=False)
    
    parser.add_argument('--save-as', type=str, default=None)
    parser.add_argument('--metric', type=str, default='MRR')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=3)

    parser.add_argument('--eps', type=float, default=5e-5)
    parser.add_argument('--thresh-cn', type=float, default=0)
    parser.add_argument('--thresh-1hop', type=float, default=1e-4)
    parser.add_argument('--thresh-non1hop', type=float, default=1e-2)

    parser.add_argument("--filter-cn", action='store_true', default=False)
    parser.add_argument("--filter-1hop", action='store_true', default=False)

    # TODO: IGNORE!!!
    parser.add_argument("--count-ra", action='store_true', default=False)
    parser.add_argument("--ablate-att", action='store_true', default=False)
    parser.add_argument("--ablate-pe", action='store_true', default=False)
    parser.add_argument("--ablate-feats", action='store_true', default=False)
    parser.add_argument("--ablate-ppr", action='store_true', default=False)
    parser.add_argument("--ablate-counts", action='store_true', default=False)
    parser.add_argument("--ablate-ppr-type", action='store_true', default=False)

    args = parser.parse_args()

    init_seed(args.seed)
    run_model(args)


if __name__ == "__main__":
    main()