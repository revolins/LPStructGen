import argparse
from BUDDY.utils import str2bool
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    # dataset
    parser.add_argument("--dataset", nargs="?", default="Cora")

    # model
    parser.add_argument("--encoder", type=str, default="GCN")
    parser.add_argument("--predictor", type=str, default="MLP")
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--n_hidden", type=int, default=256)

    # training
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--train_bsz', type=int, default=64*1024)
    parser.add_argument('--eval_bsz', type=int, default=16384)
    parser.add_argument('--eval_node_bsz', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--encoder_lr', type=float, default=1e-3)
    parser.add_argument('--predictor_lr', type=float, default=1e-3)
    parser.add_argument('--en_dp', type=float, default=0.0)
    parser.add_argument('--lp_dp', type=float, default=0.0)

    # specific for LP
    parser.add_argument('--topks', default=[10, 20])

    # experiments
    parser.add_argument("--seed", type=int, default=1,
                        help="seed to run the experiment")
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument('--model', type=str, default='GCN')

    parser.add_argument("--save", action='store_true')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--aug", action='store_true')

    parser.add_argument("--warm_up", type = int, default = 50)
    parser.add_argument("--update_interval", type = int, default = 50)
    parser.add_argument('--alpha', type=float, default=1)


    parser.add_argument('--use_val', action='store_true')
    parser.add_argument('--mask_edge_in_prop', action='store_true')
    parser.add_argument('--ln', action='store_true')
    parser.add_argument('--remove_rep', action='store_true')

    parser.add_argument('--gcn_denoise', action='store_true')

    # buddy
    parser.add_argument('--edge_drop', type=float, default=0.0)
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
    parser.add_argument('--num_negs', type=int, default=250, help='number of negatives for each positive')
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
    #parser.add_argument('--eval_batch_size', type=int, default=1024*64,
    #                    help='eval batch size should be largest the GPU memory can take - the same is not necessarily true at training time')
    
    parser.add_argument('--year', type=int, default=0, help='filter training data from before this year')

    parser.add_argument('--eval_mrr_data_name', type=str, default='ogbl-citation2')
    #parser.add_argument('--test_batch_size', type=int, default=4096)
    parser.add_argument('--fnr', action='store_true', default=False)
    parser.add_argument('--tc_buddy', action='store_true', default=False)


    
    
    return parser.parse_args()
