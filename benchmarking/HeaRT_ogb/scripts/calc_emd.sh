#!/bin/bash 

# All relevant commands for calculating Earth Mover's Distance on the dataset samples
# NOTE: These commands will need to be run in the benchmarking/HeaRT_ogb folder

python calc_emd.py --use_heuristic PA --dataset ogbl-ppa

python calc_emd.py --use_heuristic SP --dataset ogbl-collab, ogbl-ppa

python calc_emd.py --use_heuristic PA --device 3 --dataset ogbl-collab

python calc_emd.py --use_heuristic PA --device 3 --dataset ogbl-ppa

python calc_emd.py --use_heuristic PA --device 3 --dataset ogbl-ppa --edge_drop 0.3

python calc_emd.py --use_heuristic PA --device 3 --dataset ogbl-collab --edge_drop 0.3

python calc_emd.py --use_heuristic PA --device 3 --dataset ogbl-collab --eps --eps_model buddy

python calc_emd.py --use_heuristic PA --device 3 --dataset ogbl-ppa --eps

python calc_emd.py --use_heuristic SP --device 3 --dataset ogbl-ppa --eps

python calc_emd.py --use_heuristic CN --device 3 --dataset ogbl-ppa --eps

python calc_emd.py --use_heuristic PA --device 3 --dataset ogbl-collab --eps

