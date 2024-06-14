#!/bin/bash
root_dir='/egr/research-dselab/revolins/OOD/LP_OOD/benchmarking/HeaRT_ogb/dataset/' 

# COLLAB, CN
python run.py --data_name $dataset --lr $lr --all-drop $drop --gnn-layers 3 --dim 128 --batch-size 16384 --epochs 15 --eval_steps 5 --runs 5 --device 0 --root-dir $root_dir --test-batch-size 16384