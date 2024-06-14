#!/bin/bash 
 
dataset=$1 
device=$2 
runs=$3 
 
epochs=15 
bs=16384
test_bs=16384
root_dir='/egr/research-dselab/revolins/OOD/LP_OOD/benchmarking/HeaRT_ogb/dataset/'  
 
for lr in 1e-3, 1e-2; do 
   for drop in 0.1 0.3; do 
       python run.py --data_name $dataset --lr $lr --all-drop $drop --gnn-layers 3 --dim 128 --batch-size 16384 --epochs 15 --eval_steps 5 --runs 5 --device 0 --root-dir $root_dir --test-batch-size 16384
   done 
done
