#!/bin/bash 
 
dataset=$1 
device=$2 
runs=$3 
 
epochs=15 
bs=4096
test_bs=2048
root_dir='/egr/research-dselab/revolins/OOD/LP_OOD/benchmarking/HeaRT_ogb/dataset/'  
 
for lr in 1e-2; do 
   for drop in 0.1 0.3; do 
       python run.py --data_name $dataset --lr $lr --gnn-layers 3 --dim 128 --batch-size $bs --epochs $epochs --all-drop $drop --eval_steps 5 --runs $runs --device $device --root-dir $root_dir --test-batch-size $test_bs --save-as lpformer
   done 
done
