#!/bin/bash 
 
dataset=$1 
device=$2 
runs=$3 
 
epochs=100 
bs=65536 
test_bs=65536
root_dir='/egr/research-dselab/revolins/OOD/LP_OOD/benchmarking/HeaRT_ogb/dataset/'  
 
for lr in 0.001 0.01; do 
   for drop in 0.1 0.3; do 
       python main_buddy.py --cache_subgraph_features  --dataset $dataset --device $device --runs $runs --lr $lr --l2 0  --label_dropout $drop  --feature_dropout $drop --hidden_channels 256  --epochs $epochs --eval_steps 20 --kill_cnt 40 --save  --batch_size $bs --eval_batch_size $test_bs --model BUDDY > output/BUDDY_${dataset}_drp${drop}_lr${lr}_output.txt 
   done 
done