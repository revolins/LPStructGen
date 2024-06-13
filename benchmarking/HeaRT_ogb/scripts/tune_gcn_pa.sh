#!/bin/bash 
 
dataset=$1 
device=$2 
runs=$3 
 
epochs=1000 
bs=65536 
test_bs=65536
root_dir='/egr/research-dselab/revolins/OOD/LP_OOD/benchmarking/HeaRT_ogb/dataset/'  
 
 #0.01, 0.1
for lr in 0.001; do 
   for drop in 0.3; do 
       python main_gnn.py --lr $lr --dropout $drop --test_batch_size $test_bs --device $device --data_name $dataset --gnn_model GCN --runs $runs --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size $bs --save --output_dir models > output/GCN_${dataset}_drp${drop}_lr${lr}_output.txt 
   done 
done