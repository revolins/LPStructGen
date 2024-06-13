#!/bin/bash 
 
dataset=$1 
device=$2 
runs=$3 
 
epochs=10 
bs=16384 
test_bs=4096
root_dir='/egr/research-dselab/revolins/OOD/LP_OOD/benchmarking/HeaRT_ogb/dataset/'  
 
for lr in 0.001 0.01; do 
   for drop in 0.1 0.3; do 
       python main_ncn.py --dataset $dataset --device $device --predictor incn1cn1 --runs $runs --eval_steps 5 --xdp $drop --tdp $drop --pt 0.1 --gnnedp 0.25 --preedp 0.0 --gnnlr  $lr  --prelr $lr  --predp $drop --gnndp $drop --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size $bs  --ln --lnnn --save --model gcn  --testbs $test_bs --depth 2 --splitsize 131072 --maskinput --res --use_xlin --tailact > output/NCNC_${dataset}_drp${drop}_lr${lr}_output.txt 
   done 
done