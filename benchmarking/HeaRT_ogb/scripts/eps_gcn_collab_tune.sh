#!/bin/bash 

python rank.py --dataset=ogbl-collab_CN_0_1_2_seed1 --lr=0.001 --feature_dropout=0.1 --label_dropout=0.1 --dropout=0.1 --device=6 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN012seed1gcn_collabCN012seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddylr001drp1_collabCN012output.txt

python rank.py --dataset=ogbl-collab_CN_0_1_2_seed1 --lr=0.001 --feature_dropout=0.3 --label_dropout=0.3 --dropout=0.3 --device=6 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN012seed1gcn_collabCN012seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddylr001drp3_collabCN012output.txt

python rank.py --dataset=ogbl-collab_CN_0_1_2_seed1 --lr=0.01 --feature_dropout=0.3 --label_dropout=0.3 --dropout=0.3 --device=6 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN012seed1gcn_collabCN012seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddylr01drp3_collabCN012output.txt
