#!/bin/bash 

python rank.py --dataset=ogbl-ppa_PA_10000_5000_0_seed1 --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA1000050000seed1resource_allocation_ppaPA1000050000seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaPA1000050000output.txt

python rank.py --dataset=ogbl-ppa_PA_20000_10000_0_seed1 --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA20000100000seed1resource_allocation_ppaPA20000100000seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaPA20000100000output.txt

python rank.py --dataset=ogbl-ppa_PA_25000_15000_0_seed1 --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA25000150000seed1resource_allocation_ppaPA25000150000seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaPA25000150000output.txt

python rank.py --dataset=ogbl-ppa_PA_10000_5000_0_seed1 --device=7 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaPA1000050000seed1resource_allocation_ppaPA1000050000seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_ppaPA1000050000output.txt

python rank.py --dataset=ogbl-ppa_PA_20000_10000_0_seed1 --device=7 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaPA20000100000seed1resource_allocation_ppaPA20000100000seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_ppaPA20000100000output.txt

python rank.py --dataset=ogbl-ppa_PA_25000_15000_0_seed1 --device=7 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaPA25000150000seed1resource_allocation_ppaPA25000150000seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_ppaPA25000150000output.txt

python rank.py --dataset=ogbl-ppa_PA_10000_5000_0_seed1 --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA1000050000seed1gcn_ppaPA1000050000seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_ppaPA1000050000output.txt

python rank.py --dataset=ogbl-ppa_PA_20000_10000_0_seed1 --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA20000100000seed1gcn_ppaPA20000100000seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_ppaPA20000100000output.txt

python rank.py --dataset=ogbl-ppa_PA_25000_15000_0_seed1 --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA25000150000seed1gcn_ppaPA25000150000seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_ppaPA25000150000output.txt