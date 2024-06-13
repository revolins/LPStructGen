#!/bin/bash 

python rank.py --dataset=ogbl-ppa_PA_0_5000_10000_seed1 --fnr --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA0500010000seed1resource_allocation_ppaPA0500010000seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaPA0500010000output.txt

python rank.py --dataset=ogbl-ppa_PA_0_10000_20000_seed1 --fnr --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA01000020000seed1resource_allocation_ppaPA01000020000seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaPA01000020000output.txt

python rank.py --dataset=ogbl-ppa_PA_0_15000_25000_seed1 --fnr --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA01500025000seed1resource_allocation_ppaPA01500025000seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaPA01500025000output.txt

python rank.py --dataset=ogbl-ppa_PA_10000_5000_0_seed1 --fnr --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA1000050000seed1resource_allocation_ppaPA1000050000seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaPA1000050000output.txt

python rank.py --dataset=ogbl-ppa_PA_20000_10000_0_seed1 --fnr --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA20000100000seed1resource_allocation_ppaPA20000100000seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaPA20000100000output.txt

python rank.py --dataset=ogbl-ppa_PA_25000_15000_0_seed1 --fnr --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA25000150000seed1resource_allocation_ppaPA25000150000seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaPA25000150000output.txt

python rank.py --dataset=ogbl-ppa_PA_0_5000_10000_seed1 --device=7 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaPA0500010000seed1resource_allocation_ppaPA0500010000seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_ppaPA0500010000output.txt

python rank.py --dataset=ogbl-ppa_PA_0_10000_20000_seed1 --device=7 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaPA01000020000seed1resource_allocation_ppaPA01000020000seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_ppaPA01000020000output.txt

python rank.py --dataset=ogbl-ppa_PA_0_15000_25000_seed1 --device=7 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaPA01500025000seed1resource_allocation_ppaPA01500025000seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_ppaPA01500025000output.txt

python rank.py --dataset=ogbl-ppa_PA_10000_5000_0_seed1 --device=7 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaPA1000050000seed1resource_allocation_ppaPA1000050000seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_ppaPA1000050000output.txt

python rank.py --dataset=ogbl-ppa_PA_20000_10000_0_seed1 --device=7 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaPA20000100000seed1resource_allocation_ppaPA20000100000seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_ppaPA20000100000output.txt

python rank.py --dataset=ogbl-ppa_PA_25000_15000_0_seed1 --device=7 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaPA25000150000seed1resource_allocation_ppaPA25000150000seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_ppaPA25000150000output.txt

python rank.py --dataset=ogbl-ppa_PA_0_5000_10000_seed1 --fnr --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA0500010000seed1gcn_ppaPA0500010000seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_ppaPA0500010000output.txt

python rank.py --dataset=ogbl-ppa_PA_0_10000_20000_seed1 --fnr --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA01000020000seed1gcn_ppaPA01000020000seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_ppaPA01000020000output.txt

python rank.py --dataset=ogbl-ppa_PA_0_15000_25000_seed1 --fnr --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA01500025000seed1gcn_ppaPA01500025000seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_ppaPA01500025000output.txt

python rank.py --dataset=ogbl-ppa_PA_10000_5000_0_seed1 --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA1000050000seed1gcn_ppaPA1000050000seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_ppaPA1000050000output.txt

python rank.py --dataset=ogbl-ppa_PA_20000_10000_0_seed1 --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA20000100000seed1gcn_ppaPA20000100000seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_ppaPA20000100000output.txt

python rank.py --dataset=ogbl-ppa_PA_25000_15000_0_seed1 --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA25000150000seed1gcn_ppaPA25000150000seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_ppaPA25000150000output.txt