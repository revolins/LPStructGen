#!/bin/bash 

python rank.py --dataset=ogbl-ppa_CN_2_1_0_seed1 --device=5 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaCN210seed1resource_allocation_ppaCN210seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaCN210output.txt

python rank.py --dataset=ogbl-ppa_CN_4_2_0_seed1 --device=5 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaCN420seed1resource_allocation_ppaCN420seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaCN420output.txt

python rank.py --dataset=ogbl-ppa_CN_5_3_0_seed1 --device=5 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaCN530seed1resource_allocation_ppaCN530seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaCN530output.txt

python rank.py --dataset=ogbl-ppa_PA_026_022_00_seed1 --device=5 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA02602200seed1resource_allocation_ppaPA02602200seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaPA02602200output.txt

python rank.py --dataset=ogbl-ppa_PA_034_022_00_seed1 --device=5 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA03402200seed1resource_allocation_ppaPA03402200seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaPA03402200output.txt

python rank.py --dataset=ogbl-ppa_PA_034_026_00_seed1 --device=5 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA03402600seed1resource_allocation_ppaPA03402600seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaPA03402600output.txt

python rank.py --dataset=ogbl-ppa_SP_026_017_00_seed1 --device=5 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaSP02601700seed1resource_allocation_ppaSP02601700seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaSP02601700output.txt

python rank.py --dataset=ogbl-ppa_SP_036_026_00_seed1 --device=5 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaSP03602600seed1resource_allocation_ppaSP03602600seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaSP03602600output.txt