#!/bin/bash 

#python rank.py --dataset=ogbl-collab_SP_00_026_036_seed1 --device=3 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaSP00026036seed1resource_allocation_ppaSP00026036seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_collabSP00026036output.txt

python rank.py --dataset=ogbl-collab_SP_026_017_00_seed1 --device=3 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabSP02601700seed1resource_allocation_collabSP02601700seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_collabSP02601700output.txt

python rank.py --dataset=ogbl-collab_SP_036_026_00_seed1 --device=3 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabSP03602600seed1resource_allocation_collabSP03602600seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_collabSP03602600output.txt

python rank.py --dataset=ogbl-collab_PA_026_022_00_seed1 --device=3 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA02602200seed1resource_allocation_collabPA02602200seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_collabPA02602200output.txt

python rank.py --dataset=ogbl-collab_PA_034_022_00_seed1 --device=3 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA03402200seed1resource_allocation_collabPA03402200seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_collabPA03402200output.txt

python rank.py --dataset=ogbl-collab_PA_034_026_00_seed1 --device=3 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA03402600seed1resource_allocation_collabPA03402600seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_collabPA03402600output.txt