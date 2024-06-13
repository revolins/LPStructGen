#!/bin/bash 

python rank.py --dataset=ogbl-collab_PA_0_50_100_seed1 --device=6 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA050100seed1gcn_collabPA050100seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_collabPA050100output.txt

python rank.py --dataset=ogbl-collab_PA_0_100_200_seed1 --device=6 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA0100200seed1gcn_collabPA0100200seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_collabPA0100200output.txt

python rank.py --dataset=ogbl-collab_PA_0_150_250_seed1 --device=6 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA0150250seed1gcn_collabPA0150250seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_collabPA0150250output.txt

python rank.py --dataset=ogbl-collab_PA_0_50_100_seed1 --device=6 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA050100seed1buddy_collabPA050100seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabPA050100output.txt

python rank.py --dataset=ogbl-collab_PA_0_100_200_seed1 --device=6 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA0100200seed1buddy_collabPA0100200seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabPA0100200output.txt

python rank.py --dataset=ogbl-collab_PA_0_150_250_seed1 --device=6 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA0150250seed1buddy_collabPA0150250seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabPA0150250output.txt

python rank.py --dataset=ogbl-collab_PA_0_50_100_seed1 --device=6 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA050100seed1resource_allocation_collabPA050100seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_collabPA050100output.txt

python rank.py --dataset=ogbl-collab_PA_0_100_200_seed1 --device=6 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA0100200seed1resource_allocation_collabPA0100200seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_collabPA0100200output.txt

python rank.py --dataset=ogbl-collab_PA_0_150_250_seed1 --device=6 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA0150250seed1resource_allocation_collabPA0150250seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_collabPA0150250output.txt

python rank.py --dataset=ogbl-collab_PA_0_50_100_seed1 --device=6 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA050100seed1resource_allocation_collabPA050100seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_collabPA050100output.txt

python rank.py --dataset=ogbl-collab_PA_0_100_200_seed1 --device=6 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA0100200seed1resource_allocation_collabPA0100200seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_collabPA0100200output.txt

python rank.py --dataset=ogbl-collab_PA_0_150_250_seed1 --device=6 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA0150250seed1resource_allocation_collabPA0150250seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_collabPA0150250output.txt