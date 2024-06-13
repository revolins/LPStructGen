#!/bin/bash 

# python rank.py --dataset=ogbl-collab_CN_2_1_0_seed1 --device=9 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN210seed1gcn_collabCN210seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_collabCN210output.txt

# python rank.py --dataset=ogbl-collab_CN_4_2_0_seed1 --device=9 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN420seed1gcn_collabCN420seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_collabCN420output.txt

# python rank.py --dataset=ogbl-collab_CN_5_3_0_seed1 --device=9 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN530seed1gcn_collabCN530seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_collabCN530output.txt

# python rank.py --dataset=ogbl-collab_PA_026_022_00_seed1 --device=9 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA02602200seed1gcn_collabPA02602200seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_collabPA02602200output.txt

# python rank.py --dataset=ogbl-collab_PA_034_022_00_seed1 --device=9 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA03402200seed1gcn_collabPA03402200seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_collabPA03402200output.txt

# python rank.py --dataset=ogbl-collab_PA_034_026_00_seed1 --device=9 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA03402600seed1gcn_collabPA03402600seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_collabPA03402600output.txt

# python rank.py --dataset=ogbl-collab_SP_026_017_00_seed1 --device=9 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabSP02601700seed1gcn_collabSP02601700seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_collabSP02601700output.txt

# python rank.py --dataset=ogbl-collab_SP_036_026_00_seed1 --device=9 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabSP03602600seed1gcn_collabSP03602600seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_collabSP03602600output.txt

python rank.py --dataset=ogbl-collab_CN_2_1_0_seed1 --device=9 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN210seed1buddy_collabCN210seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabCN210output.txt

python rank.py --dataset=ogbl-collab_CN_4_2_0_seed1 --device=9 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN420seed1buddy_collabCN420seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabCN420output.txt

python rank.py --dataset=ogbl-collab_CN_5_3_0_seed1 --device=9 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN530seed1buddy_collabCN530seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabCN530output.txt

python rank.py --dataset=ogbl-collab_PA_026_022_00_seed1 --device=9 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA02602200seed1buddy_collabPA02602200seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabPA02602200output.txt

python rank.py --dataset=ogbl-collab_PA_034_022_00_seed1 --device=9 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA03402200seed1buddy_collabPA03402200seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabPA03402200output.txt

python rank.py --dataset=ogbl-collab_PA_034_026_00_seed1 --device=9 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA03402600seed1buddy_collabPA03402600seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabPA03402600output.txt

python rank.py --dataset=ogbl-collab_SP_026_017_00_seed1 --device=9 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabSP02601700seed1buddy_collabSP02601700seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabSP02601700output.txt

python rank.py --dataset=ogbl-collab_SP_036_026_00_seed1 --device=9 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabSP03602600seed1buddy_collabSP03602600seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabSP03602600output.txt

# python rank.py --dataset=ogbl-collab_CN_2_1_0_seed1 --device=9 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN210seed1resource_allocation_collabCN210seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_collabCN210output.txt

# python rank.py --dataset=ogbl-collab_CN_4_2_0_seed1 --device=9 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN420seed1resource_allocation_collabCN420seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_collabCN420output.txt

# python rank.py --dataset=ogbl-collab_CN_5_3_0_seed1 --device=9 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN530seed1resource_allocation_collabCN530seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_collabCN530output.txt

# python rank.py --dataset=ogbl-collab_PA_026_022_00_seed1 --device=9 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA02602200seed1resource_allocation_collabPA02602200seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_collabPA02602200output.txt

# python rank.py --dataset=ogbl-collab_PA_034_022_00_seed1 --device=9 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA03402200seed1resource_allocation_collabPA03402200seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_collabPA03402200output.txt

# python rank.py --dataset=ogbl-collab_PA_034_026_00_seed1 --device=9 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA03402600seed1resource_allocation_collabPA03402600seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_collabPA03402600output.txt

# python rank.py --dataset=ogbl-collab_SP_026_017_00_seed1 --device=9 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabSP02601700seed1resource_allocation_collabSP02601700seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_collabSP02601700output.txt

# python rank.py --dataset=ogbl-collab_SP_036_026_00_seed1 --device=9 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabSP03602600seed1resource_allocation_collabSP03602600seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_collabSP03602600output.txt

# python rank.py --dataset=ogbl-collab_CN_2_1_0_seed1 --device=9 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN210seed1resource_allocation_collabCN210seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_collabCN210output.txt

# python rank.py --dataset=ogbl-collab_CN_4_2_0_seed1 --device=9 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN420seed1resource_allocation_collabCN420seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_collabCN420output.txt

# python rank.py --dataset=ogbl-collab_CN_5_3_0_seed1 --device=9 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN530seed1resource_allocation_collabCN530seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_collabCN530output.txt

# python rank.py --dataset=ogbl-collab_PA_026_022_00_seed1 --device=9 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA02602200seed1resource_allocation_collabPA02602200seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_collabPA02602200output.txt

# python rank.py --dataset=ogbl-collab_PA_034_022_00_seed1 --device=9 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA03402200seed1resource_allocation_collabPA03402200seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_collabPA03402200output.txt

# python rank.py --dataset=ogbl-collab_PA_034_026_00_seed1 --device=9 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA03402600seed1resource_allocation_collabPA03402600seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_collabPA03402600output.txt

# python rank.py --dataset=ogbl-collab_SP_026_017_00_seed1 --device=9 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabSP02601700seed1resource_allocation_collabSP02601700seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_collabSP02601700output.txt

# python rank.py --dataset=ogbl-collab_SP_036_026_00_seed1 --device=9 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabSP03602600seed1resource_allocation_collabSP03602600seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_collabSP03602600output.txt