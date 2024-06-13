#!/bin/bash 

python rank.py --dataset=ogbl-collab_CN_0_1_2_seed1 --device=2 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN012seed1buddy_collabCN012seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabCN012output.txt

python rank.py --dataset=ogbl-collab_CN_0_2_4_seed1 --device=2 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN024seed1buddy_collabCN024seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabCN024output.txt

python rank.py --dataset=ogbl-collab_CN_0_3_5_seed1 --device=2 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN035seed1buddy_collabCN035seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabCN035output.txt

python rank.py --dataset=ogbl-collab_PA_00_022_026_seed1 --device=2 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA00022026seed1buddy_collabPA00022026seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabPA00022026output.txt

python rank.py --dataset=ogbl-collab_PA_00_022_034_seed1 --device=2 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA00022034seed1buddy_collabPA00022034seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabPA00022034output.txt

python rank.py --dataset=ogbl-collab_PA_00_026_034_seed1 --device=2 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA00026034seed1buddy_collabPA00026034seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabPA00026034output.txt

python rank.py --dataset=ogbl-collab_SP_00_017_026_seed1 --device=2 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabSP00017026seed1buddy_collabSP00017026seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabSP00017026output.txt

python rank.py --dataset=ogbl-collab_SP_00_026_036_seed1 --device=2 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabSP00026036seed1buddy_collabSP00026036seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabSP00026036output.txt

python rank.py --dataset=ogbl-collab_CN_2_1_0_seed1 --device=2 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN210seed1buddy_collabCN210seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabCN210output.txt

python rank.py --dataset=ogbl-collab_CN_4_2_0_seed1 --device=2 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN420seed1buddy_collabCN420seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabCN420output.txt

python rank.py --dataset=ogbl-collab_CN_5_3_0_seed1 --device=2 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN530seed1buddy_collabCN530seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabCN530output.txt

python rank.py --dataset=ogbl-collab_PA_026_022_00_seed1 --device=2 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA02602200seed1buddy_collabPA02602200seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabPA02602200output.txt

python rank.py --dataset=ogbl-collab_PA_034_022_00_seed1 --device=2 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA03402200seed1buddy_collabPA03402200seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabPA03402200output.txt

python rank.py --dataset=ogbl-collab_PA_034_026_00_seed1 --device=2 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA03402600seed1buddy_collabPA03402600seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabPA03402600output.txt

python rank.py --dataset=ogbl-collab_SP_026_017_00_seed1 --device=2 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabSP02601700seed1buddy_collabSP02601700seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabSP02601700output.txt

python rank.py --dataset=ogbl-collab_SP_036_026_00_seed1 --device=2 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabSP03602600seed1buddy_collabSP03602600seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabSP03602600output.txt