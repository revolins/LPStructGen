#!/bin/bash 

python rank.py --dataset=ogbl-collab_SP_00_026_036_seed1 --device=7 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabSP00026036seed1resource_allocation_collabSP00026036seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_collabSP00026036output.txt

python rank.py --dataset=ogbl-collab_SP_036_026_00_seed1 --device=7 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabSP03602600seed1resource_allocation_collabSP03602600seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_collabSP03602600output.txt

python rank.py --dataset=ogbl-collab_CN_0_1_2_seed1 --lr=0.001 --dropout=0.1 --device=7 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN012seed1gcn_collabCN012seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddylr1e3drp1_collabCN012output.txt

python rank.py --dataset=ogbl-collab_CN_0_1_2_seed1 --lr=0.001 --dropout=0.3 --device=7 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN012seed1gcn_collabCN012seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddylr1e3drp3_collabCN012output.txt

python rank.py --dataset=ogbl-collab_CN_0_1_2_seed1 --lr=0.01 --dropout=0.3 --device=7 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN012seed1gcn_collabCN012seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddylr1e2drp3_collabCN012output.txt

# python rank.py --dataset=ogbl-collab_CN_0_1_2_seed1 --device=7 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN012seed1gcn_collabCN012seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_collabCN012output.txt

# python rank.py --dataset=ogbl-collab_CN_0_2_4_seed1 --device=7 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN024seed1gcn_collabCN024seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_collabCN024output.txt

# python rank.py --dataset=ogbl-collab_CN_0_3_5_seed1 --device=7 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN035seed1gcn_collabCN035seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_collabCN035output.txt

# python rank.py --dataset=ogbl-collab_PA_00_022_026_seed1 --device=7 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA00022026seed1gcn_collabPA00022026seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_collabPA00022026output.txt

# python rank.py --dataset=ogbl-collab_PA_00_022_034_seed1 --device=7 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA00022034seed1gcn_collabPA00022034seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_collabPA00022034output.txt

# python rank.py --dataset=ogbl-collab_PA_00_026_034_seed1 --device=7 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA00026034seed1gcn_collabPA00026034seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_collabPA00026034output.txt

# python rank.py --dataset=ogbl-collab_SP_00_017_026_seed1 --device=7 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabSP00017026seed1gcn_collabSP00017026seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_collabSP00017026output.txt

# python rank.py --dataset=ogbl-collab_SP_00_026_036_seed1 --device=7 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabSP00026036seed1gcn_collabSP00026036seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_collabSP00026036output.txt

# python rank.py --dataset=ogbl-collab_CN_0_1_2_seed1 --device=7 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN012seed1buddy_collabCN012seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabCN012output.txt

# python rank.py --dataset=ogbl-collab_CN_0_2_4_seed1 --device=7 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN024seed1buddy_collabCN024seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabCN024output.txt

# python rank.py --dataset=ogbl-collab_CN_0_3_5_seed1 --device=7 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN035seed1buddy_collabCN035seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabCN035output.txt

# python rank.py --dataset=ogbl-collab_PA_00_022_026_seed1 --device=7 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA00022026seed1buddy_collabPA00022026seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabPA00022026output.txt

# python rank.py --dataset=ogbl-collab_PA_00_022_034_seed1 --device=7 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA00022034seed1buddy_collabPA00022034seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabPA00022034output.txt

# python rank.py --dataset=ogbl-collab_PA_00_026_034_seed1 --device=7 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA00026034seed1buddy_collabPA00026034seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabPA00026034output.txt

# python rank.py --dataset=ogbl-collab_SP_00_017_026_seed1 --device=7 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabSP00017026seed1buddy_collabSP00017026seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabSP00017026output.txt

# python rank.py --dataset=ogbl-collab_SP_00_026_036_seed1 --device=7 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabSP00026036seed1buddy_collabSP00026036seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_collabSP00026036output.txt

# python rank.py --dataset=ogbl-collab_CN_0_1_2_seed1 --device=7 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN012seed1resource_allocation_collabCN012seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_collabCN012output.txt

# python rank.py --dataset=ogbl-collab_CN_0_2_4_seed1 --device=7 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN024seed1resource_allocation_collabCN024seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_collabCN024output.txt

# python rank.py --dataset=ogbl-collab_CN_0_3_5_seed1 --device=7 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN035seed1resource_allocation_collabCN035seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_collabCN035output.txt

# python rank.py --dataset=ogbl-collab_PA_00_022_026_seed1 --device=7 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA00022026seed1resource_allocation_collabPA00022026seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_collabPA00022026output.txt

# python rank.py --dataset=ogbl-collab_PA_00_022_034_seed1 --device=7 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA00022034seed1resource_allocation_collabPA00022034seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_collabPA00022034output.txt

# python rank.py --dataset=ogbl-collab_PA_00_026_034_seed1 --device=7 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA00026034seed1resource_allocation_collabPA00026034seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_collabPA00026034output.txt

# python rank.py --dataset=ogbl-collab_SP_00_017_026_seed1 --device=7 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabSP00017026seed1resource_allocation_collabSP00017026seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_collabSP00017026output.txt

# python rank.py --dataset=ogbl-collab_SP_00_026_036_seed1 --device=7 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabSP00026036seed1resource_allocation_collabSP00026036seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_collabSP00026036output.txt



# python rank.py --dataset=ogbl-collab_CN_0_1_2_seed1 --device=7 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN012seed1resource_allocation_collabCN012seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_collabCN012output.txt

# python rank.py --dataset=ogbl-collab_CN_0_2_4_seed1 --device=7 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN024seed1resource_allocation_collabCN024seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_collabCN024output.txt

# python rank.py --dataset=ogbl-collab_CN_0_3_5_seed1 --device=7 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN035seed1resource_allocation_collabCN035seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_collabCN035output.txt

# python rank.py --dataset=ogbl-collab_PA_00_022_026_seed1 --device=7 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA00022026seed1resource_allocation_collabPA00022026seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_collabPA00022026output.txt

# python rank.py --dataset=ogbl-collab_PA_00_022_034_seed1 --device=7 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA00022034seed1resource_allocation_collabPA00022034seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_collabPA00022034output.txt

# python rank.py --dataset=ogbl-collab_PA_00_026_034_seed1 --device=7 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA00026034seed1resource_allocation_collabPA00026034seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_collabPA00026034output.txt

# python rank.py --dataset=ogbl-collab_SP_00_017_026_seed1 --device=7 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabSP00017026seed1resource_allocation_collabSP00017026seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_collabSP00017026output.txt

# python rank.py --dataset=ogbl-collab_SP_00_026_036_seed1 --device=7 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabSP00026036seed1resource_allocation_collabSP00026036seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_collabSP00026036output.txt