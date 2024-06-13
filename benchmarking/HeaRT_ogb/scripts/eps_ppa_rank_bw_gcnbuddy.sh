#!/bin/bash 

python rank.py --dataset=ogbl-ppa_CN_2_1_0_seed1 --device=9 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaCN210seed1gcn_ppaCN210seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_ppaCN210output.txt

python rank.py --dataset=ogbl-ppa_CN_4_2_0_seed1 --device=9 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaCN420seed1gcn_ppaCN420seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_ppaCN420output.txt

python rank.py --dataset=ogbl-ppa_CN_5_3_0_seed1 --device=9 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaCN530seed1gcn_ppaCN530seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_ppaCN530output.txt

python rank.py --dataset=ogbl-ppa_PA_026_022_00_seed1 --device=9 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA02602200seed1gcn_ppaPA02602200seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_ppaPA02602200output.txt

python rank.py --dataset=ogbl-ppa_PA_034_022_00_seed1 --device=9 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA03402200seed1gcn_ppaPA03402200seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_ppaPA03402200output.txt

python rank.py --dataset=ogbl-ppa_PA_034_026_00_seed1 --device=9 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA03402600seed1gcn_ppaPA03402600seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_ppaPA03402600output.txt

python rank.py --dataset=ogbl-ppa_SP_026_017_00_seed1 --device=9 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaSP02601700seed1gcn_ppaSP02601700seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_ppaSP02601700output.txt

python rank.py --dataset=ogbl-ppa_SP_036_026_00_seed1 --device=9 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaSP03602600seed1gcn_ppaSP03602600seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_ppaSP03602600output.txt

# python rank.py --dataset=ogbl-ppa_CN_2_1_0_seed1 --device=9 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaCN210seed1buddy_ppaCN210seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_ppaCN210output.txt

# python rank.py --dataset=ogbl-ppa_CN_4_2_0_seed1 --device=9 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaCN420seed1buddy_ppaCN420seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_ppaCN420output.txt

# python rank.py --dataset=ogbl-ppa_CN_5_3_0_seed1 --device=9 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaCN530seed1buddy_ppaCN530seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_ppaCN530output.txt

# python rank.py --dataset=ogbl-ppa_PA_026_022_00_seed1 --device=9 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaPA02602200seed1buddy_ppaPA02602200seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_ppaPA02602200output.txt

# python rank.py --dataset=ogbl-ppa_PA_034_022_00_seed1 --device=9 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaPA03402200seed1buddy_ppaPA03402200seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_ppaPA03402200output.txt

# python rank.py --dataset=ogbl-ppa_PA_034_026_00_seed1 --device=9 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaPA03402600seed1buddy_ppaPA03402600seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_ppaPA03402600output.txt

# python rank.py --dataset=ogbl-ppa_SP_026_017_00_seed1 --device=9 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaSP02601700seed1buddy_ppaSP02601700seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_ppaSP02601700output.txt

# python rank.py --dataset=ogbl-ppa_SP_036_026_00_seed1 --device=9 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaSP03602600seed1buddy_ppaSP03602600seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_ppaSP03602600output.txt

# python rank.py --dataset=ogbl-ppa_CN_2_1_0_seed1 --device=9 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaCN210seed1resource_allocation_ppaCN210seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_ppaCN210output.txt

# python rank.py --dataset=ogbl-ppa_CN_4_2_0_seed1 --device=9 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaCN420seed1resource_allocation_ppaCN420seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_ppaCN420output.txt

# python rank.py --dataset=ogbl-ppa_CN_5_3_0_seed1 --device=9 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaCN530seed1resource_allocation_ppaCN530seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_ppaCN530output.txt

# python rank.py --dataset=ogbl-ppa_PA_026_022_00_seed1 --device=9 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaPA02602200seed1resource_allocation_ppaPA02602200seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_ppaPA02602200output.txt

# python rank.py --dataset=ogbl-ppa_PA_034_022_00_seed1 --device=9 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaPA03402200seed1resource_allocation_ppaPA03402200seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_ppaPA03402200output.txt

# python rank.py --dataset=ogbl-ppa_PA_034_026_00_seed1 --device=9 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaPA03402600seed1resource_allocation_ppaPA03402600seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_ppaPA03402600output.txt

# python rank.py --dataset=ogbl-ppa_SP_026_017_00_seed1 --device=9 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaSP02601700seed1resource_allocation_ppaSP02601700seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_ppaSP02601700output.txt

# python rank.py --dataset=ogbl-ppa_SP_036_026_00_seed1 --device=9 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaSP03602600seed1resource_allocation_ppaSP03602600seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_ppaSP03602600output.txt

# python rank.py --dataset=ogbl-ppa_CN_2_1_0_seed1 --device=9 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaCN210seed1resource_allocation_ppaCN210seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaCN210output.txt

# python rank.py --dataset=ogbl-ppa_CN_4_2_0_seed1 --device=9 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaCN420seed1resource_allocation_ppaCN420seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaCN420output.txt

# python rank.py --dataset=ogbl-ppa_CN_5_3_0_seed1 --device=9 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaCN530seed1resource_allocation_ppaCN530seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaCN530output.txt

# python rank.py --dataset=ogbl-ppa_PA_026_022_00_seed1 --device=9 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA02602200seed1resource_allocation_ppaPA02602200seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaPA02602200output.txt

# python rank.py --dataset=ogbl-ppa_PA_034_022_00_seed1 --device=9 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA03402200seed1resource_allocation_ppaPA03402200seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaPA03402200output.txt

# python rank.py --dataset=ogbl-ppa_PA_034_026_00_seed1 --device=9 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA03402600seed1resource_allocation_ppaPA03402600seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaPA03402600output.txt

# python rank.py --dataset=ogbl-ppa_SP_026_017_00_seed1 --device=9 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaSP02601700seed1resource_allocation_ppaSP02601700seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaSP02601700output.txt

# python rank.py --dataset=ogbl-ppa_SP_036_026_00_seed1 --device=9 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaSP03602600seed1resource_allocation_ppaSP03602600seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaSP03602600output.txt