#!/bin/bash 

python rank.py --dataset=ogbl-ppa_CN_0_1_2_seed1 --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaCN012seed1gcn_ppaCN012seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_ppaCN012output.txt

python rank.py --dataset=ogbl-ppa_CN_0_2_4_seed1 --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaCN024seed1gcn_ppaCN024seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_ppaCN024output.txt

python rank.py --dataset=ogbl-ppa_CN_0_3_5_seed1 --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaCN035seed1gcn_ppaCN035seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_ppaCN035output.txt

python rank.py --dataset=ogbl-ppa_PA_00_022_034_seed1 --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA00022034seed1gcn_ppaPA00022034seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_ppaPA00022034output.txt

python rank.py --dataset=ogbl-ppa_SP_00_017_026_seed1 --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaSP00017026seed1gcn_ppaSP00017026seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_ppaSP00017026output.txt

python rank.py --dataset=ogbl-ppa_SP_00_026_036_seed1 --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaSP00026036seed1gcn_ppaSP00026036seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_ppaSP00026036output.txt

python rank.py --dataset=ogbl-ppa_PA_00_022_026_seed1 --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA00022026seed1gcn_ppaPA00022026seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_ppaPA00022026output.txt

python rank.py --dataset=ogbl-ppa_PA_00_026_034_seed1 --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA00026034seed1gcn_ppaPA00026034seed1gcn_0_1_sorted_edges.pt --runs=5 &> output/gcnbuddy_ppaPA00026034output.txt

# python rank.py --dataset=ogbl-ppa_CN_0_1_2_seed1 --device=7 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaCN012seed1buddy_ppaCN012seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_ppaCN012output.txt

# python rank.py --dataset=ogbl-ppa_CN_0_2_4_seed1 --device=7 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaCN024seed1buddy_ppaCN024seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_ppaCN024output.txt

# python rank.py --dataset=ogbl-ppa_CN_0_3_5_seed1 --device=7 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaCN035seed1buddy_ppaCN035seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_ppaCN035output.txt

# python rank.py --dataset=ogbl-ppa_PA_00_022_026_seed1 --device=7 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaPA00022026seed1buddy_ppaPA00022026seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_ppaPA00022026output.txt

# python rank.py --dataset=ogbl-ppa_PA_00_022_034_seed1 --device=7 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaPA00022034seed1buddy_ppaPA00022034seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_ppaPA00022034output.txt

# python rank.py --dataset=ogbl-ppa_PA_00_026_034_seed1 --device=7 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaPA00026034seed1buddy_ppaPA00026034seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_ppaPA00026034output.txt

# python rank.py --dataset=ogbl-ppa_SP_00_017_026_seed1 --device=7 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaSP00017026seed1buddy_ppaSP00017026seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_ppaSP00017026output.txt

# python rank.py --dataset=ogbl-ppa_SP_00_026_036_seed1 --device=7 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaSP00026036seed1buddy_ppaSP00026036seed1buddy_0_1_sorted_edges.pt --runs=5 &> output/buddygcn_ppaSP00026036output.txt

# python rank.py --dataset=ogbl-ppa_CN_0_1_2_seed1 --device=7 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaCN012seed1resource_allocation_ppaCN012seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_ppaCN012output.txt

# python rank.py --dataset=ogbl-ppa_CN_0_2_4_seed1 --device=7 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaCN024seed1resource_allocation_ppaCN024seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_ppaCN024output.txt

# python rank.py --dataset=ogbl-ppa_CN_0_3_5_seed1 --device=7 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaCN035seed1resource_allocation_ppaCN035seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_ppaCN035output.txt

python rank.py --dataset=ogbl-ppa_PA_00_022_026_seed1 --device=6 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaPA00022026seed1resource_allocation_ppaPA00022026seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_ppaPA00022026output.txt

python rank.py --dataset=ogbl-ppa_PA_00_022_034_seed1 --device=6 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaPA00022034seed1resource_allocation_ppaPA00022034seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_ppaPA00022034output.txt

python rank.py --dataset=ogbl-ppa_PA_00_026_034_seed1 --device=6 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaPA00026034seed1resource_allocation_ppaPA00026034seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_ppaPA00026034output.txt

python rank.py --dataset=ogbl-ppa_SP_00_017_026_seed1 --device=6 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaSP00017026seed1resource_allocation_ppaSP00017026seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_ppaSP00017026output.txt

python rank.py --dataset=ogbl-ppa_SP_00_026_036_seed1 --device=6 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaSP00026036seed1resource_allocation_ppaSP00026036seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/ragcn_ppaSP00026036output.txt

# python rank.py --dataset=ogbl-ppa_CN_0_1_2_seed1 --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaCN012seed1resource_allocation_ppaCN012seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaCN012output.txt

# python rank.py --dataset=ogbl-ppa_CN_0_2_4_seed1 --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaCN024seed1resource_allocation_ppaCN024seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaCN024output.txt

# python rank.py --dataset=ogbl-ppa_CN_0_3_5_seed1 --device=7 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaCN035seed1resource_allocation_ppaCN035seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaCN035output.txt

# python rank.py --dataset=ogbl-ppa_PA_00_022_026_seed1 --device=6 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA00022026seed1resource_allocation_ppaPA00022026seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaPA00022026output.txt

# python rank.py --dataset=ogbl-ppa_PA_00_022_034_seed1 --device=6 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA00022034seed1resource_allocation_ppaPA00022034seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaPA00022034output.txt

# python rank.py --dataset=ogbl-ppa_PA_00_026_034_seed1 --device=6 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA00026034seed1resource_allocation_ppaPA00026034seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaPA00026034output.txt

# python rank.py --dataset=ogbl-ppa_SP_00_017_026_seed1 --device=6 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaSP00017026seed1resource_allocation_ppaSP00017026seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaSP00017026output.txt

# python rank.py --dataset=ogbl-ppa_SP_00_026_036_seed1 --device=6 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaSP00026036seed1resource_allocation_ppaSP00026036seed1resource_allocation_0_1_sorted_edges.pt --runs=5 &> output/rabuddy_ppaSP00026036output.txt