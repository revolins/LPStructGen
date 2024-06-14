#!/bin/bash 

# All relevant commands to Rank edges necessary for Edge Proposal Sets
# NOTE: Edges for a given 'Filter' model must run before ranking
# NOTE: These commands will need to be run in the benchmarking/HeaRT_ogb folder

# BUDDY + GCN - Collab
python rank.py --dataset=ogbl-collab_CN_0_1_2_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN012seed1buddy_collabCN012seed1buddy_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_CN_0_2_4_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN024seed1buddy_collabCN024seed1buddy_0_1_sorted_edges.pt --runs=5 

python rank.py --dataset=ogbl-collab_CN_0_3_5_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN035seed1buddy_collabCN035seed1buddy_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_PA_0_50_100_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA050100seed1buddy_collabPA050100seed1buddy_0_1_sorted_edges.pt --runs=5 

python rank.py --dataset=ogbl-collab_PA_0_100_200_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA0100200seed1buddy_collabPA0100200seed1buddy_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_PA_0_150_250_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA0150250seed1buddy_collabPA0150250seed1buddy_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_SP_00_017_026_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabSP00017026seed1buddy_collabSP00017026seed1buddy_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_SP_00_026_036_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabSP00026036seed1buddy_collabSP00026036seed1buddy_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_CN_2_1_0_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN210seed1buddy_collabCN210seed1buddy_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_CN_4_2_0_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN420seed1buddy_collabCN420seed1buddy_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_CN_5_3_0_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN530seed1buddy_collabCN530seed1buddy_0_1_sorted_edges.pt --runs=5 

python rank.py --dataset=ogbl-collab_PA_250_150_0_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA2501500seed1buddy_collabPA2501500seed1buddy_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_PA_100_50_0_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA100500seed1buddy_collabPA100500seed1buddy_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_PA_200_100_0_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA2001000seed1buddy_collabPA2001000seed1buddy_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_SP_026_017_00_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabSP02601700seed1buddy_collabSP02601700seed1buddy_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_SP_036_026_00_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabSP03602600seed1buddy_collabSP03602600seed1buddy_0_1_sorted_edges.pt --runs=5

# GCN + BUDDY - Collab
python rank.py --dataset=ogbl-collab_CN_0_1_2_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN012seed1gcn_collabCN012seed1gcn_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_CN_0_2_4_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN024seed1gcn_collabCN024seed1gcn_0_1_sorted_edges.pt --runs=5 

python rank.py --dataset=ogbl-collab_CN_0_3_5_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN035seed1gcn_collabCN035seed1gcn_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_PA_0_50_100_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA050100seed1gcn_collabPA050100seed1gcn_0_1_sorted_edges.pt --runs=5 

python rank.py --dataset=ogbl-collab_PA_0_100_200_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA0100200seed1gcn_collabPA0100200seed1gcn_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_PA_0_150_250_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA0150250seed1gcn_collabPA0150250seed1gcn_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_SP_00_017_026_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabSP00017026seed1gcn_collabSP00017026seed1gcn_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_SP_00_026_036_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabSP00026036seed1gcn_collabSP00026036seed1gcn_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_CN_2_1_0_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN210seed1gcn_collabCN210seed1gcn_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_CN_4_2_0_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN420seed1gcn_collabCN420seed1gcn_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_CN_5_3_0_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN530seed1gcn_collabCN530seed1gcn_0_1_sorted_edges.pt --runs=5 

python rank.py --dataset=ogbl-collab_PA_250_150_0_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA2501500seed1gcn_collabPA2501500seed1gcn_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_PA_100_50_0_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA100500seed1gcn_collabPA100500seed1gcn_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_PA_200_100_0_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA2001000seed1gcn_collabPA2001000seed1gcn_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_SP_026_017_00_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabSP02601700seed1gcn_collabSP02601700seed1gcn_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_SP_036_026_00_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabSP03602600seed1gcn_collabSP03602600seed1gcn_0_1_sorted_edges.pt --runs=5

# RA + GCN - Collab
python rank.py --dataset=ogbl-collab_CN_0_1_2_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN012seed1resource_allocation_collabCN012seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_CN_0_2_4_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN024seed1resource_allocation_collabCN024seed1resource_allocation_0_1_sorted_edges.pt --runs=5 

python rank.py --dataset=ogbl-collab_CN_0_3_5_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN035seed1resource_allocation_collabCN035seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_PA_0_50_100_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA050100seed1resource_allocation_collabPA050100seed1resource_allocation_0_1_sorted_edges.pt --runs=5 

python rank.py --dataset=ogbl-collab_PA_0_100_200_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA0100200seed1resource_allocation_collabPA0100200seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_PA_0_150_250_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA0150250seed1resource_allocation_collabPA0150250seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_SP_00_017_026_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabSP00017026seed1resource_allocation_collabSP00017026seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_SP_00_026_036_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabSP00026036seed1resource_allocation_collabSP00026036seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_CN_2_1_0_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN210seed1resource_allocation_collabCN210seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_CN_4_2_0_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN420seed1resource_allocation_collabCN420seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_CN_5_3_0_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabCN530seed1resource_allocation_collabCN530seed1resource_allocation_0_1_sorted_edges.pt --runs=5 

python rank.py --dataset=ogbl-collab_PA_250_150_0_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA2501500seed1resource_allocation_collabPA2501500seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_PA_100_50_0_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA100500seed1resource_allocation_collabPA100500seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_PA_200_100_0_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabPA2001000seed1resource_allocation_collabPA2001000seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_SP_026_017_00_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabSP02601700seed1resource_allocation_collabSP02601700seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_SP_036_026_00_seed1 --device=0 --model=gcn --num_sorted_edge=250000 --sorted_edge_path=collabSP03602600seed1resource_allocation_collabSP03602600seed1resource_allocation_0_1_sorted_edges.pt --runs=5

# RA + BUDDY - Collab
python rank.py --dataset=ogbl-collab_CN_0_1_2_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN012seed1resource_allocation_collabCN012seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_CN_0_2_4_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN024seed1resource_allocation_collabCN024seed1resource_allocation_0_1_sorted_edges.pt --runs=5 

python rank.py --dataset=ogbl-collab_CN_0_3_5_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN035seed1resource_allocation_collabCN035seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_PA_0_50_100_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA050100seed1resource_allocation_collabPA050100seed1resource_allocation_0_1_sorted_edges.pt --runs=5 

python rank.py --dataset=ogbl-collab_PA_0_100_200_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA0100200seed1resource_allocation_collabPA0100200seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_PA_0_150_250_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA0150250seed1resource_allocation_collabPA0150250seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_SP_00_017_026_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabSP00017026seed1resource_allocation_collabSP00017026seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_SP_00_026_036_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabSP00026036seed1resource_allocation_collabSP00026036seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_CN_2_1_0_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN210seed1resource_allocation_collabCN210seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_CN_4_2_0_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN420seed1resource_allocation_collabCN420seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_CN_5_3_0_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabCN530seed1resource_allocation_collabCN530seed1resource_allocation_0_1_sorted_edges.pt --runs=5 

python rank.py --dataset=ogbl-collab_PA_250_150_0_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA2501500seed1resource_allocation_collabPA2501500seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_PA_100_50_0_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA100500seed1resource_allocation_collabPA100500seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_PA_200_100_0_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabPA2001000seed1resource_allocation_collabPA2001000seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_SP_026_017_00_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabSP02601700seed1resource_allocation_collabSP02601700seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-collab_SP_036_026_00_seed1 --device=0 --model=buddy --num_sorted_edge=250000 --sorted_edge_path=collabSP03602600seed1resource_allocation_collabSP03602600seed1resource_allocation_0_1_sorted_edges.pt --runs=5

# GCN + BUDDY - PPA
python rank.py --dataset=ogbl-ppa_CN_0_1_2_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaCN012seed1gcn_ppaCN012seed1gcn_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_CN_0_2_4_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaCN024seed1gcn_ppaCN024seed1gcn_0_1_sorted_edges.pt --runs=5 

python rank.py --dataset=ogbl-ppa_CN_0_3_5_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaCN035seed1gcn_ppaCN035seed1gcn_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_PA_0_5000_10000_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA0500010000seed1gcn_ppaPA0500010000seed1gcn_0_1_sorted_edges.pt --runs=5 

python rank.py --dataset=ogbl-ppa_PA_0_10000_20000_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA01000020000seed1gcn_ppaPA01000020000seed1gcn_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_PA_0_15000_25000_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA01500025000seed1gcn_ppaPA01500025000seed1gcn_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_SP_00_017_026_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaSP00017026seed1gcn_ppaSP00017026seed1gcn_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_SP_00_026_036_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaSP00026036seed1gcn_ppaSP00026036seed1gcn_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_CN_2_1_0_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaCN210seed1gcn_ppaCN210seed1gcn_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_CN_4_2_0_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaCN420seed1gcn_ppaCN420seed1gcn_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_CN_5_3_0_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaCN530seed1gcn_ppaCN530seed1gcn_0_1_sorted_edges.pt --runs=5 

python rank.py --dataset=ogbl-ppa_PA_25000_15000_0_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA25000150000seed1gcn_ppaPA25000150000seed1gcn_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_PA_10000_5000_0_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA1000050000seed1gcn_ppaPA1000050000seed1gcn_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_PA_20000_10000_0_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA20000100000seed1gcn_ppaPA20000100000seed1gcn_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_SP_026_017_00_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaSP02601700seed1gcn_ppaSP02601700seed1gcn_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_SP_036_026_00_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaSP03602600seed1gcn_ppaSP03602600seed1gcn_0_1_sorted_edges.pt --runs=5

# RA + GCN - PPA
python rank.py --dataset=ogbl-ppa_CN_0_1_2_seed1 --device=0 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaCN012seed1resource_allocation_ppaCN012seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_CN_0_2_4_seed1 --device=0 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaCN024seed1resource_allocation_ppaCN024seed1resource_allocation_0_1_sorted_edges.pt --runs=5 

python rank.py --dataset=ogbl-ppa_CN_0_3_5_seed1 --device=0 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaCN035seed1resource_allocation_ppaCN035seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_PA_0_5000_10000_seed1 --device=0 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaPA0500010000seed1resource_allocation_ppaPA0500010000seed1resource_allocation_0_1_sorted_edges.pt --runs=5 

python rank.py --dataset=ogbl-ppa_PA_0_10000_20000_seed1 --device=0 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaPA01000020000seed1resource_allocation_ppaPA01000020000seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_PA_0_15000_25000_seed1 --device=0 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaPA01500025000seed1resource_allocation_ppaPA01500025000seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_SP_00_017_026_seed1 --device=0 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaSP00017026seed1resource_allocation_ppaSP00017026seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_SP_00_026_036_seed1 --device=0 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaSP00026036seed1resource_allocation_ppaSP00026036seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_CN_2_1_0_seed1 --device=0 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaCN210seed1resource_allocation_ppaCN210seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_CN_4_2_0_seed1 --device=0 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaCN420seed1resource_allocation_ppaCN420seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_CN_5_3_0_seed1 --device=0 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaCN530seed1resource_allocation_ppaCN530seed1resource_allocation_0_1_sorted_edges.pt --runs=5 

python rank.py --dataset=ogbl-ppa_PA_25000_15000_0_seed1 --device=0 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaPA25000150000seed1resource_allocation_ppaPA25000150000seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_PA_10000_5000_0_seed1 --device=0 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaPA1000050000seed1resource_allocation_ppaPA1000050000seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_PA_20000_10000_0_seed1 --device=0 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaPA20000100000seed1resource_allocation_ppaPA20000100000seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_SP_026_017_00_seed1 --device=0 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaSP02601700seed1resource_allocation_ppaSP02601700seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_SP_036_026_00_seed1 --device=0 --model=gcn --num_sorted_edge=5000000 --sorted_edge_path=ppaSP03602600seed1resource_allocation_ppaSP03602600seed1resource_allocation_0_1_sorted_edges.pt --runs=5

# RA + BUDDY - PPA
python rank.py --dataset=ogbl-ppa_CN_0_1_2_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaCN012seed1resource_allocation_ppaCN012seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_CN_0_2_4_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaCN024seed1resource_allocation_ppaCN024seed1resource_allocation_0_1_sorted_edges.pt --runs=5 

python rank.py --dataset=ogbl-ppa_CN_0_3_5_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaCN035seed1resource_allocation_ppaCN035seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_PA_0_5000_10000_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA0500010000seed1resource_allocation_ppaPA0500010000seed1resource_allocation_0_1_sorted_edges.pt --runs=5 

python rank.py --dataset=ogbl-ppa_PA_0_10000_20000_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA01000020000seed1resource_allocation_ppaPA01000020000seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_PA_0_15000_25000_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA01500025000seed1resource_allocation_ppaPA01500025000seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_SP_00_017_026_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaSP00017026seed1resource_allocation_ppaSP00017026seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_SP_00_026_036_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaSP00026036seed1resource_allocation_ppaSP00026036seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_CN_2_1_0_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaCN210seed1resource_allocation_ppaCN210seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_CN_4_2_0_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaCN420seed1resource_allocation_ppaCN420seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_CN_5_3_0_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaCN530seed1resource_allocation_ppaCN530seed1resource_allocation_0_1_sorted_edges.pt --runs=5 

python rank.py --dataset=ogbl-ppa_PA_25000_15000_0_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA25000150000seed1resource_allocation_ppaPA25000150000seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_PA_10000_5000_0_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA1000050000seed1resource_allocation_ppaPA1000050000seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_PA_20000_10000_0_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaPA20000100000seed1resource_allocation_ppaPA20000100000seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_SP_026_017_00_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaSP02601700seed1resource_allocation_ppaSP02601700seed1resource_allocation_0_1_sorted_edges.pt --runs=5

python rank.py --dataset=ogbl-ppa_SP_036_026_00_seed1 --device=0 --model=buddy --num_sorted_edge=5000000 --sorted_edge_path=ppaSP03602600seed1resource_allocation_ppaSP03602600seed1resource_allocation_0_1_sorted_edges.pt --runs=5
