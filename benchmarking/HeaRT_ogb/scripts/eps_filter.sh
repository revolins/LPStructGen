#!/bin/bash 

# All relevant commands to Filter edges necessary for Edge Proposal Sets
# NOTE: These commands will need to be run in the benchmarking/HeaRT_ogb folder

# BUDDY - Collab
python filter.py --model=buddy --dataset=ogbl-collab_CN_0_1_2 --run=1 --device=0

python filter.py --model=buddy --dataset=ogbl-collab_CN_0_2_4 --run=1 --device=0

python filter.py --model=buddy --dataset=ogbl-collab_CN_0_3_5 --run=1 --device=0

python filter.py --model=buddy --dataset=ogbl-collab_CN_2_1_0 --run=1 --device=0

python filter.py --model=buddy --dataset=ogbl-collab_CN_4_2_0 --run=1 --device=0

python filter.py --model=buddy --dataset=ogbl-collab_CN_5_3_0 --run=1 --device=0

python filter.py --model=buddy --dataset=ogbl-collab_SP_00_017_026 --run=1 --device=0

python filter.py --model=buddy --dataset=ogbl-collab_SP_00_026_036 --run=1 --device=0

python filter.py --model=buddy --dataset=ogbl-collab_SP_026_017_00 --run=1 --device=0

python filter.py --model=buddy --dataset=ogbl-collab_SP_036_026_00 --run=1 --device=0

python filter.py --model=buddy --dataset=ogbl-collab_PA_0_50_100 --run=1 --device=0

python filter.py --model=buddy --dataset=ogbl-collab_PA_0_100_200 --run=1 --device=0

python filter.py --model=buddy --dataset=ogbl-collab_PA_0_150_250 --run=1 --device=0

python filter.py --model=buddy --dataset=ogbl-collab_PA_100_50_0 --run=1 --device=0

python filter.py --model=buddy --dataset=ogbl-collab_PA_200_100_0 --run=1 --device=0

python filter.py --model=buddy --dataset=ogbl-collab_PA_250_150_0 --run=1 --device=0

# OOM on system with 1TB of available RAM
# python filter.py --model=buddy --dataset=ogbl-ppa_CN_0_1_2_seed1 --run=1 --device=0

# python filter.py --model=buddy --dataset=ogbl-ppa_CN_0_2_4_seed1 --run=1 --device=0

# python filter.py --model=buddy --dataset=ogbl-ppa_CN_0_3_5_seed1 --run=1 --device=0

# python filter.py --model=buddy --dataset=ogbl-ppa_SP_00_017_026_seed1 --run=1 --device=0

# python filter.py --model=buddy --dataset=ogbl-ppa_SP_00_026_036_seed1 --run=1 --device=0

# python filter.py --model=buddy --dataset=ogbl-ppa_PA_0_5000_10000_seed1 --run=1 --device=0

# python filter.py --model=buddy --dataset=ogbl-ppa_PA_0_10000_20000_seed1 --run=1 --device=0

# python filter.py --model=buddy --dataset=ogbl-ppa_PA_0_15000_25000_seed1 --run=1 --device=0

# python filter.py --model=buddy --dataset=ogbl-ppa_CN_2_1_0_seed1 --run=1 --device=0

# python filter.py --model=buddy --dataset=ogbl-ppa_CN_4_2_0_seed1 --run=1 --device=0

# python filter.py --model=buddy --dataset=ogbl-ppa_CN_5_3_0_seed1 --run=1 --device=0

# python filter.py --model=buddy --dataset=ogbl-ppa_SP_026_017_00_seed1 --run=1 --device=0

# python filter.py --model=buddy --dataset=ogbl-ppa_SP_036_026_00_seed1 --run=1 --device=0

# python filter.py --model=buddy --dataset=ogbl-ppa_PA_10000_5000_0_seed1 --run=1 --device=0

# python filter.py --model=buddy --dataset=ogbl-ppa_PA_20000_10000_0_seed1 --run=1 --device=0

# python filter.py --model=buddy --dataset=ogbl-ppa_PA_25000_15000_0_seed1 --run=1 --device=0

# RA - Collab
python filter.py --model=resource_allocation --dataset=ogbl-collab_PA_0_50_100_seed1 --run=1 --device=0

python filter.py --model=resource_allocation --dataset=ogbl-collab_PA_0_100_200_seed1 --run=1 --device=0

python filter.py --model=resource_allocation --dataset=ogbl-collab_PA_0_150_250_seed1 --run=1 --device=0

python filter.py --model=resource_allocation --dataset=ogbl-collab_PA_100_50_0_seed1 --run=1 --device=0

python filter.py --model=resource_allocation --dataset=ogbl-collab_PA_200_100_0_seed1 --run=1 --device=0

python filter.py --model=resource_allocation --dataset=ogbl-collab_PA_250_150_0_seed1 --run=1 --device=0

python filter.py --model=resource_allocation --dataset=ogbl-collab_CN_0_1_2_seed1 --run=1 --device=9

python filter.py --model=resource_allocation --dataset=ogbl-collab_CN_0_2_4_seed1 --run=1 --device=9

python filter.py --model=resource_allocation --dataset=ogbl-collab_CN_0_3_5_seed1 --run=1 --device=9

python filter.py --model=resource_allocation --dataset=ogbl-collab_CN_2_1_0_seed1 --run=1 --device=9

python filter.py --model=resource_allocation --dataset=ogbl-collab_CN_4_2_0_seed1 --run=1 --device=9

python filter.py --model=resource_allocation --dataset=ogbl-collab_CN_5_3_0_seed1 --run=1 --device=9

python filter.py --model=resource_allocation --dataset=ogbl-collab_SP_026_017_00_seed1 --run=1 --device=9

python filter.py --model=resource_allocation --dataset=ogbl-collab_SP_036_026_00_seed1 --run=1 --device=9

python filter.py --model=resource_allocation --dataset=ogbl-collab_SP_00_017_026_seed1 --run=1 --device=9

python filter.py --model=resource_allocation --dataset=ogbl-collab_SP_00_026_036_seed1 --run=1 --device=9

# GCN - Collab
python filter.py --model=gcn --dataset=ogbl-collab_PA_0_50_100_seed1 --run=1 --device=0

python filter.py --model=gcn --dataset=ogbl-collab_PA_0_100_200_seed1 --run=1 --device=0

python filter.py --model=gcn --dataset=ogbl-collab_PA_0_150_250_seed1 --run=1 --device=0

python filter.py --model=gcn --dataset=ogbl-collab_PA_100_50_0_seed1 --run=1 --device=0

python filter.py --model=gcn --dataset=ogbl-collab_PA_200_100_0_seed1 --run=1 --device=0

python filter.py --model=gcn --dataset=ogbl-collab_PA_250_150_0_seed1 --run=1 --device=0

python filter.py --model=gcn --dataset=ogbl-collab_CN_0_1_2_seed1 --run=1 --device=9

python filter.py --model=gcn --dataset=ogbl-collab_CN_0_2_4_seed1 --run=1 --device=9

python filter.py --model=gcn --dataset=ogbl-collab_CN_0_3_5_seed1 --run=1 --device=9

python filter.py --model=gcn --dataset=ogbl-collab_CN_2_1_0_seed1 --run=1 --device=9

python filter.py --model=gcn --dataset=ogbl-collab_CN_4_2_0_seed1 --run=1 --device=9

python filter.py --model=gcn --dataset=ogbl-collab_CN_5_3_0_seed1 --run=1 --device=9

python filter.py --model=gcn --dataset=ogbl-collab_SP_026_017_00_seed1 --run=1 --device=9

python filter.py --model=gcn --dataset=ogbl-collab_SP_036_026_00_seed1 --run=1 --device=9

python filter.py --model=gcn --dataset=ogbl-collab_SP_00_017_026_seed1 --run=1 --device=9

python filter.py --model=gcn --dataset=ogbl-collab_SP_00_026_036_seed1 --run=1 --device=9


# RA - PPA
python filter.py --model=resource_allocation --dataset=ogbl-ppa_PA_0_5000_10000_seed1 --run=1 --device=0

python filter.py --model=resource_allocation --dataset=ogbl-ppa_PA_0_10000_20000_seed1 --run=1 --device=0

python filter.py --model=resource_allocation --dataset=ogbl-ppa_PA_0_15000_25000_seed1 --run=1 --device=0

python filter.py --model=resource_allocation --dataset=ogbl-ppa_PA_10000_5000_0_seed1 --run=1 --device=0

python filter.py --model=resource_allocation --dataset=ogbl-ppa_PA_20000_10000_0_seed1 --run=1 --device=0

python filter.py --model=resource_allocation --dataset=ogbl-ppa_PA_25000_15000_0_seed1 --run=1 --device=0

python filter.py --model=resource_allocation --dataset=ogbl-ppa_CN_0_1_2_seed1 --run=1 --device=9

python filter.py --model=resource_allocation --dataset=ogbl-ppa_CN_0_2_4_seed1 --run=1 --device=9

python filter.py --model=resource_allocation --dataset=ogbl-ppa_CN_0_3_5_seed1 --run=1 --device=9

python filter.py --model=resource_allocation --dataset=ogbl-ppa_CN_2_1_0_seed1 --run=1 --device=9

python filter.py --model=resource_allocation --dataset=ogbl-ppa_CN_4_2_0_seed1 --run=1 --device=9

python filter.py --model=resource_allocation --dataset=ogbl-ppa_CN_5_3_0_seed1 --run=1 --device=9

python filter.py --model=resource_allocation --dataset=ogbl-ppa_SP_026_017_00_seed1 --run=1 --device=9

python filter.py --model=gresource_allocation --dataset=ogbl-ppa_SP_036_026_00_seed1 --run=1 --device=9

python filter.py --model=resource_allocation --dataset=ogbl-ppa_SP_00_017_026_seed1 --run=1 --device=9

python filter.py --model=resource_allocation --dataset=ogbl-ppa_SP_00_026_036_seed1 --run=1 --device=9

# GCN - PPA
python filter.py --model=gcn --dataset=ogbl-ppa_PA_0_5000_10000_seed1 --run=1 --device=0

python filter.py --model=gcn --dataset=ogbl-ppa_PA_0_10000_20000_seed1 --run=1 --device=0

python filter.py --model=gcn --dataset=ogbl-ppa_PA_0_15000_25000_seed1 --run=1 --device=0

python filter.py --model=gcn --dataset=ogbl-ppa_PA_10000_5000_0_seed1 --run=1 --device=0

python filter.py --model=gcn --dataset=ogbl-ppa_PA_20000_10000_0_seed1 --run=1 --device=0

python filter.py --model=gcn --dataset=ogbl-ppa_PA_25000_15000_0_seed1 --run=1 --device=0

python filter.py --model=gcn --dataset=ogbl-ppa_CN_0_1_2_seed1 --run=1 --device=9

python filter.py --model=gcn --dataset=ogbl-ppa_CN_0_2_4_seed1 --run=1 --device=9

python filter.py --model=gcn --dataset=ogbl-ppa_CN_0_3_5_seed1 --run=1 --device=9

python filter.py --model=gcn --dataset=ogbl-ppa_CN_2_1_0_seed1 --run=1 --device=9

python filter.py --model=gcn --dataset=ogbl-ppa_CN_4_2_0_seed1 --run=1 --device=9

python filter.py --model=gcn --dataset=ogbl-ppa_CN_5_3_0_seed1 --run=1 --device=9

python filter.py --model=gcn --dataset=ogbl-ppa_SP_026_017_00_seed1 --run=1 --device=9

python filter.py --model=gcn --dataset=ogbl-ppa_SP_036_026_00_seed1 --run=1 --device=9

python filter.py --model=gcn --dataset=ogbl-ppa_SP_00_017_026_seed1 --run=1 --device=9

python filter.py --model=gcn --dataset=ogbl-ppa_SP_00_026_036_seed1 --run=1 --device=9