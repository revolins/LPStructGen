#!/bin/bash 

python filter.py --model=gcn --dataset=ogbl-collab_CN_0_1_2_seed1 --run=1 --device=4

python filter.py --model=gcn --dataset=ogbl-collab_CN_0_2_4_seed1 --run=1 --device=4

python filter.py --model=gcn --dataset=ogbl-collab_CN_0_3_5_seed1 --run=1 --device=4

python filter.py --model=gcn --dataset=ogbl-collab_SP_00_017_026_seed1 --run=1 --device=4

python filter.py --model=gcn --dataset=ogbl-collab_SP_00_026_036_seed1 --run=1 --device=4

python filter.py --model=gcn --dataset=ogbl-collab_PA_00_022_026_seed1 --run=1 --device=4

python filter.py --model=gcn --dataset=ogbl-collab_PA_00_026_034_seed1 --run=1 --device=4

python filter.py --model=gcn --dataset=ogbl-collab_PA_00_022_034_seed1 --run=1 --device=4