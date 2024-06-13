#!/bin/bash 

python filter.py --model=gcn --dataset=ogbl-collab_CN_2_1_0_seed1 --run=1 --device=9

python filter.py --model=gcn --dataset=ogbl-collab_CN_4_2_0_seed1 --run=1 --device=9

python filter.py --model=gcn --dataset=ogbl-collab_CN_5_3_0_seed1 --run=1 --device=9

python filter.py --model=gcn --dataset=ogbl-collab_SP_026_017_00_seed1 --run=1 --device=9

python filter.py --model=gcn --dataset=ogbl-collab_SP_036_026_00_seed1 --run=1 --device=9

python filter.py --model=gcn --dataset=ogbl-collab_PA_026_022_00_seed1 --run=1 --device=9

python filter.py --model=gcn --dataset=ogbl-collab_PA_034_026_00_seed1 --run=1 --device=9

python filter.py --model=gcn --dataset=ogbl-collab_PA_034_022_00_seed1 --run=1 --device=9