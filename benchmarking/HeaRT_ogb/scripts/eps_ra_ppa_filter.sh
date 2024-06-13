#!/bin/bash 

python filter.py --model=resource_allocation --dataset=ogbl-ppa_SP_036_026_00_seed1 --run=1 --device=9

python filter.py --model=resource_allocation --dataset=ogbl-ppa_SP_00_026_036_seed1 --run=1 --device=9

python filter.py --model=resource_allocation --dataset=ogbl-collab_SP_036_026_00_seed1 --run=1 --device=9

python filter.py --model=resource_allocation --dataset=ogbl-collab_SP_00_026_036_seed1 --run=1 --device=9