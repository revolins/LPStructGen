#!/bin/bash

# COLLAB, CN
python run.py --data_name ogbl-collab_CN_0_1_2 --lr 0.01 --all-drop 0.1 --gnn-layers 3 --dim 128 --batch-size 16384 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 16384

python run.py --data_name ogbl-collab_CN_0_2_4 --lr 0.001 --all-drop 0.1 --gnn-layers 3 --dim 128 --batch-size 16384 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 16384

python run.py --data_name ogbl-collab_CN_0_3_5 --lr 0.01 --all-drop 0.1 --gnn-layers 3 --dim 128 --batch-size 16384 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 16384

python run.py --data_name ogbl-collab_CN_2_1_0 --lr 0.001 --all-drop 0.3 --gnn-layers 3 --dim 128 --batch-size 16384 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 16384

python run.py --data_name ogbl-collab_CN_4_2_0 --lr 0.01 --all-drop 0.1 --gnn-layers 3 --dim 128 --batch-size 16384 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 16384

python run.py --data_name ogbl-collab_CN_5_3_0 --lr 0.001 --all-drop 0.1 --gnn-layers 3 --dim 128 --batch-size 16384 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 16384

# COLLAB, PA
python run.py --data_name ogbl-collab_PA_0_50_100 --lr 0.01 --all-drop 0.3 --gnn-layers 3 --dim 128 --batch-size 16384 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 16384

python run.py --data_name ogbl-collab_PA_0_100_200 --lr 0.01 --all-drop 0.3 --gnn-layers 3 --dim 128 --batch-size 16384 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 16384

python run.py --data_name ogbl-collab_PA_0_150_250 --lr 0.01 --all-drop 0.3 --gnn-layers 3 --dim 128 --batch-size 16384 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 16384

python run.py --data_name ogbl-collab_PA_100_50_0 --lr 0.001 --all-drop 0.1 --gnn-layers 3 --dim 128 --batch-size 16384 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 16384

python run.py --data_name ogbl-collab_PA_200_100_0 --lr 0.01 --all-drop 0.1 --gnn-layers 3 --dim 128 --batch-size 16384 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 16384

python run.py --data_name ogbl-collab_PA_250_150_0 --lr 0.01 --all-drop 0.1 --gnn-layers 3 --dim 128 --batch-size 16384 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 16384

# COLLAB, SP
python run.py --data_name ogbl-collab_SP_00_017_026 --lr 0.01 --all-drop 0.3 --gnn-layers 3 --dim 128 --batch-size 16384 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 16384

python run.py --data_name ogbl-collab_SP_00_026_036 --lr 0.001 --all-drop 0.3 --gnn-layers 3 --dim 128 --batch-size 16384 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 16384

python run.py --data_name ogbl-collab_SP_026_017_00 --lr 0.001 --all-drop 0.3 --gnn-layers 3 --dim 128 --batch-size 16384 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 16384

python run.py --data_name ogbl-collab_SP_036_026_0 --lr 0.001 --all-drop 0.3 --gnn-layers 3 --dim 128 --batch-size 16384 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 16384


# PPA, CN
python run.py --data_name ogbl-ppa_CN_0_1_2 --lr 0.001 --all-drop 0.3 --gnn-layers 3 --dim 128 --batch-size 4096 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 2048

python run.py --data_name ogbl-ppa_CN_0_2_4 --lr 0.01 --all-drop 0.1 --gnn-layers 3 --dim 128 --batch-size 4096 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 2048

python run.py --data_name ogbl-ppa_CN_0_3_5 --lr 0.01 --all-drop 0.1 --gnn-layers 3 --dim 128 --batch-size 4096 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 2048

python run.py --data_name ogbl-ppa_CN_2_1_0 --lr 0.01 --all-drop 0.1 --gnn-layers 3 --dim 128 --batch-size 4096 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 2048

python run.py --data_name ogbl-ppa_CN_4_2_0 --lr 0.001 --all-drop 0.1 --gnn-layers 3 --dim 128 --batch-size 4096 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 2048

python run.py --data_name ogbl-ppa_CN_5_3_0 --lr 0.001 --all-drop 0.1 --gnn-layers 3 --dim 128 --batch-size 4096 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 2048

# PPA, PA
python run.py --data_name ogbl-ppa_PA_0_5000_10000 --lr 0.001 --all-drop 0.3 --gnn-layers 3 --dim 128 --batch-size 4096 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 2048

python run.py --data_name ogbl-ppa_PA_0_10000_20000 --lr 0.001 --all-drop 0.3 --gnn-layers 3 --dim 128 --batch-size 4096 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 2048

python run.py --data_name ogbl-ppa_PA_0_15000_25000 --lr 0.001 --all-drop 0.3 --gnn-layers 3 --dim 128 --batch-size 4096 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 2048

python run.py --data_name ogbl-ppa_PA_10000_5000_0 --lr 0.01 --all-drop 0.3 --gnn-layers 3 --dim 128 --batch-size 4096 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 2048

python run.py --data_name ogbl-ppa_PA_20000_10000_0 --lr 0.01 --all-drop 0.3 --gnn-layers 3 --dim 128 --batch-size 4096 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 2048

python run.py --data_name ogbl-ppa_PA_25000_15000_0 --lr 0.01 --all-drop 0.3 --gnn-layers 3 --dim 128 --batch-size 4096 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 2048

# PPA, SP
python run.py --data_name ogbl-ppa_SP_00_017_026 --lr 0.001 --all-drop 0.3 --gnn-layers 3 --dim 128 --batch-size 4096 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 2048

python run.py --data_name ogbl-ppa_SP_00_026_036 --lr 0.001 --all-drop 0.3 --gnn-layers 3 --dim 128 --batch-size 4096 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 2048

python run.py --data_name ogbl-ppa_SP_026_017_00 --lr 0.001 --all-drop 0.3 --gnn-layers 3 --dim 128 --batch-size 4096 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 2048

python run.py --data_name ogbl-ppa_SP_036_026_00 --lr 0.001 --all-drop 0.1 --gnn-layers 3 --dim 128 --batch-size 4096 --epochs 15 --eval_steps 5 --runs 5 --device 0 --test-batch-size 2048