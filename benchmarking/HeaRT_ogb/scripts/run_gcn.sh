#!/bin/bash

# All relevant baseline GCN commands for generating benchmark results on dataset splits tested in "Understanding the Generalizability of Link Predictors Under Distribution Shifts on Graphs".
# NOTE: These commands will need to be run in the benchmarking/HeaRT_ogb folder

# COLLAB, CN
python main_gnn.py --lr 0.001 --dropout 0.3 --test_batch_size 65536 --device 0 --data_name ogbl-collab_CN_0_1_2 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr 0.001 --dropout 0.3 --test_batch_size 65536 --device 0 --data_name ogbl-collab_CN_0_2_4 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr 0.001 --dropout 0.3 --test_batch_size 65536 --device 0 --data_name ogbl-collab_CN_0_3_5 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr 0.01 --dropout 0.1  --test_batch_size 65536 --device 0 --data_name ogbl-collab_CN_2_1_0 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr 0.01 --dropout 0.3 --test_batch_size 65536 --device 0 --data_name ogbl-collab_CN_4_2_0 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr 0.01 --dropout 0.3 --test_batch_size 65536 --device 0 --data_name ogbl-collab_CN_5_3_0 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

# COLLAB, PA
python main_gnn.py --lr 0.001 --dropout 0.3 --test_batch_size 65536 --device 0 --data_name ogbl-collab_PA_0_50_100 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr 0.001 --dropout 0.3 --test_batch_size 65536 --device 0 --data_name ogbl-collab_PA_0_100_200 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr 0.001 --dropout 0.3 --test_batch_size 65536 --device 0 --data_name ogbl-collab_PA_0_150_250 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr 0.001 --dropout 0.1 --test_batch_size 65536 --device 0 --data_name ogbl-collab_PA_100_50_0 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr 0.001 --dropout 0.1 --test_batch_size 65536 --device 0 --data_name ogbl-collab_PA_200_100_0 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr 0.001 --dropout 0.1 --test_batch_size 65536 --device 0 --data_name ogbl-collab_PA_250_150_0 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

# COLLAB, SP
python main_gnn.py --lr 0.001 --dropout 0.3 --test_batch_size 65536 --device 0 --data_name ogbl-collab_SP_00_017_026 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr 0.001 --dropout 0.1 --test_batch_size 65536 --device 0 --data_name ogbl-collab_SP_00_026_036 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr 0.01 --dropout 0.1 --test_batch_size 65536 --device 0 --data_name ogbl-collab_SP_026_017_00 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr 0.001 --dropout 0.1 --test_batch_size 65536 --device 0 --data_name ogbl-collab_SP_036_026_00 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

# PPA, CN
python main_gnn.py --lr 0.01 --dropout 0.3 --test_batch_size 65536 --device 0 --data_name ogbl-ppa_CN_0_1_2 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr 0.01 --dropout 0.3 --test_batch_size 65536 --device 0 --data_name ogbl-ppa_CN_0_2_4 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr 0.01 --dropout 0.1 --test_batch_size 65536 --device 0 --data_name ogbl-ppa_CN_0_3_5 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr 0.01 --dropout 0.3 --test_batch_size 65536 --device 0 --data_name ogbl-ppa_CN_2_1_0 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr 0.001 --dropout 0.1 --test_batch_size 65536 --device 0 --data_name ogbl-ppa_CN_4_2_0 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr 0.001 --dropout 0.1 --test_batch_size 65536 --device 0 --data_name ogbl-ppa_CN_5_3_0 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

# PPA, PA
python main_gnn.py --lr 0.01 --dropout 0.3 --test_batch_size 65536 --device 0 --data_name ogbl-ppa_PA_0_5000_10000 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr 0.001 --dropout 0.1 --test_batch_size 65536 --device 0 --data_name ogbl-ppa_PA_0_10000_20000 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr 0.01 --dropout 0.3 --test_batch_size 65536 --device 0 --data_name ogbl-ppa_PA_0_15000_25000 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr 0.001 --dropout 0.1 --test_batch_size 65536 --device 0 --data_name ogbl-ppa_PA_10000_5000_0 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr 0.001 --dropout 0.1 --test_batch_size 65536 --device 0 --data_name ogbl-ppa_PA_20000_10000_0 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr 0.001 --dropout 0.1 --test_batch_size 65536 --device 0 --data_name ogbl-ppa_PA_25000_15000_0 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

# PPA, SP
python main_gnn.py --lr 0.001 --dropout 0.3 --test_batch_size 65536 --device 0 --data_name ogbl-ppa_SP_00_017_026 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr 0.001 --dropout 0.1 --test_batch_size 65536 --device 0 --data_name ogbl-ppa_SP_00_026_036 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr 0.01 --dropout 0.3 --test_batch_size 65536 --device 0 --data_name ogbl-ppa_SP_026_017_00 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr 0.01 --dropout 0.1 --test_batch_size 65536 --device 0 --data_name ogbl-ppa_SP_036_026_00 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536