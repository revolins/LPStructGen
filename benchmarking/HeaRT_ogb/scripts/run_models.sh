#!/bin/bash

# All available commands for generating benchmark results on dataset splits tested in "Understanding the Generalizability of Link Predictors Under Distribution Shifts on Graphs".
# NOTE: These commands will need to be run in the benchmarking/HeaRT_ogb folder

# GCN - Collab, CN
python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-collab_CN_0_1_2 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-collab_CN_0_2_4 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-collab_CN_0_3_5 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-collab_CN_2_1_0 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-collab_CN_4_2_0 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-collab_CN_5_3_0 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

# GCN - Collab, PA
python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-collab_PA_0_50_100 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-collab_PA_0_100_200 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-collab_PA_0_150_250 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-collab_PA_100_50_0 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-collab_PA_200_100_0 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-collab_PA_250_150_0 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

# GCN - Collab, SP
python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-collab_SP_00_017_026 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-collab_SP_00_026_036 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-collab_SP_026_017_00 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-collab_SP_036_026_00 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

# GCN - PPA, CN
python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-ppa_CN_0_1_2 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-ppa_CN_0_2_4 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-ppa_CN_0_3_5 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-ppa_CN_2_1_0 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-ppa_CN_4_2_0 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-ppa_CN_5_3_0 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

# GCN - PPA, PA
python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-ppa_PA_0_5000_10000 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-ppa_PA_0_10000_20000 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-ppa_PA_0_15000_25000 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-ppa_PA_10000_5000_0 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-ppa_PA_20000_10000_0 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-ppa_PA_25000_15000_0 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

# GCN - PPA, SP
python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-ppa_SP_00_017_026 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-ppa_SP_00_026_036 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-ppa_SP_026_017_00 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536

python main_gnn.py --lr $lr --dropout $drop --test_batch_size 65536 --device 0 --data_name ogbl-ppa_SP_036_026_00 --gnn_model GCN --runs 5 --num_layers 3 --hidden_channels 128 --num_layers_predictor 3 --epochs 1000 --kill_cnt 100 --eval_steps 20 --batch_size 65536