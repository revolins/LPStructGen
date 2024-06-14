#!/bin/bash

# COLLAB, CN
python main.py --dataset ogbl-collab_CN_0_1_2 --encoder_lr 0.001 --predictor_lr 0.001 --en_dp 0.3 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

python main.py --dataset ogbl-collab_CN_0_2_4 --encoder_lr 0.001 --predictor_lr 0.001 --en_dp 0.3 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

python main.py --dataset ogbl-collab_CN_0_3_5 --encoder_lr 0.001 --predictor_lr 0.001 --en_dp 0.3 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

python main.py --dataset ogbl-collab_CN_2_1_0 --encoder_lr 0.01 --predictor_lr 0.01 --en_dp 0.1 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

python main.py --dataset ogbl-collab_CN_4_2_0 --encoder_lr 0.01 --predictor_lr 0.01 --en_dp 0.3 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

python main.py --dataset ogbl-collab_CN_5_3_0 --encoder_lr 0.01 --predictor_lr 0.01 --en_dp 0.3 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

# COLLAB, PA
python main.py --dataset ogbl-collab_PA_0_50_100 --encoder_lr 0.001 --predictor_lr 0.001 --en_dp 0.3 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

python main.py --dataset ogbl-collab_PA_0_100_200 --encoder_lr 0.001 --predictor_lr 0.001 --en_dp 0.1 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

python main.py --dataset ogbl-collab_PA_0_100_250 --encoder_lr 0.001 --predictor_lr 0.001 --en_dp 0.1 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

python main.py --dataset ogbl-collab_PA_100_50_0 --encoder_lr 0.001 --predictor_lr 0.001 --en_dp 0.1 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

python main.py --dataset ogbl-collab_PA_200_100_0 --encoder_lr 0.001 --predictor_lr 0.001 --en_dp 0.1 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

python main.py --dataset ogbl-collab_PA_250_150_0 --encoder_lr 0.001 --predictor_lr 0.001 --en_dp 0.1 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

# COLLAB, SP
python main.py --dataset ogbl-collab_SP_00_017_026 --encoder_lr 0.001 --predictor_lr 0.001 --en_dp 0.3 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

python main.py --dataset ogbl-collab_SP_00_026_036 --encoder_lr 0.001 --predictor_lr 0.001 --en_dp 0.1 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

python main.py --dataset ogbl-collab_SP_026_017_00 --encoder_lr 0.01 --predictor_lr 0.01 --en_dp 0.3 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

python main.py --dataset ogbl-collab_SP_036_026_00 --encoder_lr 0.01 --predictor_lr 0.01 --en_dp 0.3 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

# All PPA Splits are OOM when using GCN with TC on a system with 1 TB Available RAM
# # PPA, CN
# python main.py --dataset ogbl-ppa_CN_0_1_2 --encoder_lr 0.01 --predictor_lr 0.01 --en_dp 0.3 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

# python main.py --dataset ogbl-ppa_CN_0_2_4 --encoder_lr 0.01 --predictor_lr 0.01 --en_dp 0.3 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

# python main.py --dataset ogbl-ppa_CN_0_3_5 --encoder_lr 0.01 --predictor_lr 0.01 --en_dp 0.1 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

# python main.py --dataset ogbl-ppa_CN_2_1_0 --encoder_lr 0.01 --predictor_lr 0.01 --en_dp 0.3 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

# python main.py --dataset ogbl-ppa_CN_4_2_0 --encoder_lr 0.001 --predictor_lr 0.001 --en_dp 0.1 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

# python main.py --dataset ogbl-ppa_CN_5_3_0 --encoder_lr 0.001 --predictor_lr 0.001 --en_dp 0.1 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

# # PPA, PA
# python main.py --dataset ogbl-ppa_PA_0_5000_10000 --encoder_lr 0.01 --predictor_lr 0.01 --en_dp 0.3 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

# python main.py --dataset ogbl-ppa_PA_0_10000_20000 --encoder_lr 0.001 --predictor_lr 0.001 --en_dp 0.1 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

# python main.py --dataset ogbl-ppa_PA_0_10000_25000 --encoder_lr 0.01 --predictor_lr 0.01 --en_dp 0.3 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

# python main.py --dataset ogbl-ppa_PA_10000_5000_0 --encoder_lr 0.001 --predictor_lr 0.001 --en_dp 0.1 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

# python main.py --dataset ogbl-ppa_PA_20000_10000_0 --encoder_lr 0.001 --predictor_lr 0.001 --en_dp 0.1 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

# python main.py --dataset ogbl-ppa_PA_25000_15000_0 --encoder_lr 0.001 --predictor_lr 0.001 --en_dp 0.1 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

# # PPA, SP
# python main.py --dataset ogbl-ppa_SP_00_017_026 --encoder_lr 0.001 --predictor_lr 0.001 --en_dp 0.3 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

# python main.py --dataset ogbl-ppa_SP_00_026_036 --encoder_lr 0.001 --predictor_lr 0.001 --en_dp 0.1 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

# python main.py --dataset ogbl-ppa_SP_026_017_00 --encoder_lr 0.01 --predictor_lr 0.01 --en_dp 0.3 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0

# python main.py --dataset ogbl-ppa_SP_036_026_00 --encoder_lr 0.01 --predictor_lr 0.01 --en_dp 0.1 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0