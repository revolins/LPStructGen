#!/bin/bash

# All relevant baseline BUDDY commands for generating benchmark results on dataset splits tested in "Understanding the Generalizability of Link Predictors Under Distribution Shifts on Graphs".
# NOTE: These commands will need to be run in the benchmarking/HeaRT_ogb folder


# BUDDY - COLLAB, CN
python main_buddy.py --dataset ogbl-collab_CN_0_1_2 --device 0 --runs 5 --lr 0.01 --l2 0  --label_dropout 0.1  --feature_dropout 0.1 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY

python main_buddy.py --dataset ogbl-collab_CN_0_2_4 --device 0 --runs 5 --lr 0.001 --l2 0  --label_dropout 0.3  --feature_dropout 0.3 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY

python main_buddy.py --dataset ogbl-collab_CN_0_3_5 --device 0 --runs 5 --lr 0.001 --l2 0  --label_dropout 0.1  --feature_dropout 0.1 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY

python main_buddy.py --dataset ogbl-collab_CN_2_1_0 --device 0 --runs 5 --lr 0.001 --l2 0  --label_dropout 0.1  --feature_dropout 0.1 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY

python main_buddy.py --dataset ogbl-collab_CN_4_2_0 --device 0 --runs 5 --lr 0.001 --l2 0  --label_dropout 0.1  --feature_dropout 0.1 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY

python main_buddy.py --dataset ogbl-collab_CN_5_3_0 --device 0 --runs 5 --lr 0.001 --l2 0  --label_dropout 0.1  --feature_dropout 0.1 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY


# BUDDY - COLLAB, PA
python main_buddy.py --dataset ogbl-collab_PA_0_50_100 --device 0 --runs 5 --lr 0.01 --l2 0  --label_dropout 0.3  --feature_dropout 0.3 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY

python main_buddy.py --dataset ogbl-collab_PA_0_100_200 --device 0 --runs 5 --lr 0.01 --l2 0  --label_dropout 0.3  --feature_dropout 0.3 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY

python main_buddy.py --dataset ogbl-collab_PA_0_150_250 --device 0 --runs 5 --lr 0.01 --l2 0  --label_dropout 0.3  --feature_dropout 0.3 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY

python main_buddy.py --dataset ogbl-collab_PA_100_50_0 --device 0 --runs 5 --lr 0.001 --l2 0  --label_dropout 0.1  --feature_dropout 0.1 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY

python main_buddy.py --dataset ogbl-collab_PA_200_100_0 --device 0 --runs 5 --lr 0.001 --l2 0  --label_dropout 0.1  --feature_dropout 0.1 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY

python main_buddy.py --dataset ogbl-collab_PA_250_150_0 --device 0 --runs 5 --lr 0.01 --l2 0  --label_dropout 0.3  --feature_dropout 0.3 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY


# BUDDY - COLLAB, SP
python main_buddy.py --dataset ogbl-collab_SP_00_017_026 --device 0 --runs 5 --lr 0.01 --l2 0  --label_dropout 0.3  --feature_dropout 0.3 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY

python main_buddy.py --dataset ogbl-collab_SP_00_026_036 --device 0 --runs 5 --lr 0.001 --l2 0  --label_dropout 0.3  --feature_dropout 0.3 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY

python main_buddy.py --dataset ogbl-collab_SP_026_017_00 --device 0 --runs 5 --lr 0.01 --l2 0  --label_dropout 0.1  --feature_dropout 0.1 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY

python main_buddy.py --dataset ogbl-collab_SP_036_026_00 --device 0 --runs 5 --lr 0.001 --l2 0  --label_dropout 0.1  --feature_dropout 0.1 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY


# BUDDY - PPA, CN
python main_buddy.py --dataset ogbl-ppa_CN_0_1_2 --device 0 --runs 5 --lr 0.01 --l2 0  --label_dropout 0.3  --feature_dropout 0.3 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY

python main_buddy.py --dataset ogbl-ppa_CN_0_2_4 --device 0 --runs 5 --lr 0.01 --l2 0  --label_dropout 0.3  --feature_dropout 0.3 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY

python main_buddy.py --dataset ogbl-ppa_CN_0_3_5 --device 0 --runs 5 --lr 0.01 --l2 0  --label_dropout 0.3  --feature_dropout 0.3 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY

python main_buddy.py --dataset ogbl-ppa_CN_2_1_0 --device 0 --runs 5 --lr 0.01 --l2 0  --label_dropout 0.3  --feature_dropout 0.3 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY

python main_buddy.py --dataset ogbl-ppa_CN_4_2_0 --device 0 --runs 5 --lr 0.001 --l2 0  --label_dropout 0.1  --feature_dropout 0.1 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY

python main_buddy.py --dataset ogbl-ppa_CN_5_3_0 --device 0 --runs 5 --lr 0.001 --l2 0  --label_dropout 0.1  --feature_dropout 0.1 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY


# BUDDY - PPA, PA
python main_buddy.py --dataset ogbl-ppa_PA_0_5000_10000 --device 0 --runs 5 --lr 0.01 --l2 0  --label_dropout 0.1  --feature_dropout 0.1 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY

python main_buddy.py --dataset ogbl-ppa_PA_0_10000_20000 --device 0 --runs 5 --lr 0.01 --l2 0  --label_dropout 0.1  --feature_dropout 0.1 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY

python main_buddy.py --dataset ogbl-ppa_PA_0_15000_25000 --device 0 --runs 5 --lr 0.001 --l2 0  --label_dropout 0.3  --feature_dropout 0.3 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY

python main_buddy.py --dataset ogbl-ppa_PA_10000_5000_0 --device 0 --runs 5 --lr 0.01 --l2 0  --label_dropout 0.3  --feature_dropout 0.3 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY

python main_buddy.py --dataset ogbl-ppa_PA_20000_10000_0 --device 0 --runs 5 --lr 0.01 --l2 0  --label_dropout 0.3  --feature_dropout 0.3 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY

python main_buddy.py --dataset ogbl-ppa_PA_25000_15000_0 --device 0 --runs 5 --lr 0.01 --l2 0  --label_dropout 0.3  --feature_dropout 0.3 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY


# BUDDY - PPA, SP
python main_buddy.py --dataset ogbl-ppa_SP_00_017_026 --device 0 --runs 5 --lr 0.01 --l2 0  --label_dropout 0.1  --feature_dropout 0.1 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY

python main_buddy.py --dataset ogbl-ppa_SP_00_026_036 --device 0 --runs 5 --lr 0.001 --l2 0  --label_dropout 0.1  --feature_dropout 0.1 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY

python main_buddy.py --dataset ogbl-ppa_SP_026_017_00 --device 0 --runs 5 --lr 0.01 --l2 0  --label_dropout 0.1  --feature_dropout 0.1 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY

python main_buddy.py --dataset ogbl-ppa_SP_036_026_00 --device 0 --runs 5 --lr 0.001 --l2 0  --label_dropout 0.1  --feature_dropout 0.1 --hidden_channels 256  --epochs 100 --eval_steps 20 --kill_cnt 40 --batch_size 65536 --eval_batch_size 65536 --model BUDDY