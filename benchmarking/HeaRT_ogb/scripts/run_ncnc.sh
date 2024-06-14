#!/bin/bash

# All relevant baseline NCNC commands for generating benchmark results on dataset splits tested in "Understanding the Generalizability of Link Predictors Under Distribution Shifts on Graphs".
# NOTE: These commands will need to be run in the benchmarking/HeaRT_ogb folder

# COLLAB, CN
python main_ncn.py --dataset ogbl-collab_CN_0_1_2 --gnnlr 0.01 --prelr 0.01 --predp 0.3 --gnndp 0.3 --xdp 0.3 --tdp 0.3 --predictor incn1cn1 --runs 5 --eval_steps 5 --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --depth 2 --splitsize 131072 --maskinput --res --use_xlin --tailact

python main_ncn.py --dataset ogbl-collab_CN_0_2_4 --gnnlr 0.001 --prelr 0.001 --predp 0.1 --gnndp 0.1 --xdp 0.1 --tdp 0.1 --predictor incn1cn1 --runs 5 --eval_steps 5 --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --depth 2 --splitsize 131072 --maskinput --res --use_xlin --tailact

python main_ncn.py --dataset ogbl-collab_CN_0_3_5 --gnnlr 0.001 --prelr 0.001 --predp 0.3 --gnndp 0.3 --xdp 0.3 --tdp 0.3 --predictor incn1cn1 --runs 5 --eval_steps 5 --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --depth 2 --splitsize 131072 --maskinput --res --use_xlin --tailact

python main_ncn.py --dataset ogbl-collab_CN_2_1_0 --gnnlr 0.001 --prelr 0.001 --predp 0.1 --gnndp 0.1 --xdp 0.1 --tdp 0.1 --predictor incn1cn1 --runs 5 --eval_steps 5 --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --depth 2 --splitsize 131072 --maskinput --res --use_xlin --tailact

python main_ncn.py --dataset ogbl-collab_CN_4_2_0 --gnnlr 0.001 --prelr 0.001 --predp 0.1 --gnndp 0.1 --xdp 0.1 --tdp 0.1 --predictor incn1cn1 --runs 5 --eval_steps 5 --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --depth 2 --splitsize 131072 --maskinput --res --use_xlin --tailact

python main_ncn.py --dataset ogbl-collab_CN_5_3_0 --gnnlr 0.01 --prelr 0.01 --predp 0.3 --gnndp 0.3 --xdp 0.3 --tdp 0.3 --predictor incn1cn1 --runs 5 --eval_steps 5 --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --depth 2 --splitsize 131072 --maskinput --res --use_xlin --tailact

# COLLAB, PA
python main_ncn.py --dataset ogbl-collab_PA_0_50_100 --gnnlr 0.001 --prelr 0.001 --predp 0.1 --gnndp 0.1 --xdp 0.1 --tdp 0.1 --predictor incn1cn1 --runs 5 --eval_steps 5 --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --depth 2 --splitsize 131072 --maskinput --res --use_xlin --tailact

python main_ncn.py --dataset ogbl-collab_PA_0_100_200 --gnnlr 0.01 --prelr 0.01 --predp 0.1 --gnndp 0.1 --xdp 0.1 --tdp 0.1 --predictor incn1cn1 --runs 5 --eval_steps 5 --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --depth 2 --splitsize 131072 --maskinput --res --use_xlin --tailact

python main_ncn.py --dataset ogbl-collab_PA_0_150_250 --gnnlr 0.01 --prelr 0.01 --predp 0.1 --gnndp 0.1 --xdp 0.1 --tdp 0.1 --predictor incn1cn1 --runs 5 --eval_steps 5 --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --depth 2 --splitsize 131072 --maskinput --res --use_xlin --tailact

python main_ncn.py --dataset ogbl-collab_PA_100_50_0 --gnnlr 0.001 --prelr 0.001 --predp 0.3 --gnndp 0.3 --xdp 0.3 --tdp 0.3 --predictor incn1cn1 --runs 5 --eval_steps 5 --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --depth 2 --splitsize 131072 --maskinput --res --use_xlin --tailact

python main_ncn.py --dataset ogbl-collab_PA_200_100_0 --gnnlr 0.001 --prelr 0.001 --predp 0.3 --gnndp 0.3 --xdp 0.3 --tdp 0.3 --predictor incn1cn1 --runs 5 --eval_steps 5 --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --depth 2 --splitsize 131072 --maskinput --res --use_xlin --tailact

python main_ncn.py --dataset ogbl-collab_PA_250_150_0 --gnnlr 0.001 --prelr 0.001 --predp 0.3 --gnndp 0.3 --xdp 0.3 --tdp 0.3 --predictor incn1cn1 --runs 5 --eval_steps 5 --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --depth 2 --splitsize 131072 --maskinput --res --use_xlin --tailact

# COLLAB, SP
python main_ncn.py --dataset ogbl-collab_SP_00_017_026 --gnnlr 0.001 --prelr 0.001 --predp 0.3 --gnndp 0.3 --xdp 0.3 --tdp 0.3 --predictor incn1cn1 --runs 5 --eval_steps 5 --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --depth 2 --splitsize 131072 --maskinput --res --use_xlin --tailact

python main_ncn.py --dataset ogbl-collab_SP_00_026_036 --gnnlr 0.001 --prelr 0.001 --predp 0.1 --gnndp 0.1 --xdp 0.1 --tdp 0.1 --predictor incn1cn1 --runs 5 --eval_steps 5 --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --depth 2 --splitsize 131072 --maskinput --res --use_xlin --tailact

python main_ncn.py --dataset ogbl-collab_SP_026_017_00 --gnnlr 0.01 --prelr 0.01 --predp 0.1 --gnndp 0.1 --xdp 0.1 --tdp 0.1 --predictor incn1cn1 --runs 5 --eval_steps 5 --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --depth 2 --splitsize 131072 --maskinput --res --use_xlin --tailact

python main_ncn.py --dataset ogbl-collab_SP_036_026_00 --gnnlr 0.001 --prelr 0.001 --predp 0.1 --gnndp 0.1 --xdp 0.1 --tdp 0.1 --predictor incn1cn1 --runs 5 --eval_steps 5 --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --depth 2 --splitsize 131072 --maskinput --res --use_xlin --tailact


# PPA, CN
python main_ncn.py --dataset ogbl-ppa_CN_0_1_2 --gnnlr 0.01 --prelr 0.01 --predp 0.1 --gnndp 0.1 --xdp 0.1 --tdp 0.1 --predictor incn1cn1 --runs 5 --eval_steps 5 --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --depth 2 --splitsize 131072 --maskinput --res --use_xlin --tailact

python main_ncn.py --dataset ogbl-ppa_CN_0_2_4 --gnnlr 0.01 --prelr 0.01 --predp 0.1 --gnndp 0.1 --xdp 0.1 --tdp 0.1 --predictor incn1cn1 --runs 5 --eval_steps 5 --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --depth 2 --splitsize 131072 --maskinput --res --use_xlin --tailact

python main_ncn.py --dataset ogbl-ppa_CN_0_3_5 --gnnlr 0.01 --prelr 0.01 --predp 0.1 --gnndp 0.1 --xdp 0.1 --tdp 0.1 --predictor incn1cn1 --runs 5 --eval_steps 5 --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --depth 2 --splitsize 131072 --maskinput --res --use_xlin --tailact

python main_ncn.py --dataset ogbl-ppa_CN_2_1_0 --gnnlr 0.01 --prelr 0.01 --predp 0.1 --gnndp 0.1 --xdp 0.1 --tdp 0.1 --predictor incn1cn1 --runs 5 --eval_steps 5 --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --depth 2 --splitsize 131072 --maskinput --res --use_xlin --tailact

python main_ncn.py --dataset ogbl-ppa_CN_4_2_0 --gnnlr 0.01 --prelr 0.01 --predp 0.1 --gnndp 0.1 --xdp 0.1 --tdp 0.1 --predictor incn1cn1 --runs 5 --eval_steps 5 --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --depth 2 --splitsize 131072 --maskinput --res --use_xlin --tailact

python main_ncn.py --dataset ogbl-ppa_CN_5_3_0 --gnnlr 0.01 --prelr 0.01 --predp 0.3 --gnndp 0.3 --xdp 0.3 --tdp 0.3 --predictor incn1cn1 --runs 5 --eval_steps 5 --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --depth 2 --splitsize 131072 --maskinput --res --use_xlin --tailact

# PPA, PA
python main_ncn.py --dataset ogbl-ppa_PA_0_5000_10000 --gnnlr 0.001 --prelr 0.001 --predp 0.1 --gnndp 0.1 --xdp 0.1 --tdp 0.1 --predictor incn1cn1 --runs 5 --eval_steps 5  --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --maskinput --res --use_xlin --tailact

python main_ncn.py --dataset ogbl-ppa_PA_0_10000_20000 --gnnlr 0.01 --prelr 0.01 --predp 0.1 --gnndp 0.1 --xdp 0.1 --tdp 0.1 --predictor incn1cn1 --runs 5 --eval_steps 5  --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --maskinput --res --use_xlin --tailact

python main_ncn.py --dataset ogbl-ppa_PA_0_15000_25000 --gnnlr 0.01 --prelr 0.01 --predp 0.1 --gnndp 0.1 --xdp 0.1 --tdp 0.1 --predictor incn1cn1 --runs 5 --eval_steps 5  --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --maskinput --res --use_xlin --tailact

python main_ncn.py --dataset ogbl-ppa_PA_10000_5000_0 --gnnlr 0.001 --prelr 0.001 --predp 0.1 --gnndp 0.1 --xdp 0.1 --tdp 0.1 --predictor incn1cn1 --runs 5 --eval_steps 5  --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --maskinput --res --use_xlin --tailact

python main_ncn.py --dataset ogbl-ppa_PA_20000_10000_0 --gnnlr 0.01 --prelr 0.01 --predp 0.1 --gnndp 0.1 --xdp 0.1 --tdp 0.1 --predictor incn1cn1 --runs 5 --eval_steps 5  --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --maskinput --res --use_xlin --tailact

python main_ncn.py --dataset ogbl-ppa_PA_25000_15000_0 --gnnlr 0.01 --prelr 0.01 --predp 0.1 --gnndp 0.1 --xdp 0.1 --tdp 0.1 --predictor incn1cn1 --runs 5 --eval_steps 5  --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --maskinput --res --use_xlin --tailact

# PPA, SP
python main_ncn.py --dataset ogbl-ppa_SP_00_017_026 --gnnlr 0.01 --prelr 0.01 --predp 0.1 --gnndp 0.1 --xdp 0.1 --tdp 0.1 --predictor incn1cn1 --runs 5 --eval_steps 5 --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --depth 2 --splitsize 131072 --maskinput --res --use_xlin --tailact

python main_ncn.py --dataset ogbl-ppa_SP_00_026_036 --gnnlr 0.01 --prelr 0.01 --predp 0.1 --gnndp 0.1 --xdp 0.1 --tdp 0.1 --predictor incn1cn1 --runs 5 --eval_steps 5 --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --depth 2 --splitsize 131072 --maskinput --res --use_xlin --tailact

python main_ncn.py --dataset ogbl-ppa_SP_026_017_00 --gnnlr 0.01 --prelr 0.01 --predp 0.1 --gnndp 0.1 --xdp 0.1 --tdp 0.1 --predictor incn1cn1 --runs 5 --eval_steps 5 --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --depth 2 --splitsize 131072 --maskinput --res --use_xlin --tailact

python main_ncn.py --dataset ogbl-ppa_SP_036_026_00 --gnnlr 0.01 --prelr 0.01 --predp 0.3 --gnndp 0.3 --xdp 0.3 --tdp 0.3 --predictor incn1cn1 --runs 5 --eval_steps 5 --device 0 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --mplayers 3  --hiddim 256  --epochs 15 --kill_cnt 10  --batch_size 16384  --ln --lnnn --model gcn  --testbs 4096 --depth 2 --splitsize 131072 --maskinput --res --use_xlin --tailact