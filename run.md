# Generating Results

In order to reproduce results for specific experiments, we recommend running a command from the scripts stored in the following folders:

* Current scripts for reproducing the: tuned baselines, EdgeDrop, Edge Proposal Set, and emd results are available in 'benchmarking/HeaRT_ogb/scripts' folder and should be run in the 'benchmarking/HeaRT_ogb' folder.
* TC scripts are available in 'LPStructGen/TC-expr-collab' as 'tc_buddy.sh' and 'tc_gcn.sh', and should be run in the 'LPStructGen/TC-expr-collab' folder.
* LPFormer scripts are stored in 'benchmarking/lpformer' as 'run_lpfrmr.sh', and should be run in the 'benchmarking/lpformer' folder.
* To run EdgeDrop for BUDDY, simply append '--edge_drop 0.1' to a command that corresponds to the dataset that you wish to test, all relevant scripts can be pulled from the 'benchmarking/HeaRT_ogb/scripts/run_buddy.sh' file.

# All commands can be run as-is directly on the command-line within the appropriate folder. Please open an issue for questions or if a command fails to run.

For example, the entire GCN experiment with Topological Concentration can be run with:

```
cd TC-expr-collab
bash tc_gcn.sh
```

Or any command can be extracted from the tc_gcn.sh to run a single test, within the 'TC-expr-collab' folder:

```
cd TC-expr-collab
python main.py --dataset ogbl-collab_CN_5_3_0 --encoder_lr 0.01 --predictor_lr 0.01 --en_dp 0.3 --n_layers 3 --n_hidden 128 --runs 5 --encoder GCN --predictor MLP --epochs 1000 --model GCN-aug --remove_rep --aug --gcn_denoise --train --device 0
```

