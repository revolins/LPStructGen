#!/bin/bash

# ogbl-collab

### ogbl-collab_CN_0_1_2
### Example script with ogbl-collab, using the CN heuristic, with the following thresholds, train=0, valid=1, test=2:
python gen_synth.py --data_name ogbl-collab --valid_rat 1 --test_rat 2 --split_type cn

### ogbl-collab_CN_0_2_4
python gen_synth.py --data_name ogbl-collab --valid_rat 2 --test_rat 4 --split_type cn

### ogbl-collab_CN_0_3_5
python gen_synth.py --data_name ogbl-collab --valid_rat 3 --test_rat 5 --split_type cn

### ogbl-collab_CN_2_1_0
python gen_synth.py --data_name ogbl-collab --valid_rat 1 --test_rat 2 --split_type cn --inverse

### ogbl-collab_CN_4_2_0
python gen_synth.py --data_name ogbl-collab --valid_rat 2 --test_rat 4 --split_type cn --inverse

### ogbl-collab_CN_5_3_0
python gen_synth.py --data_name ogbl-collab --valid_rat 3 --test_rat 5 --split_type cn --inverse

## ogbl-collab - PA

### ogbl-collab_PA_0_50_100
python gen_synth.py --data_name ogbl-collab --valid_rat 50 --test_rat 100 --split_type pa

### ogbl-collab_PA_0_100_200
python gen_synth.py --data_name ogbl-collab --valid_rat 100 --test_rat 200 --split_type pa

### ogbl-collab_PA_0_150_250
python gen_synth.py --data_name ogbl-collab --valid_rat 150 --test_rat 250 --split_type pa

### ogbl-collab_PA_100_50_0
python gen_synth.py --data_name ogbl-collab --valid_rat 50 --test_rat 100 --split_type pa --inverse

### ogbl-collab_PA_200_100_200
python gen_synth.py --data_name ogbl-collab --valid_rat 100 --test_rat 200 --split_type pa --inverse

### ogbl-collab_PA_250_150_0
python gen_synth.py --data_name ogbl-collab --valid_rat 150 --test_rat 250 --split_type pa --inverse

## ogbl-collab - SP

### ogbl-collab_SP_00_017_026
python gen_synth.py --data_name ogbl-collab --valid_rat 0.17 --test_rat 0.26 --split_type sp

### ogbl-collab_SP_00_026_036
python gen_synth.py --data_name ogbl-collab --valid_rat 0.26 --test_rat 0.36 --split_type sp

### ogbl-collab_SP_026_017_00
python gen_synth.py --data_name ogbl-collab --valid_rat 0.17 --test_rat 0.26 --split_type sp --inverse

### ogbl-collab_SP_036_026_00
python gen_synth.py --data_name ogbl-collab --valid_rat 0.26 --test_rat 0.36 --split_type sp --inverse


# ogbl-ppa


### ogbl-ppa_CN_0_1_2
### Example script with ogbl-ppa, using the CN heuristic, with the following thresholds, train=0, valid=1, test=2:
python gen_synth.py --data_name ogbl-ppa --valid_rat 1 --test_rat 2 --split_type cn

### ogbl-ppa_CN_0_2_4
python gen_synth.py --data_name ogbl-ppa --valid_rat 2 --test_rat 4 --split_type cn

### ogbl-ppa_CN_0_3_5
python gen_synth.py --data_name ogbl-ppa --valid_rat 3 --test_rat 5 --split_type cn

### ogbl-ppa_CN_2_1_0
python gen_synth.py --data_name ogbl-ppa --valid_rat 1 --test_rat 2 --split_type cn --inverse

### ogbl-ppa_CN_4_2_0
python gen_synth.py --data_name ogbl-ppa --valid_rat 2 --test_rat 4 --split_type cn --inverse

### ogbl-ppa_CN_5_3_0
python gen_synth.py --data_name ogbl-ppa --valid_rat 3 --test_rat 5 --split_type cn --inverse

## oogbl-ppa - PA

### ogbl-ppa_PA_0_5000_10000
python gen_synth.py --data_name ogbl-ppa --valid_rat 5000 --test_rat 10000 --split_type pa

### ogbl-ppa_PA_0_10000_20000
python gen_synth.py --data_name ogbl-ppa --valid_rat 10000 --test_rat 20000 --split_type pa

### ogbl-ppa_PA_0_15000_25000
python gen_synth.py --data_name ogbl-ppa --valid_rat 15000 --test_rat 25000 --split_type pa

### ogbl-ppa_PA_10000_5000_0
python gen_synth.py --data_name ogbl-ppa --valid_rat 5000 --test_rat 10000 --split_type pa --inverse

### ogbl-ppa_PA_20000_10000_0
python gen_synth.py --data_name ogbl-ppa --valid_rat 10000 --test_rat 20000 --split_type pa --inverse

### ogbl-ppa_PA_25000_15000_0
python gen_synth.py --data_name ogbl-ppa --valid_rat 15000 --test_rat 25000 --split_type pa --inverse

## ogbl-ppa - SP

### ogbl-ppa_SP_00_017_026
python gen_synth.py --data_name ogbl-ppa --valid_rat 0.17 --test_rat 0.26 --split_type sp

### ogbl-ppa_SP_00_026_036
python gen_synth.py --data_name ogbl-ppa --valid_rat 0.26 --test_rat 0.36 --split_type sp

### ogbl-ppa_SP_026_017_00
python gen_synth.py --data_name ogbl-ppa --valid_rat 0.17 --test_rat 0.26 --split_type sp --inverse

### ogbl-ppa_SP_036_026_00
python gen_synth.py --data_name ogbl-ppa --valid_rat 0.26 --test_rat 0.36 --split_type sp --inverse