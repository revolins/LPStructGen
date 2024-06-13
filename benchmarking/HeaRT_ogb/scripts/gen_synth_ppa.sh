#!/bin/bash 

python gen_synth.py --data_name ogbl-ppa --valid_rat 5000 --test_rat 10000 --split_type pa

python gen_synth.py --data_name ogbl-ppa --valid_rat 10000 --test_rat 20000 --split_type pa

python gen_synth.py --data_name ogbl-ppa --valid_rat 15000 --test_rat 25000 --split_type pa