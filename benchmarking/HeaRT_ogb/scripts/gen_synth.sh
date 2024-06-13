#!/bin/bash 

python gen_synth.py --data_name ogbl-collab --valid_rat 50 --test_rat 100 --split_type pa

python gen_synth.py --data_name ogbl-collab --valid_rat 100 --test_rat 200 --split_type pa

python gen_synth.py --data_name ogbl-collab --valid_rat 150 --test_rat 250 --split_type pa

python gen_synth.py --data_name ogbl-collab --valid_rat 50 --test_rat 100 --split_type pa --inverse

python gen_synth.py --data_name ogbl-collab --valid_rat 100 --test_rat 200 --split_type pa --inverse

python gen_synth.py --data_name ogbl-collab --valid_rat 150 --test_rat 250 --split_type pa --inverse
