import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='tuning script for BUDDY on all available splits and viable hyperparamaters')
    parser.add_argument('--data', type=str, default="ogbl-collab")
    parser.add_argument('--device', type=str, default='0')
    args = parser.parse_args()

    # collab --- 0.1,1e-2	0.3,1e-3 CN  3,5,0.1,0.001 -- SP 16,25,0.3,0.01 - 25,35,0.3,0.001
    # PA  -- 21,25,0.3,0.01 - 21,33,0.3,0.01 - 25,34,0.3,0.01
    # collab --- CN -- 2,1,0.1,0.001 - 4,2,0.1,0.001 - 5,3,0.1,0.001 -- SP - 25,17,0.1,0.01 - 36,25,0.1,0.001
    
    split_type = [["ogbl-ppa_CN_0_1_2", "0.1", "0.01"], ["ogbl-ppa_CN_0_2_4", "0.3", "0.001"],\
                 ["ogbl-ppa_CN_0_3_5", "0.3", "0.001"],["ogbl-ppa_CN_2_1_0", "0.1", "0.01"],\
                  ["ogbl-ppa_CN_4_2_0", "0.1", "0.01" ], ["ogbl-ppa_CN_5_3_0", "0.1", "0.01"],\
                  ['ogbl-ppa_SP_00_017_026', '0.1', '0.001'], ['ogbl-ppa_SP_00_026_036', '0.3', '0.01'], \
                    ['ogbl-ppa_SP_026_017_00', '0.3', '0.01'], ['ogbl-ppa_SP_036_026_00', '0.3', '0.01'],\
                ['ogbl-ppa_PA_00_022_026', '0.1', '0.01'], ['ogbl-ppa_PA_00_022_034', '0.1', '0.01'], \
                  ['ogbl-ppa_PA_00_026_034', '0.1', '0.001'], ['ogbl-ppa_PA_026_022_00', '0.1', '0.01'], \
                   ['ogbl-ppa_PA_034_022_00', '0.1', '0.01'], ['ogbl-ppa_PA_034_026_00', '0.3', '0.01'], ]
    
    cmd_list = ['python', 'main.py',\
          '--n_layers', '3', '--n_hidden', '128',\
            '--runs', '5', '--encoder', 'GCN', '--predictor', 'MLP',\
            '--epochs', '1000', '--model', 'GCN-aug', '--save', '--remove_rep', '--aug', '--gcn_denoise',\
            '--train']

    for i in split_type:
        full_data_name = i[0]
        output_file = "output/TC_" + full_data_name  + "_drp" + i[1] + "_lr" + i[2] + "_heart_output.txt"

        cmd_list.extend(["--dataset", full_data_name])
        cmd_list.extend(["--encoder_lr", i[2]])
        cmd_list.extend(["--predictor_lr", i[2]])
        cmd_list.extend(["--device", args.device])
        cmd_list.extend(["--en_dp", i[1]])

        with open(output_file, "w") as file:
            subprocess.run(cmd_list, stdout=file)
        file.close()

if __name__ == "__main__":
    main()