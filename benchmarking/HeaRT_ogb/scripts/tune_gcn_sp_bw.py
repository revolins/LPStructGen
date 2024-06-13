import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='tuning script for GCN on all available splits and viable hyperparamaters')
    parser.add_argument('--data', type=str, default="ogbl-collab")
    parser.add_argument('--device', type=str, default='0')
    args = parser.parse_args()

    data_name = ['ogbl-ppa'] #'ogbl-ppa']
    lr = ['0.001', '0.01']
    dropout = ['0.1', '0.3']
    split_type = ["SP_036_026_00", "SP_026_017_00"] #"PA_034_026_00","PA_034_022_00", "PA_026_022_00"
    #["CN_0_1_2", "CN_0_2_4", "CN_0_3_5", "CN_2_1_0", "CN_4_2_0", "CN_5_3_0"]
    #["SP_00_017_026",  "SP_00_026_036","SP_036_026_00", "SP_026_017_00"]
    #edgedrop_list = ['0.1', '0.3', '0.5']

    cmd_list = ["python", "main_gnn.py", "--test_batch_size", "65536", "--gnn_model", "GCN", \
         "--num_layers", "3", "--hidden_channels", "128",  "--num_layers_predictor", "3", "--use_hard_samp",\
            "--epochs", "1000", "--kill_cnt", "100", "--eval_steps", "20",  "--batch_size", "65536", "--save", "--output_dir", "models"]

    for dn in data_name:
        for l in lr:
            for d in dropout:
                for s in split_type:
                    full_data_name = dn + "_" + s
                    output_file = "output/GCN_" + full_data_name  + "_drp" + d + "_lr" + l + "_heart_output.txt"

                    cmd_list.extend(["--data_name", full_data_name])
                    cmd_list.extend(["--lr", l])
                    cmd_list.extend(["--dropout", d])
                    cmd_list.extend(['--device', args.device])
                    #cmd_list.extend(['--edge_drop', ed])

                    with open(output_file, "w") as file:
                        subprocess.run(cmd_list, stdout=file)
                    file.close()

if __name__ == "__main__":
    main()