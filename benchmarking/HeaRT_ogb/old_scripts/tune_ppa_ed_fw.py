import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='tuning script for GCN on all available splits and viable hyperparamaters')
    parser.add_argument('--data', type=str, default="ogbl-ppa")
    parser.add_argument('--device', type=str, default='0')
    args = parser.parse_args()

    edge_drop = ['0.1', '0.3', '0.5']

    split_type = [["ogbl-ppa_SP_00_026_036", "0.1", "0.001"],\
                      ["ogbl-ppa_PA_00_022_034", "0.1", "0.01"],\
                   ["ogbl-ppa_PA_00_022_026", "0.1", "0.01"],\
                      ["ogbl-ppa_PA_00_026_034", "0.1", "0.01"]]

    cmd_list = ["python", "main_gnn.py", "--test_batch_size", "65536", "--gnn_model", "GCN", \
         "--num_layers", "3", "--hidden_channels", "128",  "--num_layers_predictor", "3", "--use_hard_samp",\
            "--epochs", "1000", "--kill_cnt", "100", "--eval_steps", "20",  "--batch_size", "65536", "--save", "--output_dir", "models"]

    for s in split_type:
        for ed in edge_drop:
            full_data_name = s[0]
            output_file = "output/GCN_" + full_data_name  + "_drp" + s[1] + "_lr" + s[2] + "_ed" + ed + "_heart_output.txt"

            cmd_list.extend(["--data_name", full_data_name])
            cmd_list.extend(["--lr", s[2]])
            cmd_list.extend(["--dropout", s[1]])
            cmd_list.extend(['--device', args.device])
            cmd_list.extend(['--edge_drop', ed])

            with open(output_file, "w") as file:
                subprocess.run(cmd_list, stdout=file)
            file.close()

if __name__ == "__main__":
    main()