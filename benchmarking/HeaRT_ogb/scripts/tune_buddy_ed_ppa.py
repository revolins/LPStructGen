import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='tuning script for GCN on all available splits and viable hyperparamaters')
    parser.add_argument('--data', type=str, default="ogbl-ppa")
    parser.add_argument('--device', type=str, default='0')
    args = parser.parse_args()

    edge_drop = ['0.1']

    split_type = [["ogbl-ppa_PA_0_15000_25000", "0.3", "0.001"],\
                    ["ogbl-ppa_PA_10000_5000_0", "0.3", "0.01"],\
                    ["ogbl-ppa_PA_20000_10000_0", "0.3", "0.01"],\
                    ["ogbl-ppa_PA_25000_15000_0", "0.3", "0.01"]]
    # ["ogbl-ppa_PA_0_5000_10000", "0.1", "0.01"],\
    #    ["ogbl-ppa_PA_0_10000_20000", "0.1", "0.01"],\

    cmd_list = ["python", "main_buddy.py", "--cache_subgraph_features", "--eval_batch_size", "65536", '--l2', '0',\
                 "--hidden_channels", "256", "--runs", "5", "--model", "BUDDY", "--num_negs", "1",\
            "--epochs", "100", "--kill_cnt", "40", "--eval_steps", "20",  "--batch_size", "65536", "--save", "--output_dir", "models"]

    for s in split_type:
        for ed in edge_drop:
            full_data_name = s[0]
            output_file = "output/BUDDY_" + full_data_name  + "_drp" + s[1] + "_lr" + s[2] + "_edgedrop0.1_output.txt"

            cmd_list.extend(["--dataset", full_data_name])
            cmd_list.extend(["--lr", s[2]])
            cmd_list.extend(["--feature_dropout", s[1]])
            cmd_list.extend(["--label_dropout", s[1]])
            cmd_list.extend(['--device', args.device])
            cmd_list.extend(['--edge_drop', "0.1"])

            with open(output_file, "w") as file:
                subprocess.run(cmd_list, stdout=file)
            file.close()

if __name__ == "__main__":
    main()