import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='tuning script for GCN on all available splits and viable hyperparamaters')
    parser.add_argument('--data', type=str, default="ogbl-collab")
    parser.add_argument('--device', type=str, default='0')
    args = parser.parse_args()

    edge_drop = ['0.1', '0.3', '0.5']

    split_type = [["ogbl-collab_PA_0_50_100", "0.3", "0.001"],\
                   ["ogbl-collab_PA_0_100_200", "0.1", "0.001"],\
                    ["ogbl-collab_PA_0_150_250", "0.1", "0.001"],\
                    ["ogbl-collab_PA_100_50_0", "0.1", "0.001"],\
                    ["ogbl-collab_PA_200_100_0", "0.1", "0.001"],\
                    ["ogbl-collab_PA_250_150_0", "0.1", "0.001"]]
                #["ogbl-collab_CN_0_1_2", "0.3", "0.001"],\
                #   ["ogbl-collab_CN_0_2_4", "0.3", "0.001"],\
                #      ["ogbl-collab_CN_0_3_5", "0.3", "0.001"],\
                #      ["ogbl-collab_SP_00_017_026", "0.3", "0.001"],\
                #      ["ogbl-collab_SP_00_026_036", "0.1", "0.001"],\
                #["ogbl-collab_CN_2_1_0", "0.1", "0.01"],\
                #   ["ogbl-collab_CN_4_2_0", "0.3", "0.01"],\
                #      ["ogbl-collab_CN_5_3_0", "0.3", "0.01"],\
                #        ["ogbl-collab_SP_026_017_00", "0.3", "0.01"],\
                #      ["ogbl-collab_SP_036_026_00", "0.3", "0.01"],\

    cmd_list = ["python", "main_gnn.py", "--test_batch_size", "65536", "--gnn_model", "GCN", \
         "--num_layers", "3", "--hidden_channels", "128",  "--num_layers_predictor", "3", "--runs", "5",\
            "--epochs", "1000", "--kill_cnt", "100", "--eval_steps", "20",  "--batch_size", "65536", "--save", "--output_dir", "models"]

    for s in split_type:
        for ed in edge_drop:
            full_data_name = s[0]
            output_file = "output/GCN_" + full_data_name  + "_drp" + s[1] + "_lr" + s[2] + "_ed" + ed + "_output.txt"

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