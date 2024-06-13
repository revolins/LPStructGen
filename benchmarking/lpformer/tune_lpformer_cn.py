import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='tuning script for NCN on all available splits and viable hyperparamaters')
    parser.add_argument('--data', type=str, default="ogbl-collab")
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--runs', type=str, default='1')
    args = parser.parse_args()

    data_name = ['ogbl-collab']
    split_type = ["CN_0_1_2", "CN_0_2_4", "CN_0_3_5", "CN_2_1_0", "CN_4_2_0", "CN_5_3_0"]

    cmd_list = ["bash", "tune_lpformer.sh"]

    for dn in data_name:
        for s in split_type:
            full_data_name = dn + "_" + s
            output_file = "output/lpformer_" + full_data_name  + "_heart_output.txt"

            cmd_list.extend([full_data_name, args.device, args.runs])

            with open(output_file, "a") as file:
                subprocess.run(cmd_list, stdout=file)
            file.close()

if __name__ == "__main__":
    main()