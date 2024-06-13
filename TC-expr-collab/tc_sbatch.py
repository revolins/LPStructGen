import subprocess
import argparse
import os

def run_slurm(args):

    toy_str = "python main.py --device 0 --dataset='ogbl-collab_CN_0_1_2' --n_layers=3 --n_hidden=128 --en_dp=0.3 \
        --encoder_lr=0.001 --predictor_lr=0.001 --runs=5 --encoder='GCN' --predictor='MLP' --epochs=1000 \
            --model='GCN-aug' --save --remove_rep --aug --gcn_denoise --train"

    part_cmd = "python main.py --device 0 --n_layers=3 --n_hidden=128 --runs=5 --encoder='GCN' --predictor='MLP' --epochs=1000 --model='GCN-aug' --save --remove_rep --aug --gcn_denoise --train"
    #cmd_list = [["ogbl-collab_CN_0_1_2", "0.3", "0.001"]]#, ["ogbl-collab_CN_0_2_4", "0.1", "0.001"], ["ogbl-collab_CN_0_3_5", "0.3", "0.001"],\
    #["ogbl-collab_SP_00_017_026", "0.3", "0.01"], ["ogbl-collab_SP_00_026_036", "0.3", "0.01"],\
    #["ogbl-collab_PA_00_022_026", "0.1", "0.001" ], ["ogbl-collab_PA_00_022_034", "0.3", "0.001"], ["ogbl-collab_PA_00_026_034", "0.1", "0.001"],\
    cmd_list = [["ogbl-collab_CN_2_1_0", "0.1", "0.01"], ["ogbl-collab_CN_4_2_0", "0.1", "0.01" ], ["ogbl-collab_CN_5_3_0", "0.3", "0.001"],\
                  ["ogbl-collab_SP_026_017_00", "0.3", "0.01"], ["ogbl-collab_SP_036_026_00", "0.3", "0.01"],\
                  ["ogbl-collab_PA_026_022_00", "0.1", "0.001"], ["ogbl-collab_PA_034_022_00", "0.3", "0.001"], ["ogbl-collab_PA_034_026_00", "0.3", "0.001"]]
    for split_type in cmd_list:
        format_cmd = ''
        output_folder=f'{split_type[0]}split_{split_type[1]}dp_{split_type[2]}lr'
        if os.path.exists("run_exp.sb"):
            os.remove("run_exp.sb")
        with open('run_exp.sb', 'w+') as f:
            f.seek(0)
            f.write('#!/bin/bash --login\n')

            f.write(f'#SBATCH --job-name={output_folder}_TC\n')
            f.write('#SBATCH --nodes=1\n')
            f.write('#SBATCH --cpus-per-task=32\n')
            f.write('#SBATCH --mem-per-cpu=1G\n')
            f.write('#SBATCH --gpus=v100s:1\n')
            f.write(f'#SBATCH --time=01:00:00\n')
            f.write('#SBATCH --mail-type=END\n')
            f.write(f'#SBATCH --mail-user=revolins@msu.edu\n')
            f.write('#SBATCH --output=%x-%j.SLURMout\n')

            f.write('module load Conda/3\n')
            f.write('conda activate heart_env\n')
            format_cmd = f' --dataset={split_type[0]} --en_dp={split_type[1]} --encoder_lr={split_type[2]} --predictor_lr={split_type[2]}'
            f.write(part_cmd + format_cmd + '\n')

            f.write('scontrol show job $SLURM_JOB_ID\n')
            f.write('js -j $SLURM_JOB_ID\n')
        f.close()
        if args.run:
            subprocess.run(['sbatch', 'run_exp.sb'])
        else:
            assert args.run, "type 'sbatch run_exp.sb' to execute current experiment. \n\
            WARNING: run_exp.sb will be overwritten if slurm.py is executed again.\n\
            WARNING: double check experiment settings in run_exp.sb"

def main():
    arg_parser = argparse.ArgumentParser(
        description='SLURM Interface Script for submitting Topological Concentration Jobs')
    arg_parser.add_argument("--run", "--r", action='store_true', default=False, help="(bool) (DEFAULT=False) Set to automatically submit SLURM job")
    args = arg_parser.parse_args()  

    run_slurm(args)
    
if __name__ == "__main__":
    main()