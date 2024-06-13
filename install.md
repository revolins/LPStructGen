# Installation

We detail the installation process using Conda on Linux. Note that all environments used can be found in the `envs` directory.


## 1. Install Conda
```
curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

## 2. Setup Code and Packages for Most Experiments

The system and CUDA version used for the majority of experiments are:
- Ubuntu 20.04.6 LTS
- CUDA 11.6
- Python 3.9.13

We first clone the repository.
```
git clone git@github.com:revolins/LPStructGen.git
cd LPStructGen
```

The package requirements can be found in the `benchmarking/HeaRT_ogb/environment.yml` file. Installing this will also create an environment for the project, `py39`. 
```
# Install environment requirements
conda env create -f benchmarking/HeaRT_ogb/environment.yml  

# Activate environment
conda activate py39
```
