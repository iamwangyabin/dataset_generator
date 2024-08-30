#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=scavenger_8h100
#SBATCH --account=ecsstaff
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00

eval "$(conda shell.bash hook)"
conda init bash
conda activate cl

export HF_HOME=/scratch/yw26g23/cache/

python generator_sd3_mp.py


