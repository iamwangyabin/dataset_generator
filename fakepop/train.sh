#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=swarm_h100
#SBATCH --account=ecsstaff
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:3
#SBATCH --time=12:00:00

eval "$(conda shell.bash hook)"
conda init bash
conda activate cl

export HF_HOME=/scratch/yw26g23/cache/

python generator_flux.py


