#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=scavenger_8h100  #swarm_a100                   #scavenger_8h100
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=240000

eval "$(conda shell.bash hook)"
conda init bash
conda activate cl


export HF_HOME=/scratch/yw26g23/cache/


#python launch.py \
#    --mode "sd15" \
#    --root_dir "./" \
#    --split "00371" \
#    --raw_image_dir "raw_images" \
#    --instruction_dir "instructions" \
#    --mask_dir "masks" \
#    --generated_dir "generated" \
#    --infer_steps_min 27 \
#    --infer_steps_max 50 \
#    --guidance_scale 7.5 \
#    --strength_min 0.75 \
#    --strength_max 1.0 \
#    --expansion_pixels_min 3 \
#    --expansion_pixels_max 12 \
#    --num_samples 10000

#python launch.py \
#    --mode "sdxl" \
#    --root_dir "./" \
#    --split "00220" \
#    --raw_image_dir "raw_images" \
#    --instruction_dir "instructions" \
#    --mask_dir "masks" \
#    --generated_dir "generated" \
#    --infer_steps_min 27 \
#    --infer_steps_max 50 \
#    --guidance_scale 3.5 \
#    --strength_min 0.9 \
#    --strength_max 1.0 \
#    --expansion_pixels_min 3 \
#    --expansion_pixels_max 12 \
#    --num_samples 10000


#python launch.py \
#    --mode "flux" \
#    --root_dir "./" \
#    --split "00028" \
#    --raw_image_dir "raw_images" \
#    --instruction_dir "instructions" \
#    --mask_dir "masks" \
#    --generated_dir "generated" \
#    --infer_steps_min 27 \
#    --infer_steps_max 50 \
#    --guidance_scale 3.5 \
#    --strength_min 0.8 \
#    --strength_max 1.0 \
#    --expansion_pixels_min 3 \
#    --expansion_pixels_max 12 \
#    --num_samples 10000

#python launch.py \
#    --mode "sd3" \
#    --root_dir "./" \
#    --split "00199" \
#    --raw_image_dir "raw_images" \
#    --instruction_dir "instructions" \
#    --mask_dir "masks" \
#    --generated_dir "generated" \
#    --expansion_pixels_min 3 \
#    --expansion_pixels_max 12 \
#    --num_samples 10000

#
python launch.py \
    --mode "sd2" \
    --root_dir "./" \
    --split "00209" \
    --raw_image_dir "raw_images" \
    --instruction_dir "instructions" \
    --mask_dir "masks" \
    --generated_dir "generated" \
    --infer_steps_min 27 \
    --infer_steps_max 50 \
    --guidance_scale 7.5 \
    --strength_min 0.75 \
    --strength_max 1.0 \
    --expansion_pixels_min 3 \
    --expansion_pixels_max 12 \
    --num_samples 10000










