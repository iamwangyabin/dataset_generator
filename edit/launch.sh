#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=a100
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00

eval "$(conda shell.bash hook)"
conda init bash
conda activate cl

export HF_HOME=/scratch/yw26g23/cache/

# Set default values
ROOT_DIR="./"
SPLIT="split1"
RAW_IMAGE_DIR="raw_images"
INSTRUCTION_DIR="instructions"
MASK_DIR="masks"
GENERATED_DIR="generated"
INPAINTER_MODEL="black-forest-labs/FLUX.1-dev"
INFER_STEPS_MIN=10
INFER_STEPS_MAX=30
GUIDANCE_SCALE=3.5
STRENGTH_MIN=0.8
STRENGTH_MAX=1.0
EXPANSION_PIXELS_MIN=3
EXPANSION_PIXELS_MAX=12
NUM_SAMPLES=5000

python launch.py \
    --root_dir "$ROOT_DIR" \
    --split "$SPLIT" \
    --raw_image_dir "$RAW_IMAGE_DIR" \
    --instruction_dir "$INSTRUCTION_DIR" \
    --mask_dir "$MASK_DIR" \
    --generated_dir "$GENERATED_DIR" \
    --inpainter_model "$INPAINTER_MODEL" \
    --infer_steps_min $INFER_STEPS_MIN \
    --infer_steps_max $INFER_STEPS_MAX \
    --guidance_scale $GUIDANCE_SCALE \
    --strength_min $STRENGTH_MIN \
    --strength_max $STRENGTH_MAX \
    --expansion_pixels_min $EXPANSION_PIXELS_MIN \
    --expansion_pixels_max $EXPANSION_PIXELS_MAX \
    --num_samples $NUM_SAMPLES
