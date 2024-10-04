import os
import json
import random
import shutil
import argparse
import logging
import torch
from PIL import Image
from vllm import Qwen2VLImageEditor
from sam import ImageMaskProcessor
from inpaint import FluxInpainter
from filter import ImageTextSimilarityScorer


def parse_arguments():
    """Parse command line arguments and return the configuration object."""
    parser = argparse.ArgumentParser(description="Image Inpainting Pipeline Configuration")

    # Directory configuration
    parser.add_argument('--root_dir', type=str, default='./', help='Root directory path')
    parser.add_argument('--split', type=str, default='split1', help='Data split name')
    parser.add_argument('--raw_image_dir', type=str, default='raw_image', help='Raw image directory')
    parser.add_argument('--instruction_dir', type=str, default='instruction', help='Instruction file directory')
    parser.add_argument('--mask_dir', type=str, default='mask', help='Mask file directory')
    parser.add_argument('--generated_dir', type=str, default='generated', help='Generated image directory')

    # Model and processing parameters
    parser.add_argument('--inpainter_model', type=str, default='black-forest-labs/FLUX.1-dev', help='Inpainting model path')
    parser.add_argument('--infer_steps_min', type=int, default=10, help='Minimum inference steps')
    parser.add_argument('--infer_steps_max', type=int, default=30, help='Maximum inference steps')
    parser.add_argument('--guidance_scale', type=float, default=3.5, help='Guidance scale')
    parser.add_argument('--strength_min', type=float, default=0.8, help='Minimum strength')
    parser.add_argument('--strength_max', type=float, default=1.0, help='Maximum strength')
    parser.add_argument('--expansion_pixels_min', type=int, default=3, help='Minimum mask expansion pixels')
    parser.add_argument('--expansion_pixels_max', type=int, default=12, help='Maximum mask expansion pixels')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to process')

    return parser.parse_args()


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("inpainting_pipeline.log"),
            logging.StreamHandler()
        ]
    )


def create_directories(config):
    """Create necessary directories if they don't exist."""
    directories = [
        os.path.join(config.root_dir, config.split, config.raw_image_dir),
        os.path.join(config.root_dir, config.split, config.instruction_dir),
        os.path.join(config.root_dir, config.split, config.mask_dir),
        os.path.join(config.root_dir, config.split, config.generated_dir)
    ]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Ensure directory exists: {dir_path}")


def initialize_processors(config):
    """Initialize the required processors and models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    editor = Qwen2VLImageEditor()
    processor = ImageMaskProcessor(device)
    inpainter = FluxInpainter(model_path=config.inpainter_model, device=device)
    scorer = ImageTextSimilarityScorer(device=device)

    logging.info("Initialization of all processors and models completed.")
    return editor, processor, inpainter, scorer


def get_image_list(config):
    """Get all image files in the raw image directory and randomly shuffle the order."""
    raw_image_path = os.path.join(config.root_dir, config.split, config.raw_image_dir)
    image_extensions = ('.png', '.jpg', '.jpeg')
    image_list = [
        f for f in os.listdir(raw_image_path)
        if f.lower().endswith(image_extensions)
    ]
    random.shuffle(image_list)
    logging.info(f"Found {len(image_list)} images to process.")
    return image_list


def load_edit_suggestion(editor, image_path, instruction_path):
    """Generate edit suggestion and save it as a JSON file."""
    edit_suggestion = editor.generate_edit_suggestion("", image_path)
    with open(instruction_path, 'w') as f:
        json.dump(edit_suggestion, f, indent=4)
    logging.info(f"Saved edit suggestion to {instruction_path}")
    return edit_suggestion


def create_mask(processor, image_path, area_to_replace, mask_path, expansion_pixels):
    """Generate mask based on edit suggestion and save it."""
    processor.save_mask_for_image(
        image_path,
        area_to_replace,
        mask_path,
        expansion_pixels=expansion_pixels
    )
    logging.info(f"Generated and saved mask to {mask_path}")
    return mask_path


def inpaint_image(inpainter, image_path, mask_path, prompt, config):
    """Perform image inpainting and return the list of generated image paths."""
    num_inference_steps = random.randint(config.infer_steps_min, config.infer_steps_max)
    strength = random.uniform(config.strength_min, config.strength_max)

    output_paths = inpainter(
        image_path=image_path,
        mask_path=mask_path,
        prompt=prompt,
        output_dir=os.path.join(config.root_dir, config.split, config.generated_dir),
        num_inference_steps=num_inference_steps,
        guidance_scale=config.guidance_scale,
        strength=strength
    )

    logging.info(f"Image inpainting completed, generated {len(output_paths)} images.")
    return output_paths


def evaluate_similarity(scorer, output_paths, edited_description):
    """Evaluate the similarity between generated images and the edited description, and rename files."""
    similarity_scores = scorer.process_batch(output_paths, edited_description)

    for score, old_path in zip(similarity_scores, output_paths):
        directory = os.path.dirname(old_path)
        old_filename = os.path.basename(old_path)
        base_name = old_filename.split('_')[0]
        new_filename = f"{base_name}_{score:.4f}.png"
        new_path = os.path.join(directory, new_filename)

        os.rename(old_path, new_path)
        logging.info(f"Renamed {old_path} to {new_path}, similarity score: {score:.4f}")


def process_image(image_name, config, editors_processors):
    """Process a single image."""
    editor, processor, inpainter, scorer = editors_processors
    image_path = os.path.join(config.root_dir, config.split, config.raw_image_dir, image_name)
    base_name = os.path.splitext(image_name)[0]

    # Check if the image has already been processed
    generated_dir = os.path.join(config.root_dir, config.split, config.generated_dir)
    existing_files = [
        f for f in os.listdir(generated_dir)
        if f.startswith(base_name)
    ]
    if existing_files:
        logging.info(f"Skipping {image_name}, already processed.")
        return

    logging.info(f"Start processing image: {image_name}")
    try:
        # Generate edit suggestion and save it
        instruction_path = os.path.join(config.root_dir, config.split, config.instruction_dir, f"{base_name}.json")
        edit_suggestion = load_edit_suggestion(editor, image_path, instruction_path)

        # Generate mask and save it
        area_to_replace = edit_suggestion.get("area_to_replace")
        mask_path = os.path.join(config.root_dir, config.split, config.mask_dir, f"{base_name}_mask.png")
        expansion_pixels = random.randint(config.expansion_pixels_min, config.expansion_pixels_max)
        mask_path = create_mask(processor, image_path, area_to_replace, mask_path, expansion_pixels)

        # Perform image inpainting
        prompt = edit_suggestion.get("new_object", "")
        output_paths = inpaint_image(inpainter, image_path, mask_path, prompt, config)

        # Evaluate similarity and rename
        edited_description = edit_suggestion.get("edited_description", "")
        evaluate_similarity(scorer, output_paths, edited_description)

    except Exception as e:
        logging.error(f"Error processing {image_name}: {e}", exc_info=True)


def main():
    """Main function to coordinate the entire image inpainting pipeline."""
    # Parse command line arguments
    config = parse_arguments()

    # Set up logging
    setup_logging()
    logging.info("Starting image inpainting pipeline.")

    # Create necessary directories
    create_directories(config)

    # Initialize processors and models
    editors_processors = initialize_processors(config)

    # Get image list
    image_list = get_image_list(config)

    # Limit the number of samples to process
    num_samples = min(config.num_samples, len(image_list))
    logging.info(f"Will process {num_samples} images.")

    # Process each image
    for image_name in image_list[:num_samples]:
        process_image(image_name, config, editors_processors)

    logging.info("All images processed.")


if __name__ == "__main__":
    main()
