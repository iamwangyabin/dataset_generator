import argparse
import os
import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKL

def load_vae():
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae.eval()
    return vae

def process_image(image_path, vae, output_folder):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
    image = image.cuda()
    # Encode and decode image
    with torch.no_grad():
        latents = vae.encode(image).latent_dist.sample()
        decoded_image = vae.decode(latents).sample.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

    # Save decoded image
    decoded_image = np.clip(decoded_image * 255, 0, 255).astype(np.uint8)
    decoded_image = Image.fromarray(decoded_image)

    # Generate output path
    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_ae.png")
    decoded_image.save(output_path)
    print(f"Saved processed image to {output_path}")

def main(input_folder, output_folder):
    # Load the VAE model
    vae = load_vae()
    vae = vae.cuda()

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".jpeg", ".png")):  # Only process image files
            image_path = os.path.join(input_folder, filename)
            process_image(image_path, vae, output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images from input folder and save them to output folder.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing images.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder to save processed images.")
    
    args = parser.parse_args()
    main(args.input_folder, args.output_folder)
