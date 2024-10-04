from diffusers import FluxInpaintPipeline
import torch
from PIL import Image
import os
import random
from typing import Tuple
IMAGE_SIZE = 1920


'''
How to use the FluxInpainter class:

1. Import required libraries:
   Ensure you have the diffusers library installed and torch available.

2. Initialize the FluxInpainter:
   inpainter = FluxInpainter(model_path="black-forest-labs/FLUX.1-dev", device="cuda")
   This loads the FLUX model and prepares it for inpainting. You can specify a different model path if needed.

3. Prepare your images:
   - Have the original image file ready (the image you want to edit).
   - Create a mask image where the area to be inpainted is white and the rest is black.

4. Call the inpainter:
   inpainted_image = inpainter(
       image_path="path/to/original/image.jpg",
       mask_path="path/to/mask/image.png",
       prompt="Description of what you want to inpaint",
       output_path="path/to/save/result.png",
       num_inference_steps=50,
       guidance_scale=3.5,
       strength=0.8
   )

Parameters:
- image_path: Path to the original image file.
- mask_path: Path to the mask image file.
- prompt: Text description of what you want to inpaint.
- output_path: Where to save the resulting inpainted image.
- num_inference_steps: Number of denoising steps (default 50, increase for potentially better quality).
- guidance_scale: Controls how closely the model follows the prompt (default 3.5).
- strength: Controls the extent of inpainting (default 0.8, range 0-1).

5. The inpainted image will be saved to the specified output_path and also returned by the function.

Notes:
- The mask is automatically blurred to create smoother transitions.
- Adjust num_inference_steps, guidance_scale, and strength to fine-tune results.
- Ensure you have sufficient GPU memory, as FLUX models can be resource-intensive.
- The quality of inpainting depends on the prompt, mask, and parameter settings.

This class provides a straightforward interface for inpainting using the FLUX model, 
suitable for integration into larger image processing pipelines or for standalone use.
'''

def resize_image_dimensions(
    original_resolution_wh: Tuple[int, int],
    factor: int
) -> Tuple[int, int]:
    width, height = original_resolution_wh
    maximum_dimension = IMAGE_SIZE

    if width > height:
        scaling_factor = maximum_dimension / width
    else:
        scaling_factor = maximum_dimension / height

    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)

    new_width = new_width - (new_width % factor)
    new_height = new_height - (new_height % factor)

    return new_width, new_height


class FluxInpainter:
    def __init__(self, model_path="black-forest-labs/FLUX.1-dev", device="cuda"):
        self.pipe = FluxInpaintPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(device)

    def __call__(self, image_path, mask_path, prompt, output_dir,
                 num_inference_steps=30, guidance_scale=3.5, strength=1):
        init_image = Image.open(image_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("RGB")

        width, height = resize_image_dimensions(original_resolution_wh=init_image.size, factor=8)
        init_image = init_image.resize((width, height), Image.LANCZOS)
        mask_image = mask_image.resize((width, height), Image.LANCZOS)

        # Ensure the mask is black and white
        mask_image = mask_image.convert("L")
        mask_image = mask_image.convert("RGB")
        blur_factor = random.randint(10, 30)
        blurred_mask = self.pipe.mask_processor.blur(mask_image, blur_factor=blur_factor)

        # Call the pipeline
        output = self.pipe(
            prompt=prompt,
            image=init_image,
            mask_image=blurred_mask,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            num_images_per_prompt=2,
        )

        image_name = os.path.basename(image_path)
        image_name_without_extension = os.path.splitext(image_name)[0]
        saved_paths = []
        for i, image in enumerate(output.images):
            output_path = os.path.join(output_dir, f"{image_name_without_extension}_{i}.png")
            image.save(output_path)
            saved_paths.append(output_path)  # Store the path

        return saved_paths














# import torch
# from diffusers import StableDiffusionInpaintPipeline
# from PIL import Image
# import numpy as np
#
# def inpaint_image(
#     image_path: str,
#     mask_path: str,
#     prompt: str,
#     output_path: str,
#     num_inference_steps: int = 50,
#     guidance_scale: float = 7.5,
# ):
#     # Load the inpainting model
#     pipe = StableDiffusionInpaintPipeline.from_pretrained(
#         "benjamin-paine/stable-diffusion-v1-5-inpainting",
#         torch_dtype=torch.float16,
#     )
#     pipe = pipe.to("cuda")
#
#     # Load the original image and the mask
#     init_image = Image.open(image_path).convert("RGB")
#     mask_image = Image.open(mask_path).convert("RGB")
#
#     # Ensure the mask is black and white
#     mask_image = mask_image.convert("L")
#     mask_image = mask_image.convert("RGB")
#     blurred_mask = pipe.mask_processor.blur(mask_image, blur_factor=10)
#
#     # Perform inpainting
#     image = pipe(
#         prompt=prompt,
#         image=init_image,
#         mask_image=blurred_mask,
#         num_inference_steps=num_inference_steps,
#         guidance_scale=guidance_scale,
#     ).images[0]
#
#     # Save the result
#     image.save(output_path)

















