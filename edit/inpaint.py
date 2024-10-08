
from diffusers.utils import load_image, check_min_version

from diffusers import (
    FluxInpaintPipeline,
    StableDiffusionInpaintPipeline,
    AutoPipelineForInpainting,
    StableDiffusion3ControlNetInpaintingPipeline
)
from diffusers.models.controlnet_sd3 import SD3ControlNetModel


import torch
from PIL import Image
import os
import random
from typing import Tuple
IMAGE_SIZE = 1920


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




class SD15Inpainter:
    def __init__(self, model_path="benjamin-paine/stable-diffusion-v1-5-inpainting", device="cuda"):
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(model_path, variant="fp16", torch_dtype=torch.float16,
                                                                   safety_checker=None,
                                                                   requires_safety_checker=False
                                                                   )
        self.pipe = self.pipe.to(device)

    def __call__(self, image_path, mask_path, prompt, output_dir,
                 num_inference_steps=30, guidance_scale=7.5, strength=1):
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




class SDXLInpainter:
    def __init__(self, model_path="diffusers/stable-diffusion-xl-1.0-inpainting-0.1", device="cuda"):
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            model_path,
            variant="fp16",
            torch_dtype=torch.float16
        ).to(device)

    def __call__(self, image_path, mask_path, prompt, output_dir,
                 num_inference_steps=20, guidance_scale=8.0, strength=0.99):
        init_image = Image.open(image_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("RGB")

        # SDXL typically uses 1024x1024 resolution
        width, height = resize_image_dimensions(original_resolution_wh=init_image.size, factor=8)
        init_image = init_image.resize((width, height), Image.LANCZOS)
        mask_image = mask_image.resize((width, height), Image.LANCZOS)

        # Ensure the mask is black and white
        mask_image = mask_image.convert("L")
        mask_image = mask_image.convert("RGB")

        # Apply blur to mask (Note: SDXL might handle this internally, but we'll keep it for consistency)
        blur_factor = random.randint(10, 30)
        blurred_mask = self.pipe.mask_processor.blur(mask_image, blur_factor=blur_factor)

        generator = torch.Generator(device="cuda").manual_seed(0)

        # Call the pipeline
        output = self.pipe(
            prompt=prompt,
            image=init_image,
            mask_image=blurred_mask,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            strength=strength,
            generator=generator,
            num_images_per_prompt=2,
        )

        image_name = os.path.basename(image_path)
        image_name_without_extension = os.path.splitext(image_name)[0]
        saved_paths = []
        for i, image in enumerate(output.images):
            output_path = os.path.join(output_dir, f"{image_name_without_extension}_{i}.png")
            image.save(output_path)
            saved_paths.append(output_path)

        return saved_paths



class SD3CNInpainter:
    def __init__(self, model_path="stabilityai/stable-diffusion-3-medium-diffusers", device="cuda"):
        controlnet = SD3ControlNetModel.from_pretrained(
            "alimama-creative/SD3-Controlnet-Inpainting", use_safetensors=True, extra_conditioning_channels=1
        )
        self.pipe = StableDiffusion3ControlNetInpaintingPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            controlnet=controlnet,
            torch_dtype=torch.float16,
        ).to(device)
        self.pipe.text_encoder.to(torch.float16)
        self.pipe.controlnet.to(torch.float16)


    def __call__(self, image_path, mask_path, prompt, output_dir,
                 num_inference_steps=20, guidance_scale=8.0, strength=0.99):
        init_image = Image.open(image_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("RGB")

        width, height = resize_image_dimensions(original_resolution_wh=init_image.size, factor=8)
        init_image = init_image.resize((width, height), Image.LANCZOS)
        mask_image = mask_image.resize((width, height), Image.LANCZOS)

        # Ensure the mask is black and white
        mask_image = mask_image.convert("L")
        mask_image = mask_image.convert("RGB")

        # Apply blur to mask (Note: SDXL might handle this internally, but we'll keep it for consistency)
        blur_factor = random.randint(10, 30)
        blurred_mask = self.pipe.mask_processor.blur(mask_image, blur_factor=blur_factor)

        generator = torch.Generator(device="cuda").manual_seed(0)

        # Call the pipeline
        output = self.pipe(
            negative_prompt="deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW",
            image=init_image,
            prompt=prompt,
            height=height,
            width=width,
            control_image=init_image,
            control_mask=blurred_mask,
            num_inference_steps=28,
            generator=generator,
            controlnet_conditioning_scale=0.95,
            guidance_scale=7,
            num_images_per_prompt=2,
        )

        image_name = os.path.basename(image_path)
        image_name_without_extension = os.path.splitext(image_name)[0]
        saved_paths = []
        for i, image in enumerate(output.images):
            output_path = os.path.join(output_dir, f"{image_name_without_extension}_{i}.png")
            image.save(output_path)
            saved_paths.append(output_path)

        return saved_paths

