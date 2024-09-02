import os
import numpy as np
import random
import requests
import base64
from tqdm import tqdm
import json
import torch
from diffusers import DiffusionPipeline
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline


with open('chunk_1.0.json', 'r') as json_file:
    file1 = json.load(json_file)

with open('chunk_2.0.json', 'r') as json_file:
    file2 = json.load(json_file)

with open('chunk_3.0.json', 'r') as json_file:
    file3 = json.load(json_file)

combined_dict = {}
combined_dict.update(file1)
combined_dict.update(file2)
combined_dict.update(file3)

prior = StableCascadePriorPipeline.from_pretrained(
    "stabilityai/stable-cascade-prior", 
    variant="bf16", torch_dtype=torch.bfloat16,
    safety_checker = None,
    requires_safety_checker = False
    )

decoder = StableCascadeDecoderPipeline.from_pretrained(
    "stabilityai/stable-cascade", 
    variant="bf16", torch_dtype=torch.float16,
    safety_checker = None,
    requires_safety_checker = False
    )

prior = prior.to('cuda')
decoder = decoder.to('cuda')

negative_prompt=""
output_dir = 'stable-cascade'
os.makedirs(output_dir, exist_ok=True)

def make_divisible_by_8(value):
    return (value // 8) * 8

file_paths = list(combined_dict.keys())[:3000]
random.shuffle(file_paths)

for file_path in tqdm(file_paths):
    output_file = os.path.join(output_dir, file_path.split('.')[0] + '_0.png')
    if os.path.exists(output_file):
        continue
    else:
        promp = combined_dict[file_path]['cogvlm_caption']
        aspect_ratio = combined_dict[file_path]['width']/combined_dict[file_path]['height']
        ran_size = 1024
        total_pixels = ran_size*ran_size
        width = int(np.sqrt(total_pixels * aspect_ratio))
        height = int(total_pixels / width)
        prior_output = prior(
            prompt=prompt,
            height=make_divisible_by_8(height),
            width=make_divisible_by_8(width),
            negative_prompt=negative_prompt,
            guidance_scale=4.0,
            num_images_per_prompt=1,
            num_inference_steps=20
        )
        images = decoder(
            image_embeddings=prior_output.image_embeddings.to(torch.float16),
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=0.0,
            num_inference_steps=10
        ).images

        for idx, img in enumerate(images):
            img.save(os.path.join(output_dir, file_path.split('.')[0] + f'_{idx}.png'))

