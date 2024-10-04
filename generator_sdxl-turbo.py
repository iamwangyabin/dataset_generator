import os
import numpy as np
import random
import requests
import base64
from tqdm import tqdm
import json

with open('coco_width.json', 'r') as json_file:
    file_name2width = json.load(json_file)

with open('coco_height.json', 'r') as json_file:
    file_name2height = json.load(json_file)

with open('coco_captions.json', 'r') as json_file:
    file_name2captions = json.load(json_file)    



import torch
from diffusers import AutoPipelineForText2Image

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", 
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker = None,
            requires_safety_checker = False)
pipe.to("cuda")

output_dir = 'stable-diffusion-xl-turbo'
os.makedirs(output_dir, exist_ok=True)


def make_divisible_by_8(value):
    return (value // 8) * 8

for file_path in tqdm(list(file_name2captions.keys())):
    output_file = os.path.join(output_dir, file_path.split('.')[0] + '_0.png')
    if os.path.exists(output_file):
        continue
    else:
        promp = file_name2captions[file_path][0]
        aspect_ratio = file_name2width[file_path]/file_name2height[file_path]
        ran_size = 512
        total_pixels = ran_size*ran_size
        width = int(np.sqrt(total_pixels * aspect_ratio))
        height = int(total_pixels / width)
        images = pipe(prompt=promp,
                    height=make_divisible_by_8(height), width=make_divisible_by_8(width),
                    guidance_scale=5.0, num_images_per_prompt=1,
                    ).images
        for idx, img in enumerate(images):
            img.save(os.path.join(output_dir, file_path.split('.')[0] + f'_{idx}.png'))
