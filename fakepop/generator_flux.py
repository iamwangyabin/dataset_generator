import os
import numpy as np
import random
import requests
import base64
from tqdm import tqdm
import json
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16,
    use_safetensors=True,
    safety_checker = None,
    requires_safety_checker = False,
)

pipe = pipe.to('cuda')

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

output_dir = 'FLUX'
os.makedirs(output_dir, exist_ok=True)


def make_divisible_by_8(value):
    return (value // 8) * 8

for file_path in tqdm(list(combined_dict.keys())):
    output_file = os.path.join(output_dir, file_path + '.png')
    if os.path.exists(output_file):
        continue
    else:
        promp = combined_dict[file_path]['cogvlm_caption']
        aspect_ratio = combined_dict[file_path]['width']/combined_dict[file_path]['height']
        ran_size = 1024
        total_pixels = ran_size*ran_size
        width = int(np.sqrt(total_pixels * aspect_ratio))
        height = int(total_pixels / width)
        image = pipe(prompt=promp,
                    height=make_divisible_by_8(height), width=make_divisible_by_8(width),
                    guidance_scale=3.5, num_images_per_prompt=1, max_sequence_length=512,
                    ).images[0]
        image.save(output_file)

