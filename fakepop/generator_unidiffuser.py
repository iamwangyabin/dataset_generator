import os
import numpy as np
import random
import requests
import base64
from tqdm import tqdm
import json
import torch
from diffusers import UniDiffuserPipeline

pipe = UniDiffuserPipeline.from_pretrained(
    "thu-ml/unidiffuser-v1",
    torch_dtype=torch.float16,
    safety_checker = None,
    requires_safety_checker = False
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

output_dir = 'unidiffuser-v1'
os.makedirs(output_dir, exist_ok=True)


file_paths = list(combined_dict.keys())[:3000]
random.shuffle(file_paths)

for file_path in tqdm(file_paths):
    output_file = os.path.join(output_dir, file_path + '.png')
    if os.path.exists(output_file):
        continue
    else:
        promp = combined_dict[file_path]['cogvlm_caption']
        images = pipe(prompt=promp,
                    height=512, width=512,
                    num_inference_steps=50,
                    guidance_scale=8.0, num_images_per_prompt=1,
                    ).images[0]
        images.save(output_file)


