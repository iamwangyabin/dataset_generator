import os
import numpy as np
import random
import requests
import base64
from tqdm import tqdm
import json
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch



with open('coco_width.json', 'r') as json_file:
    file_name2width = json.load(json_file)

with open('coco_height.json', 'r') as json_file:
    file_name2height = json.load(json_file)

with open('coco_captions.json', 'r') as json_file:
    file_name2captions = json.load(json_file)    

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil

stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", torch_dtype=torch.float16,
                                            safety_checker = None,
                                            requires_safety_checker = False
                                            )

stage_2 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16,
                                            safety_checker = None,
                                            requires_safety_checker = False
)
stage_1 = stage_1.to('cuda')
stage_2 = stage_2.to('cuda')
output_dir = 'IF'
os.makedirs(output_dir, exist_ok=True)


for file_path in tqdm(list(file_name2captions.keys())):
    output_file = os.path.join(output_dir, file_path.split('.')[0] + '_0.png')
    if os.path.exists(output_file):
        continue
    else:
        promp = file_name2captions[file_path][0]
        prompt_embeds, negative_embeds = stage_1.encode_prompt(promp)
        image = stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt").images
        image = stage_2(
            image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt"
        ).images
        for idx, img in enumerate(pt_to_pil(image)):
            img.save(os.path.join(output_dir, file_path.split('.')[0] + f'_{idx}.png'))
