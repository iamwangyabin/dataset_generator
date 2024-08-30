import os
import numpy as np
import random
import requests
import base64
from tqdm import tqdm
import json
import torch
from diffusers import DiffusionPipeline
from torch.multiprocessing import Process, set_start_method

set_start_method('spawn', force=True)

def process_images(device_id, combined_dict, output_dir):

    pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            safety_checker = None,
            requires_safety_checker = False
        )
    pipe = pipe.to(device_id)

    for file_path, value in tqdm(combined_dict):
        output_file = os.path.join(output_dir, file_path + '.png')
        if os.path.exists(output_file):
            continue
        else:
            promp = value['cogvlm_caption']
            aspect_ratio = value['width'] / value['height']
            ran_size = 1024
            total_pixels = ran_size * ran_size
            width = int(np.sqrt(total_pixels * aspect_ratio))
            height = int(total_pixels / width)
            image = pipe(prompt=promp,
                         height=make_divisible_by_8(height), width=make_divisible_by_8(width),
                         num_inference_steps=50,
                         guidance_scale=3, num_images_per_prompt=1,
                         ).images[0]
            image.save(output_file)

def make_divisible_by_8(value):
    return (value // 16) * 16

if __name__ == '__main__':
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

    num_gpus = torch.cuda.device_count()
    combined_dict_parts = [[] for _ in range(num_gpus)]

    keys = list(combined_dict.keys())
    for i, key in enumerate(keys):
        combined_dict_parts[i % num_gpus].append((key, combined_dict[key]))


    output_dir = 'stable-diffusion-3-medium-diffusers'
    os.makedirs(output_dir, exist_ok=True)

    processes = []

    for device_id in range(num_gpus):
        p = Process(target=process_images, args=(f'cuda:{device_id}', combined_dict_parts[device_id], output_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()



