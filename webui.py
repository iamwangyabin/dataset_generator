import os
import numpy as np
import random
import requests
import base64
from tqdm import tqdm

def sample_value(mean=512, std_dev=80, lower_bound=256, upper_bound=768):
    value = np.random.normal(mean, std_dev)
    while value<lower_bound or value>upper_bound:
        value = np.random.normal(mean, std_dev)
    return int(value)

def sample_value_float(mean=512, std_dev=80, lower_bound=0, upper_bound=1):
    value = np.random.gumbel(mean, std_dev)
    while value<lower_bound or value>upper_bound:
        value = np.random.gumbel(mean, std_dev)
    return round(value, 1)

items = [
    "Ugly", "Bad anatomy", "Bad proportions", "Bad quality", "Blurry", "Cropped", "Deformed", 
    "Disconnected limbs", "Out of frame", "Out of focus", "Dehydrated", "Error", "Disfigured", 
    "Disgusting", "Extra arms", "Extra limbs", "Extra hands", "Fused fingers", "Gross proportions", 
    "Long neck", "Low res", "Low quality", "Jpeg", "Jpeg artifacts", "Malformed limbs", "Mutated", 
    "Mutated hands", "Mutated limbs", "Missing arms", "Missing fingers", "Picture frame", 
    "Poorly drawn hands", "Poorly drawn face", "Text", "Signature", "Username", "Watermark", 
    "Worst quality", "Collage", "Pixel", "Pixelated", "Grainy", "Bad anatomy", "Bad hands", 
    "Amputee", "Missing fingers", "Missing hands", "Missing limbs", "Missing arms", "Extra fingers", 
    "Extra hands", "Extra limbs", "Mutated hands", "Mutated", "Mutation", "Multiple heads", 
    "Malformed limbs", "Disfigured", "Poorly drawn hands", "Poorly drawn face", "Long neck", 
    "Fused fingers", "Fused hands", "Dismembered", "Duplicate", "Improper scale", "Ugly body", 
    "Cloned face", "Cloned body", "Gross proportions", "Body horror", "Too many fingers", "Cartoon", 
    "CGI", "Render", "3D", "Artwork", "Illustration", "3D render", "Cinema 4D", "Artstation", 
    "Octane render", "Painting", "Oil painting", "Anime", "2D", "Sketch", "Drawing", "Bad photography", 
    "Bad photo", "Deviant art", "Overexposed", "Simple background", "Plain background", "Grainy", 
    "Portrait", "Grayscale", "Monochrome", "Underexposed", "Low contrast", "Low quality", "Dark", 
    "Distorted", "White spots", "Deformed structures", "Macro", "Multiple angles"
]

def generate_random_negative(items):
    num_items = np.random.randint(0, 10)
    selected_items = np.random.choice(items, num_items, replace=False)
    result_string = ', '.join(selected_items)
    return result_string

# sampler_methods = [
#     'Euler a', 'Euler', 'LMS', 'Heun', 'DPM2', 'DPM2 a', 'DPM++ 2S a', 'DPM++ 2M', 'DPM++ SDE', 
#     'DPM++ fast', 'DPM++ adaptive', 'LMS Karras', 'DPM2 Karras', 'DPM2 a Karras', 
#     'DPM++ 2S a Karras', 'DPM++ 2M Karras', 'DPM++ SDE Karras', 'DDIM', 'PLMS', 'UniPC', 
#     'DPM++ 2M SDE Karras', 'DPM++ 2M SDE Exponential', 'DPM++ 2M SDE Heun Karras', 
#     'DPM++ 2M SDE Heun Exponential', 'DPM++ 3M SDE Karras', 'DPM++ 3M SDE Exponential'
# ]

sampler_methods = [
    "DPM++ 2M",
    "DPM++ SDE",
    "DPM++ 2M SDE",
    "DPM++ 2M SDE Heun",
    "DPM++ 2S a",
    "DPM++ 3M SDE",
    "Euler a",
    "Euler",
    "LMS",
    "Heun",
    "DPM2",
    "DPM2 a",
    "DPM fast",
    "DPM adaptive",
    "Restart",
    "DDIM",
    "PLMS"
]

def select_random_sampler(sampler_methods):
    selected_sampler = np.random.choice(sampler_methods)
    return selected_sampler

hires_upscalers = [
    "Latent",
    "Latent (antialiased)",
    "Latent (bicubic)",
    "Latent (bicubic antialiased)",
    "Latent (nearest)",
    "Latent (nearest-exact)",
    "Lanczos",
    "Nearest"
]

def select_random_upscaler(upscalers):
    selected_upscaler = random.choice(upscalers)
    return selected_upscaler

def thirty_percent_true():
    random_number = random.random()
    return random_number < 0.3


import json

with open('image_width.json', 'r') as json_file:
    file_name2width = json.load(json_file)

with open('coco_height.json', 'r') as json_file:
    file_name2height = json.load(json_file)

with open('coco_captions.json', 'r') as json_file:
    file_name2captions = json.load(json_file)    

output_dir = 'gen'
os.makedirs(output_dir, exist_ok=True)


for file_path in tqdm(list(file_name2captions.keys())):
    output_file = os.path.join(output_dir, file_path.split('.')[0] + '.png')
    
    if os.path.exists(output_file):
        continue
    try:
        promp = np.random.choice(file_name2captions[file_path])
        aspect_ratio = file_name2width[file_path]/file_name2height[file_path]
        ran_size = sample_value_float(mean=500, std_dev=100, lower_bound=256, upper_bound=768)
        total_pixels = ran_size*ran_size
        width = int(np.sqrt(total_pixels * aspect_ratio))
        height = int(total_pixels / width)

        payload = {
            "prompt": promp,
            "negative_prompt": generate_random_negative(items),
            "seed": random.randint(1, 1000000000),
            "width": width,
            "height": height,
            "cfg_scale": sample_value_float(mean=7, std_dev=2, lower_bound=1, upper_bound=30),
            "sampler_name": select_random_sampler(sampler_methods),
            "scheduler": 'Automatic',
            "steps": sample_value(mean=30, std_dev=10, lower_bound=5, upper_bound=50),
            "enable_hr": thirty_percent_true(),
            "hr_scale": sample_value_float(mean=1.5, std_dev=0.1, lower_bound=1, upper_bound=2),
            "hr_upscaler": select_random_upscaler(hires_upscalers),
            "hr_second_pass_steps": sample_value(mean=15, std_dev=5, lower_bound=1, upper_bound=30),
            "denoising_strength": sample_value_float(mean=0.5, std_dev=0.1, lower_bound=0, upper_bound=1),
        }

        response = requests.post(url='http://127.0.0.1:9386/sdapi/v1/txt2img', json=payload)

        r = response.json()
        
        with open(os.path.join('gen',file_path.split('.')[0]+'.png'), 'wb') as f:
            f.write(base64.b64decode(r['images'][0]))
    except:
        print('dd')