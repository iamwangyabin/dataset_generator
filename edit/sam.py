import os
import cv2
import supervision as sv
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Any, Dict, Tuple, Union
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForCausalLM, AutoProcessor




""""

# How to use this code:

# 1. Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Initialize the ImageMaskProcessor
processor = ImageMaskProcessor(device)

# 3. Use the save_mask_for_image method to generate and save a mask
image_path = "path/to/your/image.jpg"
text_input = "object or area you want to mask"
output_path = "path/to/save/mask.png"
expansion_pixels = 10  # Optional: adjust the mask expansion

processor.save_mask_for_image(image_path, text_input, output_path, expansion_pixels)

# This will load the image, generate a mask based on the text input,
# expand the mask, and save it to the specified output path.

"""




class ImageMaskProcessor:
    def __init__(self, device: torch.device):
        self.device = device
        self.florence_model, self.florence_processor = self.load_florence_model()
        self.sam_image_model = self.load_sam_image_model()

    def load_florence_model(self):
        model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True).to(self.device).eval()
        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
        return model, processor

    def load_sam_image_model(self):
        model = build_sam2("sam2_hiera_l.yaml", "./checkpoints/sam2_hiera_large.pt", device=self.device)
        return SAM2ImagePredictor(sam_model=model)

    def run_florence_inference(self, image: Image, task: str, text: str = "") -> Tuple[str, Dict]:
        prompt = task + text
        inputs = self.florence_processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        generated_ids = self.florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
        generated_text = self.florence_processor.batch_decode(
            generated_ids, skip_special_tokens=False)[0]
        response = self.florence_processor.post_process_generation(
            generated_text, task=task, image_size=image.size)
        return generated_text, response

    def run_sam_inference(self, image: Image, detections: sv.Detections) -> sv.Detections:
        image = np.array(image.convert("RGB"))
        self.sam_image_model.set_image(image)
        mask, score, _ = self.sam_image_model.predict(box=detections.xyxy, multimask_output=False)
        if len(mask.shape) == 4:
            mask = np.squeeze(mask)
        detections.mask = mask.astype(bool)
        return detections

    def get_mask_given_content(self, image_input: Image, text_input: str):
        texts = [prompt.strip() for prompt in text_input.split(",")]
        detections_list = []
        for text in texts:
            _, result = self.run_florence_inference(
                image=image_input,
                task='<OPEN_VOCABULARY_DETECTION>',
                text=text
            )
            detections = sv.Detections.from_lmm(
                lmm=sv.LMM.FLORENCE_2,
                result=result,
                resolution_wh=image_input.size
            )
            detections = self.run_sam_inference(image_input, detections)
            detections_list.append(detections)
        detections = sv.Detections.merge(detections_list)
        detections = self.run_sam_inference(image_input, detections)
        return detections.mask

    def clean_mask(self, mask):
        mask_copy = mask.copy()
        # import pdb;pdb.set_trace()
        h, w = mask.shape[:2]
        for pt in [(0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1)]:
            cv2.floodFill(mask_copy, None, pt, 255)
        mask_inv = cv2.bitwise_not(mask_copy)
        return cv2.bitwise_or(mask, mask_inv)


    def save_mask_for_image(self, image_path: str, text_input: str, output_path: str, expansion_pixels: int = 10):
        image = Image.open(image_path)
        mask = self.get_mask_given_content(image, text_input)
        combined_mask = np.any(mask, axis=0)
        expanded_mask = self.expand_mask(combined_mask, expansion_pixels)
        expanded_mask.save(output_path)

    def expand_mask(self, mask, expansion_pixels=5):
        binary_mask = mask.astype(np.uint8)
        kernel = np.ones((expansion_pixels * 2 + 1, expansion_pixels * 2 + 1), np.uint8)
        expanded_mask = cv2.dilate(binary_mask, kernel, iterations=1)

        expanded_mask = self.clean_mask((expanded_mask * 255).astype(np.uint8))


        return Image.fromarray(expanded_mask, mode='L')
