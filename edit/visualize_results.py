"""
root/
│
└── split1/
    ├── raw_image/        # Extracted images
    ├── instruction/      # Generated instructions
    ├── mask/             # Generated masks
    └── generated/        # Inpainted images

just a visualization of our inpaining results.
raw image vs mask vs generated
"""
import os
import json
import matplotlib.pyplot as plt
from PIL import Image
import random
import cv2
import numpy as np
import shutil

def visualize_inpainting_results(root_dir, num_samples=5):
    raw_dir = os.path.join(root_dir, '00211', 'raw_images')
    mask_dir = os.path.join(root_dir, '00211', 'masks')
    generated_dir = os.path.join(root_dir, '00211', 'generated')
    instruction_dir = os.path.join(root_dir, '00211', 'instructions')

    generated_files = [f for f in os.listdir(generated_dir) if f.endswith('.jpg') or f.endswith('.png')]
    generated_files = random.sample(generated_files, num_samples)
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))

    for i, generated_file in enumerate(generated_files):
        base_name = generated_file.split('_')[0]  # 假设生成的文件名格式为 "base_name_score.jpg"

        raw_path = os.path.join(raw_dir, f"{base_name}.jpg")  # 假设原始图像为jpg格式
        mask_path = os.path.join(mask_dir, f"{base_name}_mask.png")
        generated_path = os.path.join(generated_dir, generated_file)

        raw_img = Image.open(raw_path)
        mask_img = Image.open(mask_path)
        generated_img = Image.open(generated_path)

        mask_np = np.array(mask_img)
        generated_np = np.array(generated_img)

        # 边缘检测以获取轮廓
        edges = cv2.Canny(mask_np, threshold1=50, threshold2=150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 绘制轮廓在生成的图像上
        contour_img = generated_np.copy()
        cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)  # Blue contours

        # 转换回PIL图像
        contour_pil = Image.fromarray(contour_img)

        # 获取当前行的子图
        ax1, ax2, ax3 = axes[i]

        ax1.imshow(raw_img)
        ax1.set_title("raw image")
        ax1.axis('off')

        ax2.imshow(mask_img, cmap='gray')
        ax2.set_title("mask")
        ax2.axis('off')

        ax3.imshow(contour_pil)
        ax3.set_title("inpainted")
        ax3.axis('off')

    plt.tight_layout()

    output_path = os.path.join(root_directory, "comparison_results.webp")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # 关闭图形以释放内存



root_directory = "./"
visualize_inpainting_results(root_directory, num_samples=5)




# import requests
#
# API_URL = "https://api-inference.huggingface.co/models/OnomaAIResearch/Illustrious-xl-early-release-v0"
# headers = {"Authorization": "Bearer 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"}
#
# def query(payload):
#     response = requests.post(API_URL, headers=headers, json=payload)
#     return response.content
#
#
# image_bytes = query({
# 	"inputs": "Astronaut riding a horse",
# })
# # You can access the image with PIL.Image for example
# import io
# from PIL import Image
# image = Image.open(io.BytesIO(image_bytes))
#




