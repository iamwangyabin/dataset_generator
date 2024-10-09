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

def visualize_inpainting_results(root_dir, num_samples=5):
    raw_dir = os.path.join(root_dir, '00115', 'raw_images')
    mask_dir = os.path.join(root_dir, '00115', 'masks')
    generated_dir = os.path.join(root_dir, '00115', 'generated')
    instruction_dir = os.path.join(root_dir, '00115', 'instructions')

    # 获取生成的图像文件名
    generated_files = [f for f in os.listdir(generated_dir) if f.endswith('.jpg') or f.endswith('.png')]

    # 限制样本数量
    generated_files = random.sample(generated_files, num_samples)

    # 创建一个大图来存储所有比较结果
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))

    for i, generated_file in enumerate(generated_files):
        base_name = generated_file.split('.')[0]  # 假设生成的文件名格式为 "base_name_score.jpg"

        raw_path = os.path.join(raw_dir, f"{base_name}.jpg")  # 假设原始图像为jpg格式
        mask_path = os.path.join(mask_dir, f"{base_name}_mask.png")
        generated_path = os.path.join(generated_dir, generated_file)
        json_path = os.path.join(instruction_dir, f"{base_name}.json")

        # 读取JSON文件
        with open(json_path, 'r') as f:
            instruction = json.load(f)

        # 打开图像
        raw_img = Image.open(raw_path)
        mask_img = Image.open(mask_path)
        generated_img = Image.open(generated_path)

        # 获取当前行的子图
        ax1, ax2, ax3 = axes[i]

        ax1.imshow(raw_img)
        ax1.set_title("raw image")
        ax1.axis('off')

        ax2.imshow(mask_img, cmap='gray')
        ax2.set_title("mask")
        ax2.axis('off')

        ax3.imshow(generated_img)
        ax3.set_title("inpainted")
        ax3.axis('off')
        info_text = f"selected region: {instruction['area_to_replace']}\nedition: {instruction['new_object']}"
        fig.text(0.5, 1 - (i + 1) / num_samples + 0.02, info_text, ha='center', va='center',
                 fontsize=10, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5},
                 transform=fig.transFigure, wrap=True)


    plt.tight_layout()

    output_path = os.path.join(root_directory, "comparison_results.webp")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # 关闭图形以释放内存



root_directory = "./"
visualize_inpainting_results(root_directory, num_samples=5)










