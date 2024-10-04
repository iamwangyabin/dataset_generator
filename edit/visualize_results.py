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


def visualize_inpainting_results(root_dir, num_samples=5):
    raw_dir = os.path.join(root_dir, '00013', 'raw_images')
    mask_dir = os.path.join(root_dir, '00013', 'masks')
    generated_dir = os.path.join(root_dir, '00013', 'generated')
    instruction_dir = os.path.join(root_dir, '00013', 'instructions')

    # 获取生成的图像文件名
    generated_files = [f for f in os.listdir(generated_dir) if f.endswith('.jpg') or f.endswith('.png')]

    # 限制样本数量
    generated_files = generated_files[:num_samples]

    for generated_file in generated_files:
        base_name = generated_file.split('_')[0]  # 假设生成的文件名格式为 "base_name_score.jpg"

        raw_path = os.path.join(raw_dir, f"{base_name}.jpg")  # 假设原始图像为jpg格式
        mask_path = os.path.join(mask_dir, f"{base_name}_mask.png")
        generated_path = os.path.join(generated_dir, generated_file)
        json_path = os.path.join(instruction_dir, f"{base_name}.json")

        # 检查文件是否存在
        if not all(os.path.exists(path) for path in [raw_path, mask_path, generated_path, json_path]):
            print(f"缺少文件: {base_name}")
            continue

        # 读取JSON文件
        with open(json_path, 'r') as f:
            instruction = json.load(f)

        # 打开图像
        raw_img = Image.open(raw_path)
        mask_img = Image.open(mask_path)
        generated_img = Image.open(generated_path)

        # 创建图表
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))

        ax1.imshow(raw_img)
        ax1.set_title("raw image")
        ax1.axis('off')

        ax2.imshow(mask_img, cmap='gray')
        ax2.set_title("mask")
        ax2.axis('off')

        ax3.imshow(generated_img)
        ax3.set_title("inpainted")
        ax3.axis('off')

        plt.suptitle(f"Image: {base_name}", fontsize=16)

        # 添加指令信息
        info_text = f"selected region: {instruction['area_to_replace']}\n edition: {instruction['new_object']}"
        plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=12,
                    bbox={"facecolor": "orange", "alpha": 0.5, "pad": 1})
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)  # 为底部文本留出空间

        output_path = os.path.join(root_directory, f"{base_name}_comparison.webp")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # 关闭图形以释放内存





root_directory = "./"
visualize_inpainting_results(root_directory, num_samples=5)










