from PIL import Image
import os
import argparse

def resize_and_crop(image_path):
    with Image.open(image_path) as img:
        # 计算新的尺寸，保持宽高比
        original_width, original_height = img.size
        new_height = 256
        new_width = int(original_width * (new_height / original_height))

        # 等比例缩放
        img_resized = img.resize((new_width, new_height), Image.ANTIALIAS)

        # 计算裁剪框
        left = (new_width - 256) / 2
        top = 0  # 由于高度已经是256，所以top为0
        right = (new_width + 256) / 2
        bottom = 256

        # 裁剪图像
        img_cropped = img_resized.crop((left, top, right, bottom))
        img_cropped.save(image_path)  # 覆盖原图像

def process_folder(folder_path):
    for subdir, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                resize_and_crop(os.path.join(subdir, file))
                print(f"resize&crop: {os.path.join(subdir, file)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resize and crop images in specified folders.')
    parser.add_argument('data_folder', type=str, help='Path to the data folder')
    # parser.add_argument('sketch_folder', type=str, help='Path to the sketch folder')
    args = parser.parse_args()

    process_folder(args.data_folder)
    # process_folder(args.sketch_folder)
