import os
from PIL import Image

def crop_bottom(image_path, crop_height):
    with Image.open(image_path) as img:
        width, height = img.size
        crop_area = (0, 0, width, height - crop_height)
        cropped_img = img.crop(crop_area)
        cropped_img.save(image_path)  # 覆蓋原始圖片

def process_folders(root_folder, crop_height):
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".png"):
                image_path = os.path.join(subdir, file)
                crop_bottom(image_path, crop_height)
                print(f"Cropped and overwritten {image_path}")


root_folder = "./DATA_PATH/v1"  
crop_height = 45  # 裁剪的像素高度，大概估算字幕高度
process_folders(root_folder, crop_height)
