import os
import argparse
from pathlib import Path

def sort_and_rename_images(folder_path):
    # 獲取資料夾中的所有檔案
    files = os.listdir(folder_path)

    # 過濾出圖片檔案，假設圖片檔案是以 .jpg 結尾
    images = [file for file in files if file.endswith('.png')]

    # 根據檔名中的數字排序
    images.sort(key=lambda x: int(os.path.splitext(x)[0]))

    # 重新命名檔案
    for i, image in enumerate(images):
        # 建立新的檔名
        new_name = f"{i}.png"
        # 獲取原始檔案的完整路徑
        original_path = os.path.join(folder_path, image)
        # 獲取新檔案的完整路徑
        new_path = os.path.join(folder_path, new_name)
        # 重新命名檔案
        os.rename(original_path, new_path)

# 使用函數
# folder_path = './SKETCH_PATH/1_1'
# sort_and_rename_images(folder_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rename order")
    parser.add_argument("--path", type=Path, help="Directory that contains anime frames / sketch frames")
    args = parser.parse_args()

    sort_and_rename_images(args.path)