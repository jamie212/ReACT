import shutil
import argparse
import cv2 as cv
import numpy as np
import os
from pathlib import Path
from natsort import natsorted


def calc_hist(img: np.array) -> np.array:
    hist_b = cv.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv.calcHist([img], [1], None, [256], [0, 256])
    hist_r = cv.calcHist([img], [2], None, [256], [0, 256])
    hist = np.concatenate([hist_b, hist_g, hist_r], axis=0)

    return hist[:, 0]


def calc_difference(anime_dir: Path, thre, min_frames=8, mindiff=10):
    # f = open(f"{anime_dir}/separate.txt", "w")
    anime = str(anime_dir).split("/")[-1]
    print(anime)
    pathlist = list(anime_dir.glob("*.png"))
    nums = len(pathlist)
    print(f"original: {nums} img")
    start_num = 0
    sum_diff = 0
    del_list = []
    count = 1
    for num in range(1, nums-1):
        img_pre = cv.imread(str(f"{anime_dir}/{num -1}.png"))
        if img_pre is None:
            print("圖像pre加載錯誤")
        hist_pre = calc_hist(img_pre)

        img_cur = cv.imread(str(f"{anime_dir}/{num}.png"))
        if img_cur is None:
            print("圖像cur加載錯誤")
        hist_cur = calc_hist(img_cur)

        img_nex = cv.imread(str(f"{anime_dir}/{num+1}.png"))
        if img_nex is None:
            print("圖像nex加載錯誤")
        hist_nex = calc_hist(img_nex)

        diff_pre = (np.abs(hist_cur - hist_pre)).mean()
        diff_nex = (np.abs(hist_nex - hist_cur)).mean()
        # print(f"{num} and {num+1} diff : {diff_nex}")
        # print(f'{num}: pre: {diff_pre}, nex: {diff_nex}')
        if diff_pre < 10 and diff_nex < 10:
            # print(f"del: {num}")
            del_list.append(num)

    print(f"remain: {nums - len(del_list)}")

    for idx, num_file in enumerate(natsorted(os.listdir(anime_dir))):
        num = os.path.splitext(num_file)[0]
        if num == "separate":
            continue
        if int(num) in del_list:
            img_to_delete =  Path(anime_dir / f"{num}.png")
            if img_to_delete.exists():
                img_to_delete.unlink()
            # img_sketch =  Path(f"./SKETCH_PATH/{anime}/{num}.png")
            # if img_sketch.exists():
            #     img_sketch.unlink()


        
    remain_img = len(os.listdir(anime_dir))
    print(f"there are {remain_img} img left")


        # # 删除差异值小于 3 的图像
        # if diff < 4:
        #     img_to_delete = anime_dir / f"{num}.png"
        #     # if img_to_delete.exists():
        #     #     img_to_delete.unlink()
        #     print(f"Deleted {img_to_delete} due to low diff: {diff}")

        # elif diff > thre:
        #     # if num - start_num > min_frames and sum_diff > mindiff:
        #     if len(frame_list) > min_frames and sum_diff > mindiff:
        #         print(start_num, num)
        #         # f.write(f"{num},{start_num}\n")
        #     sum_diff = 0
        #     # start_num = num + 1
        #     start_num = next_num
        #     print(f'len:{len(frame_list)}')
        #     # if len(frame_list) > 35:

        #     frame_list.clear()

        # else:
        #     count = count + 1
        #     sum_diff = sum_diff + diff
        #     frame_list.append(next_num)
    # print(f"select shot: {anime_dir}")

      


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select scene")
    parser.add_argument("--d", type=Path, help="Directory that contains anime frames")
    parser.add_argument("--th", type=int, help="Threshold that indicates transition of the scenes")
    parser.add_argument("--mindiff", type=int, default=150, help="Threshold that indicates min transition of the scenes")
    args = parser.parse_args()

    calc_difference(args.d, args.th, mindiff=args.mindiff)
