import shutil
import argparse
import cv2 as cv
import numpy as np

from pathlib import Path


def calc_hist(img: np.array) -> np.array:
    hist_b = cv.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv.calcHist([img], [1], None, [256], [0, 256])
    hist_r = cv.calcHist([img], [2], None, [256], [0, 256])
    hist = np.concatenate([hist_b, hist_g, hist_r], axis=0)

    return hist[:, 0]

def delete_frames(frames_diff, max_frames, data_path):
    # Sort frames by diff and keep only the frames with the largest diffs
    frames_diff.sort(key=lambda x: x[1])
    for i in range(len(frames_diff) - max_frames):
        frame_num = frames_diff[i][0]
        frame_path = data_path / f"{frame_num}.png"
        frame_path.unlink()  # Delete the frame


def calc_difference(anime_dir: Path,
                    thre,
                    min_frames=8, 
                    max_frames=41):
    f = open(f"{anime_dir}/separate.txt", "w")
    pathlist = list(anime_dir.glob("*.png"))
    nums = len(pathlist)
    start_num = 0
    sum_len = 0
    frames_diff = []  # List to store frame number and diff
    for num in range(0, nums-1):
        img = cv.imread(str(f"{anime_dir}/{num}.png"))
        hist = calc_hist(img)
        img_1 = cv.imread(str(f"{anime_dir}/{num + 1}.png"))
        hist_1 = calc_hist(img_1)

        diff = (np.abs(hist_1 - hist)).mean()
        frames_diff.append((num, diff))

        if diff > thre: # change scene
            if num - start_num > min_frames:
                print(num, start_num, num - start_num)
                f.write(f"{num},{start_num}\n")
                sum_len += num - start_num
            start_num = num + 1
            frames_diff = [] 
        # else: # same scene



        if num == nums-2:
            print(num+1, start_num, num+1 - start_num)
            # if len(frames_diff) > max_frames:
            #     delete_frames(frames_diff, max_frames, anime_dir)
            f.write(f"{num+1},{start_num}\n")
            # sum_len += num+1 - start_num

        
        
    print(f"sum len {sum_len}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select scene")
    parser.add_argument("--d", type=Path, help="Directory that contains anime frames")
    parser.add_argument("--th", type=int, help="Threshold that indicates transition of the scenes")
    args = parser.parse_args()

    calc_difference(args.d, args.th)