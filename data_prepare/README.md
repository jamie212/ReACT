
# Dataset Preparation Guide

## Required Tools
- ffmpeg
- Python 3.x
- Necessary Python libraries (such as OpenCV, NumPy, etc.)

## Steps

### 1. Video to Frame Conversion

1. Download the animation video (.mp4 format). Named 'v1' for example.

2. Use ffmpeg to convert the video to frames:
   ```
   ffmpeg -i "video/v1" -vf "fps=30,scale=-1:256" -start_number 0 "DATA_PATH/v1/%d.png"
   ```

3. Manually remove the frames of the opening and ending credits.

4. Place the processed frame folder into `./DATA_PATH`.

5. Rename and reorder the frames:
   ```
   python rename_order.py --path "./DATA_PATH/v1"
   ```

6. Remove the subtitle area (ensure the correct frame folder path is set in the script):
   ```
   python cut_word.py
   ```

7. Resize the frames to 256x256:
   ```
   python resize_and_crop.py './DATA_PATH/v1'
   ```

### 2. Frame to Sequence Processing

8. Remove frames that are too similar:
   ```
   python del_dup.py --d "./DATA_PATH/v1" --th 200
   ```

9. Rename and reorder the frames again:
   ```
   python rename_order.py --path "./DATA_PATH/v1"
   ```

10. Shorten sequences that are too long:
    ```
    python shorten_scene.py --d "./DATA_PATH/1_1" --th 200
    ```

11. Rename and reorder the frames again:
    ```
    python rename_order.py --path "./DATA_PATH/v1"
    ```

12. Slice into sequences and record the results:
    ```
    python slice_seq.py --d "./DATA_PATH/v1" --th 200
    ```
    The results will be saved in the `seperate.txt` file in the frame folder.

### 3. Generate Sketch

13. Generate the sketches (ensure the correct `subfolders` are set in the script):
    ```
    python xdog.py
    ```

### 4. Generate Distance Field Map

14. Generate distance field maps from the sketches:
    ```
    python get_distance.py "./SKETCH_PATH/v1" "./DIST_PATH/v1"
    ```

## Notes

- Ensure that all paths are correctly set in the scripts.
- Some steps may need to be adjusted based on the specific video content, such as the number of frames removed for the opening and ending.
- When processing large datasets, ensure sufficient disk space is available.
- Regularly back up processed data to prevent accidental loss.

## Output

After completing all steps, you will have:
1. Processed RGB frame sequences (in `DATA_PATH`)
2. Corresponding sketches (in `SKETCH_PATH`)
3. Distance field maps (in `DIST_PATH`)
4. Sequence information (in `DATA_PATH/v1/seperate.txt`)