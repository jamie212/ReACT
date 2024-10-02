# 資料集準備指南

## 準備工具
- ffmpeg
- Python 3.x
- 必要的 Python 庫（如 OpenCV、NumPy 等）

## 步驟

### 1. 視頻到幀的轉換

1. 下載動畫視頻（.mp4 格式）。

2. 使用 ffmpeg 將視頻轉換為幀：
   ```
   ffmpeg -i "video/v1" -vf "fps=30,scale=-1:256" -start_number 0 "DATA_PATH/v1/%d.png"
   ```

3. 手動刪除片頭曲和片尾曲的幀。

4. 將處理後的幀資料夾放入 `./DATA_PATH`。

5. 重新排序幀的命名：
   ```
   python rename_order.py --path "./DATA_PATH/v1"
   ```

6. 切除字幕區域（需要先在腳本中設置正確的幀資料夾路徑）：
   ```
   python cut_word.py
   ```

7. 調整幀大小為 256x256：
   ```
   python resize_and_crop.py './DATA_PATH/v1'
   ```

### 2. 幀到序列的處理

8. 刪除過於相似的幀：
   ```
   python del_dup.py --d "./DATA_PATH/v1" --th 200
   ```

9. 再次重新排序幀的命名：
   ```
   python rename_order.py --path "./DATA_PATH/v1"
   ```

10. 縮短過長的序列：
    ```
    python shorten_scene.py --d "./DATA_PATH/1_1" --th 200
    ```

11. 再次重新排序幀的命名：
    ```
    python rename_order.py --path "./DATA_PATH/v1"
    ```

12. 切分成序列並記錄結果：
    ```
    python slice_seq.py --d "./DATA_PATH/v1" --th 200
    ```
    結果將保存在幀資料夾中的 `seperate.txt` 文件中。

### 3. 生成草圖（Sketch）

13. 生成草圖（需要在腳本中設置正確的 `subfolders`）：
    ```
    python xdog.py
    ```

### 4. 生成距離場圖（Distance Field Map）

14. 從草圖生成距離場圖：
    ```
    python get_distance.py "./SKETCH_PATH/v1" "./DIST_PATH/v1"
    ```

## 注意事項

- 確保所有腳本中的路徑設置正確。
- 某些步驟可能需要根據具體視頻內容進行調整，例如刪除片頭片尾的幀數。
- 處理大量數據時，請確保有足夠的磁盤空間。
- 建議定期備份處理後的數據，以防意外發生。

## 輸出

完成所有步驟後，您將得到：
1. 處理後的 RGB 幀序列（在 `DATA_PATH` 中）
2. 對應的草圖（在 `SKETCH_PATH` 中）
3. 距離場圖（在 `DIST_PATH` 中）
4. 序列信息（在 `DATA_PATH/v1/seperate.txt` 文件中）
