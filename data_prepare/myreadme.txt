準備資料集的步驟：

video -> frame -> 刪除片頭片尾、字幕 -> resize
    1. 下載動畫影片（.mp4）
    2. 用ffmpeg轉成frame
        `ffmpeg -i "video/v1" -vf "fps=30,scale=-1:256" -start_number 0 "DATA_PATH/v1/%d.png"`
    3. 手動將片頭曲、片尾曲的段落刪除
    4. 將frame資料夾放進./DATA_PATH
    5. `python rename_order.py --path "./DATA_PATH/v1"` -> 重整frame命名
    6. `python cut_word.py` 先去裡面改frame資料夾路徑 -> 切掉字幕區塊
    7. `python resize_and_crop.py './DATA_PATH/v1'` -> 調整成256*256大小

frame -> sequence
    8. `python del_dup.py --d "./DATA_PATH/v1" --th 200` -> 刪除過於重複的幀數
    9. `python rename_order.py --path "./DATA_PATH/v1"` -> 重整frame命名
    10. `python shorten_scene.py --d "./DATA_PATH/1_1" --th 200` -> 將太長的sequencen縮短
    11. `python rename_order.py --path "./DATA_PATH/v1"` -> 重整frame命名
    12. `python slice_seq.py --d "./DATA_PATH/v1" --th 200` -> 切成sequence，並且將結果存到frame資料夾中的seperate.txt中

generate sketch
    13. `python xdog.py` 先去裡面改subfolders = ['v1']放要轉換的影片名 -> RGB轉成sketch

generate distance field map
    14. `python get_distance.py "./SKETCH_PATH/v1" "./DIST_PATH/v1"` 





