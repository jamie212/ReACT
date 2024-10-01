- 建立環境：
    python 3.7
    pytorch 1.13.1
    pip install
        opencv-python
        natsort
        imageio
        scipy
        scikit-image
        tqdm
        matplotlib
        pytorch_fid (if you want to calculate FID)
    
    或使用conda env create -f environment.yml
    ！注意硬體設備對應版本（我的是cuda 11.6）!

- 建立資料集:
    1. 依照data_prepare中readme的方法做
    2. 獲得RGB frame, dist field map, sketch三種圖

- 訓練方式：
    1. 擺放資料
        - DATA_PATH中的RGB frame（已經切好sequence）放到./data/color
        - DIST_PATH中的dist放到./data/dist
        - （如果要跑test_shwoatt才需要）SKETCH_PATH中的sketch放到./data/sketch
    2. 到config.py確認要用於訓練的資料集TRAIN_ANIMES
    3. `python train.py` 

- 測試方式：
    - 若沒有自行訓練，可以先用我提供的example weight進行測試
        https://drive.google.com/file/d/1sfaadpcwvPLBDHzLgF-H9Ks0rztP6Nlf/view?usp=drive_link
        將下載的weight.pth放到`./checkpoints/example`
    * 視覺化
        1. 到config.py確認視覺化測試資料TEST_ANIMES_VISUAL
        2. 到test.py第43行改要使用的weight路徑
        3. `python test.py`
        4. 結果會存在test_visualize中
    * 數值化 
        1. 到config.py確認視覺化測試資料TEST_ANIMES_METRIC
        - 計算SSIM, PSNR: 
            2. 到test_metric.py第45行改要使用的weight路徑
            3. `python test_metric.py`
        - 計算FID: 
            2. 到test_metric.py第49行改要使用的weight路徑
            3. `python test_FID.py`
    * Attention map視覺化
        1. 到config.py確認視覺化測試資料TEST_VISUALIZE_ATTN
        2. 到test_showattn.py第148行改要使用的weight路徑
        3. `python test_showattn.py`
        4. 結果會存在test_attention中


