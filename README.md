# ReACT: Reference-based Anime Colorization Transformer

## 環境設置

### 系統要求
- Python 3.7
- PyTorch 1.13.1

### 安裝依賴
使用 pip 安裝以下套件：
```
pip install opencv-python natsort imageio scipy scikit-image tqdm matplotlib
```

或者，您可以直接使用提供的環境文件：
```
conda env create -f environment.yml
```

**注意**：請確保安裝與您的硬體設備相對應的 CUDA 版本（例如，CUDA 11.6）。

## 數據準備

1. 按照 `data_prepare` 目錄中的 README 文件的說明準備數據。
2. 您將獲得三種類型的圖像：
   - RGB frame
   - Distance field map
   - Sketch

## 訓練

1. 數據放置：
   - 將 RGB frame（已切好序列）放入 `./data/color`
   - 將 distance field 放入 `./data/dist`
   - 將 sketch 放入 `./data/sketch`（僅在運行 `test_showatt` 時需要）

2. 在 `config.py` 中確認用於訓練的數據集 `TRAIN_ANIMES`。

3. 執行訓練：
   ```
   python train.py
   ```

## 測試

### 若沒有自行訓練，可以先用以下連結提供的範例權重測試
1. 從[此連結](https://drive.google.com/file/d/1sfaadpcwvPLBDHzLgF-H9Ks0rztP6Nlf/view?usp=drive_link)下載 weight.pth 文件
2. 放到`./checkpoints/example`

### 視覺化測試
1. 在 `config.py` 中確認視覺化測試數據 `TEST_ANIMES_VISUAL`。
2. 在 `test.py` 第 43 行設置要使用的權重路徑。
3. 執行：
   ```
   python test.py
   ```
4. 結果將保存在 `test_visualize` 目錄中。

### 數值化測試
1. 在 `config.py` 中確認測試數據 `TEST_ANIMES_METRIC`。

- 計算 SSIM 和 PSNR：
  1. 在 `test_metric.py` 第 45 行設置要使用的權重路徑。
  2. 執行：
     ```
     python test_metric.py
     ```

- 計算 FID：
  1. 在 `test_metric.py` 第 49 行設置要使用的權重路徑。
  2. 執行：
     ```
     python test_FID.py
     ```

### Attention Map 視覺化
1. 在 `config.py` 中確認視覺化數據 `TEST_VISUALIZE_ATTN`。
2. 在 `test_showattn.py` 第 148 行設置要使用的權重路徑。
3. 執行：
   ```
   python test_showattn.py
   ```
4. 結果將保存在 `test_attention` 目錄中。

## 注意事項

- 確保所有的數據路徑在 `config.py` 中正確設置。
- 在運行測試之前，請確保已經完成了訓練或有可用的預訓練權重。
- 對於 FID 計算，需要安裝 `pytorch_fid` 包。