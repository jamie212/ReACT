# ReACT: Reference-based Anime Colorization Transformer

## Environment Setup

### System Requirements
- Python 3.7
- PyTorch 1.13.1

### Installing Dependencies
Install the following packages using pip:
```
pip install opencv-python natsort imageio scipy scikit-image tqdm matplotlib
```

Alternatively, you can use the provided environment file:
```
conda env create -f environment.yml
```

**Note**: Ensure you install the CUDA version corresponding to your hardware (e.g., CUDA 11.6).

## Data Preparation

1. Follow the instructions in the README file in the `data_prepare` directory to prepare the data.
2. You will obtain three types of images:
   - RGB frame
   - Distance field map
   - Sketch

## Training

1. Data placement:
   - Put RGB frames (properly sequenced) in `./data/color`
   - Put distance fields in `./data/dist`
   - Put sketches in `./data/sketch` (only needed when running `test_showatt`)

2. Confirm the dataset `TRAIN_ANIMES` used for training in `config.py`.

3. Run the training:
   ```
   python train.py
   ```

## Testing

### If you haven't trained yourself, you can test using the sample weights provided in the following link
1. Download the weight.pth file from [this link](https://drive.google.com/file/d/1sfaadpcwvPLBDHzLgF-H9Ks0rztP6Nlf/view?usp=drive_link)
2. Place it in `./checkpoints/example`

### Visual Testing
1. Confirm the visual test data `TEST_ANIMES_VISUAL` in `config.py`.
2. Set the path to the weights to be used in line 43 of `test.py`.
3. Execute:
   ```
   python test.py
   ```
4. Results will be saved in the `test_visualize` directory.

### Numerical Testing
1. Confirm the test data `TEST_ANIMES_METRIC` in `config.py`.

- Calculating SSIM and PSNR:
  1. Set the path to the weights to be used in line 45 of `test_metric.py`.
  2. Execute:
     ```
     python test_metric.py
     ```

- Calculating FID:
  1. Set the path to the weights to be used in line 49 of `test_metric.py`.
  2. Execute:
     ```
     python test_FID.py
     ```

### Attention Map Visualization
1. Confirm the visualization data `TEST_VISUALIZE_ATTN` in `config.py`.
2. Set the path to the weights to be used in line 148 of `test_showattn.py`.
3. Execute:
   ```
   python test_showattn.py
   ```
4. Results will be saved in the `test_attention` directory.

## Notes

- Ensure all data paths are correctly set in `config.py`.
- Make sure you have completed training or have available pre-trained weights before running tests.
- For FID calculation, you need to install the `pytorch_fid` package.