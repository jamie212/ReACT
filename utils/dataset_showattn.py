import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, Normalize, ToTensor

class AnimationColorizationDataset(Dataset):
    def __init__(self, root_dir, animes, transform=None):
        """
        root_dir: 資料集的根目錄
        animes: 一個列表，包含用於訓練的影片資料夾名稱
        transform: 圖像轉換/預處理
        """
        self.root_dir = root_dir
        self.animes = animes
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for movie in self.animes:
            sequence_path = os.path.join(self.root_dir, 'color', movie, 'separate.txt')
            with open(sequence_path, 'r') as file:
                for line in file:
                    end, start = map(int, line.strip().split(','))
                    for frame in range(start, end + 1):
                        if frame != start and frame != end:  # Exclude the reference frame
                            samples.append((movie, start, end, frame))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        movie, ref1_frame, ref2_frame, target_frame = self.samples[idx]

        color_dir = os.path.join(self.root_dir, 'color', movie)
        sketch_dir = os.path.join(self.root_dir, 'sketch', movie)
        dist_dir = os.path.join(self.root_dir, 'dist', movie)
        
        ref1_dist_path = os.path.join(dist_dir, f'{ref1_frame}.png')
        ref1_color_path = os.path.join(color_dir, f'{ref1_frame}.png')
        ref1_sketch_path = os.path.join(sketch_dir, f'{ref1_frame}.png')

        ref2_dist_path = os.path.join(dist_dir, f'{ref2_frame}.png')
        ref2_color_path = os.path.join(color_dir, f'{ref2_frame}.png')
        ref2_sketch_path = os.path.join(sketch_dir, f'{ref2_frame}.png')

        target_dist_path = os.path.join(dist_dir, f'{target_frame}.png')
        target_color_path = os.path.join(color_dir, f'{target_frame}.png')
        target_sketch_path = os.path.join(sketch_dir, f'{target_frame}.png')
        
        ref1_dist = read_image(ref1_dist_path)
        ref1_color = read_image(ref1_color_path)
        ref1_sketch = read_image(ref1_sketch_path)
        ref2_dist = read_image(ref2_dist_path)
        ref2_color = read_image(ref2_color_path)
        ref2_sketch = read_image(ref2_sketch_path)
        target_dist = read_image(target_dist_path)
        target_color = read_image(target_color_path)
        target_sketch = read_image(target_sketch_path)

        if self.transform:
            ref1_dist = self.transform(ref1_dist)
            ref1_color = self.transform(ref1_color)
            ref1_sketch = self.transform(ref1_sketch)
            ref2_dist = self.transform(ref2_dist)
            ref2_color = self.transform(ref2_color)
            ref2_sketch = self.transform(ref2_sketch)
            target_dist = self.transform(target_dist)
            target_color = self.transform(target_color)
            target_sketch = self.transform(target_sketch)

        return ref1_dist, ref1_color, ref2_dist, ref2_color, target_dist, target_color, ref1_sketch, ref2_sketch, target_sketch

