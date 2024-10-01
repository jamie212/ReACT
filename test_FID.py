import torch
import os
import datetime
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Lambda, Normalize
import torch.optim as optim
import torch.nn as nn
from torchvision.transforms.functional import normalize
from models.model import ImageColorizationTransformer  
from utils.dataset import AnimationColorizationDataset 
from config import TEST_ANIMES_METRIC
from tqdm import tqdm  
import gc
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import subprocess
import shutil
from torchvision.transforms.functional import to_pil_image


def dynamic_normalize(tensor):
    if tensor.dtype == torch.uint8: 
        tensor = tensor.float() / 255.0  # uint8 to float and normalize to [0, 1]
    return tensor

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Save checkpoint if a new best is achieved"""
    print("=> Saving a new best")
    torch.save(state, filename)  # save checkpoint

def main():

    transform = Compose([
        Resize((256, 256)),
        Lambda(dynamic_normalize),
    ])

    dataset = AnimationColorizationDataset(root_dir='./data', animes=TEST_ANIMES_METRIC, transform=transform)
    
    batch_size = 2

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ImageColorizationTransformer(sketch_dim=256*256, color_dim=3*256*256, d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    checkpoint_path = "./checkpoints/example/weight.pth"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()  

    output_dir = './FID/output'
    target_dir = './FID/target'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    num_samples = 0

    with torch.no_grad(): 
        for i, (ref1_sketch, ref1_color, ref2_sketch, ref2_color, target_sketch, target_color) in enumerate(tqdm(dataloader, desc="Processing", leave=True)):
            ref1_sketch, ref1_color, ref2_sketch, ref2_color, target_sketch, target_color = ref1_sketch.to(device), ref1_color.to(device), ref2_sketch.to(device), ref2_color.to(device), target_sketch.to(device), target_color.to(device)

            outputs = model(target_sketch, ref1_sketch, ref1_color, ref2_sketch, ref2_color)

            for j, output in enumerate(outputs):
                target = target_color[j:j+1].squeeze(0)
                output_img = to_pil_image(output)
                target_img = to_pil_image(target)

                output_img.save(os.path.join(output_dir, f'{i * batch_size + j}.png'))
                target_img.save(os.path.join(target_dir, f'{i * batch_size + j}.png'))
        
    fid_command = f'python -m pytorch_fid {output_dir} {target_dir}'
    subprocess.run(fid_command.split())

    shutil.rmtree(output_dir)
    shutil.rmtree(target_dir)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    main()

