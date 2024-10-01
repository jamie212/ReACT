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
    
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = ImageColorizationTransformer(sketch_dim=256*256, color_dim=3*256*256, d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    checkpoint_path = "./checkpoints/example/weight.pth"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()  

    ssim_total = 0
    psnr_total = 0
    num_samples = 0

    with torch.no_grad(): 
        for i, (ref1_sketch, ref1_color, ref2_sketch, ref2_color, target_sketch, target_color) in enumerate(tqdm(dataloader, desc="Processing", leave=True)):
            ref1_sketch, ref1_color, ref2_sketch, ref2_color, target_sketch, target_color = ref1_sketch.to(device), ref1_color.to(device), ref2_sketch.to(device), ref2_color.to(device), target_sketch.to(device), target_color.to(device)

            outputs = model(target_sketch, ref1_sketch, ref1_color, ref2_sketch, ref2_color)

            for j in range(outputs.size(0)):
                output = outputs[j].unsqueeze(0)
                target_img_np = target_color[j].cpu().numpy().transpose(1, 2, 0)
                output_img_np = output.squeeze().cpu().numpy().transpose(1, 2, 0)

                ssim_val = compare_ssim(target_img_np, output_img_np, channel_axis=-1)
                psnr_val = compare_psnr(target_img_np, output_img_np)

                ssim_total += ssim_val
                psnr_total += psnr_val
                num_samples += 1
                
    avg_ssim = ssim_total / num_samples
    avg_psnr = psnr_total / num_samples

    print(f'Average SSIM: {avg_ssim}')
    print(f'Average PSNR: {avg_psnr}')

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    main()

