import torch
import os
import datetime
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Lambda, Normalize
import torch.optim as optim
import torch.nn as nn
from torchvision.transforms.functional import normalize
from models.model_showattn import ImageColorizationTransformer  
from utils.dataset_showattn import AnimationColorizationDataset 
from config import TEST_VISUALIZE_ATTN 
from tqdm import tqdm  
import gc
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from PIL import Image
import matplotlib.patches as patches


def dynamic_normalize(tensor):
    if tensor.dtype == torch.uint8: 
        tensor = tensor.float() / 255.0  # uint8 to float and normalize to [0, 1]
    return tensor

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Save checkpoint if a new best is achieved"""
    print("=> Saving a new best")
    torch.save(state, filename)  # save checkpoint


def normalize_attention(attention_map):
    min_val = np.min(attention_map)
    max_val = np.max(attention_map)
    if max_val > min_val:  # Avoid division by zero
        attention_map = (attention_map - min_val) / (max_val - min_val)
    return attention_map

def plot_attention(output, target_sketch, ref1_sketch, ref2_sketch, target_groundtruth, target_dist, ref1_dist, ref2_dist, attention_map, save_path):
    original_size, feature_size = 256, 32
    point = (60, 155) 
    scale = original_size / feature_size
    region_x = int(point[0] / scale)
    region_y = int(point[1] / scale)

    index = region_y * 32 + region_x
    attention_row_ref1 = attention_map[index, :1024].reshape(32, 32)
    attention_row_ref2 = attention_map[index, 1024:2048].reshape(32, 32)

    attention_map_ref1 = np.kron(attention_row_ref1, np.ones((8, 8)))
    attention_map_ref2 = np.kron(attention_row_ref2, np.ones((8, 8)))

    attention_map_ref1 = normalize_attention(attention_map_ref1)
    attention_map_ref2 = normalize_attention(attention_map_ref2)

    target_sketch = target_sketch.squeeze(0).repeat(3, 1, 1)
    ref1_sketch = ref1_sketch.squeeze(0).repeat(3, 1, 1)
    ref2_sketch = ref2_sketch.squeeze(0).repeat(3, 1, 1)
    output = output.squeeze(0)
    target_groundtruth = target_groundtruth.squeeze(0)
    target_dist = target_dist.squeeze(0).repeat(3, 1, 1)
    ref1_dist = ref1_dist.squeeze(0).repeat(3, 1, 1)
    ref2_dist = ref2_dist.squeeze(0).repeat(3, 1, 1)

    fig, ax = plt.subplots(2, 4, figsize=(24, 12))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(wspace=0.05, hspace=0.1) 

    output = output.float()
    output = (output - output.min()) / (output.max() - output.min())

    def add_border(axis):
        for spine in axis.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(3)

    ax[0, 0].imshow(target_groundtruth.permute(1, 2, 0).numpy())
    rect = patches.Rectangle((region_x*scale, region_y*scale), scale, scale, linewidth=1, edgecolor='r', facecolor='none')
    ax[0, 0].add_patch(rect)
    ax[0, 0].set_title('Ground truth', color='black', fontsize=16)
    ax[0, 0].axis('off')
    add_border(ax[0, 0])

    ax[0, 1].imshow(target_dist.permute(1, 2, 0).numpy(), cmap='gray')
    rect = patches.Rectangle((region_x*scale, region_y*scale), scale, scale, linewidth=1, edgecolor='r', facecolor='none')
    ax[0, 1].add_patch(rect)
    ax[0, 1].set_title('Target dist', color='black', fontsize=16)
    ax[0, 1].axis('off')
    add_border(ax[0, 1])

    ax[0, 2].imshow(ref1_dist.permute(1, 2, 0).numpy(), cmap='gray')
    ax[0, 2].set_title('Ref 1 dist', color='black', fontsize=16)
    ax[0, 2].axis('off')
    add_border(ax[0, 2])

    ax[0, 3].imshow(ref2_dist.permute(1, 2, 0).numpy(), cmap='gray')
    ax[0, 3].set_title('Ref 2 dist', color='black', fontsize=16)
    ax[0, 3].axis('off')
    add_border(ax[0, 3])

    ax[1, 0].imshow(output.permute(1, 2, 0).numpy())
    rect = patches.Rectangle((region_x*scale, region_y*scale), scale, scale, linewidth=1, edgecolor='r', facecolor='none')
    ax[1, 0].add_patch(rect)
    ax[1, 0].set_title('Output', color='black', fontsize=16)
    ax[1, 0].axis('off')
    add_border(ax[1, 0])

    ax[1, 1].imshow(target_sketch.permute(1, 2, 0).numpy(), cmap='gray')
    rect = patches.Rectangle((region_x*scale, region_y*scale), scale, scale, linewidth=1, edgecolor='r', facecolor='none')
    ax[1, 1].add_patch(rect)
    ax[1, 1].set_title('Target sketch', color='black', fontsize=16)
    ax[1, 1].axis('off')
    add_border(ax[1, 1])

    ax[1, 2].imshow(ref1_sketch.permute(1, 2, 0).numpy(), cmap='gray')
    ax[1, 2].imshow(plt.cm.Reds(attention_map_ref1), cmap='Reds', alpha=0.6)
    ax[1, 2].set_title('Ref 1 with Attention Heatmap', color='black', fontsize=16)
    ax[1, 2].axis('off')
    add_border(ax[1, 2])

    ax[1, 3].imshow(ref2_sketch.permute(1, 2, 0).numpy(), cmap='gray')
    ax[1, 3].imshow(plt.cm.Reds(attention_map_ref2), cmap='Reds', alpha=0.6)
    ax[1, 3].set_title('Ref 2 with Attention Heatmap', color='black', fontsize=16)
    ax[1, 3].axis('off')
    add_border(ax[1, 3])

    # Save the figure
    plt.savefig(save_path)
    plt.close(fig)

def main():

    transform = Compose([
        Resize((256, 256)),
        Lambda(dynamic_normalize),
    ])

    dataset = AnimationColorizationDataset(root_dir='./data', animes=TEST_VISUALIZE_ATTN , transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = ImageColorizationTransformer(sketch_dim=256*256, color_dim=3*256*256, d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048)

    device = torch.device("cpu")
    model = model.to(device)

    checkpoint_path = "./checkpoints/example/weight.pth"
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()  
    test_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    attention_dir = f"./test_attention/{test_time}"
    os.makedirs(attention_dir, exist_ok=True) 

    with torch.no_grad(): 
        for i, (ref1_dist, ref1_color, ref2_dist, ref2_color, target_dist, target_color, ref1_sketch, ref2_sketch, target_sketch) in enumerate(dataloader):
            ref1_dist, ref1_color, ref2_dist, ref2_color, target_dist, target_color, ref1_sketch, ref2_sketch, target_sketch = ref1_dist.to(device), ref1_color.to(device), ref2_dist.to(device), ref2_color.to(device), target_dist.to(device), target_color.to(device), ref1_sketch.to(device), ref2_sketch.to(device), target_sketch.to(device)

            outputs, attention_maps = model(target_dist, ref1_dist, ref1_color, ref2_dist, ref2_color)
            
            layer_idx = 2  # 0, 1, 2 layer (2 is last)
            head_idx = 0 

            attention_map = attention_maps[layer_idx][head_idx].detach().cpu().numpy()
            plot_attention(outputs, target_sketch, ref1_sketch, ref2_sketch, target_color, target_dist, ref1_dist, ref2_dist, attention_map, os.path.join(attention_dir, f"output_{i}.png"))

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    main()

