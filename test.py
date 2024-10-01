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
from config import TEST_ANIMES_VISUAL  
from tqdm import tqdm  
import gc
from torchvision.utils import save_image

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

    dataset = AnimationColorizationDataset(root_dir='./data', animes=TEST_ANIMES_VISUAL, transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = ImageColorizationTransformer(sketch_dim=256*256, color_dim=3*256*256, d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    checkpoint_path = "./checkpoints/example/weight.pth"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()  
    test_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./test_visualize/{test_time}"
    os.makedirs(output_dir, exist_ok=True) 

    with torch.no_grad(): 
        for i, (ref1_dist, ref1_color, ref2_dist, ref2_color, target_dist, target_color) in enumerate(dataloader):
            ref1_dist, ref1_color, ref2_dist, ref2_color, target_dist, target_color = ref1_dist.to(device), ref1_color.to(device), ref2_dist.to(device), ref2_color.to(device), target_dist.to(device), target_color.to(device)
            outputs = model(target_dist, ref1_dist, ref1_color, ref2_dist, ref2_color)

            for j, output in enumerate(outputs):
                combined_image = torch.cat((ref1_color[j:j+1], ref2_color[j:j+1], target_color[j:j+1], output.unsqueeze(0)), dim=3)
                save_image(combined_image, os.path.join(output_dir, f"output_{i*len(outputs)+j}.png"))


    print(f'Finished Testing, outputs saved in {output_dir}')

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    main()

