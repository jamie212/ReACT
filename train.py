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
from config import TRAIN_ANIMES  
from tqdm import tqdm  
import gc
from torchvision.models import vgg19
from utils.loss import perceptual_style_loss
from torchvision.models import VGG19_Weights
import matplotlib.pyplot as plt
import argparse
from torch.cuda.amp import GradScaler, autocast

def plot_and_save_loss(losses, loss_name, file_name):
    plt.figure(figsize=(8, 5))  
    plt.plot(losses, label=f'{loss_name} Loss')
    plt.legend()
    plt.title(f'{loss_name} Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'{file_name}.png')  
    plt.close()  

def dynamic_normalize(tensor):
    if tensor.dtype == torch.uint8: 
        tensor = tensor.float() / 255.0  # uint8 to float and normalize to [0, 1]

    return tensor

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Save checkpoint if a new best is achieved"""
    print("=> Saving a new best")
    torch.save(state, filename) 

def main():
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"./checkpoints/{start_time}"
    os.makedirs(checkpoint_dir, exist_ok=True) 

    transform = Compose([
        Resize((256, 256)),
        Lambda(dynamic_normalize),
    ])
    
    parser = argparse.ArgumentParser(description="Train model with optional checkpoint")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a checkpoint to resume training')
    args = parser.parse_args()

    dataset = AnimationColorizationDataset(root_dir='./data', animes=TRAIN_ANIMES, transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = ImageColorizationTransformer(sketch_dim=256*256, color_dim=3*256*256, d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    l1_criterion = nn.L1Loss()
    # for perceptual loss & style loss
    vgg_model = vgg19(weights=VGG19_Weights.DEFAULT).to(device)
    vgg_model.eval()
    layer_indices_VGG19 = [1, 6, 11, 20, 29]  # ReLU 1_1, ReLU 2_1, ReLU 3_1, ReLU 4_1, ReLU 5_1
    start_epoch = 0

    lambda_p = 1
    lambda_s = 1000
    lambda_l1 = 10

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  

    if args.checkpoint:
        print(f"Resuming training from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']  
        loss = checkpoint['loss']

    num_epochs = 80
    save_count = 10
    best_loss = float('inf') 

    l1_losses = []
    perceptual_losses = []
    style_losses = []
    print(f'Total epochs: {num_epochs}')

    for epoch in range(start_epoch,num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}") 

        for i, (ref1_sketch, ref1_color, ref2_sketch, ref2_color, target_sketch, target_color) in enumerate(dataloader):
            ref1_sketch, ref1_color, ref2_sketch, ref2_color, target_sketch, target_color = ref1_sketch.to(device), ref1_color.to(device), ref2_sketch.to(device), ref2_color.to(device), target_sketch.to(device), target_color.to(device)

            
            outputs = model(target_sketch, ref1_sketch, ref1_color, ref2_sketch, ref2_color)
        
            l1_loss = l1_criterion(outputs, target_color)
            p_loss, s_loss = perceptual_style_loss(vgg_model, target_color, outputs, layer_indices_VGG19, lambda_p, lambda_s)

            loss = lambda_l1 * l1_loss + lambda_p * p_loss + lambda_s * s_loss

            l1_loss_val = l1_loss.item()
            p_loss_val = p_loss.item()
            s_loss_val = s_loss.item()

            l1_losses.append(l1_loss_val)
            perceptual_losses.append(p_loss_val)
            style_losses.append(s_loss_val)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            progress_bar.update(1)
            progress_bar.set_description(f"Epoch {epoch+1} Total Loss(ave): {running_loss / (i + 1):.4f} / L1: {lambda_l1 * l1_loss_val:.4f} / perc: {lambda_p * p_loss_val:.4f} / style: {lambda_s * s_loss_val:.4f}")

            torch.cuda.empty_cache()
            

        print(f'Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}')

        is_best = running_loss < best_loss 
        best_loss = min(running_loss, best_loss)
        if is_best:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': running_loss,
            }, filename=checkpoint_path)

    print('Finished Training')
    plot_and_save_loss(l1_losses, 'L1', 'l1_loss')
    plot_and_save_loss(perceptual_losses, 'Perceptual', 'perceptual_loss')
    plot_and_save_loss(style_losses, 'Style', 'style_loss')

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    main()

