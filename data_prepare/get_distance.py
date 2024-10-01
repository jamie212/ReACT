import numpy as np
import imageio.v2 as imageio
import os
import argparse
from scipy.ndimage import gaussian_filter
from scipy.ndimage import distance_transform_edt
import cv2
from skimage.color import rgb2gray


def distance_field(path): ## snowy's
    img = imageio.imread(path, pilmode='RGBA')
    
    img = np.clip(np.float64(img) / 255, 0, None) # 將元素限制在max, min之間
    mask = np.zeros((img.shape[0], img.shape[1], 4)) # (256, 256, 4)
    mask[img < 0.95] = 1.0 # 把線條部分標成1
    img = np.expand_dims(mask[:, :, 0], axis=2)
    # sdf = snowy.unitize(snowy.generate_sdf(img != 0.0))
    sdf = unitize(generate_sdf(img != 0.0)) # input a boolean (256, 256, 1) map, True on 線條, False on 空白處
    return sdf

def calculate_density(img_binary, sigma=1):
    density = gaussian_filter(img_binary.astype(float), sigma=sigma)
    return density

def adjust_distance_by_density(distances, density, density_threshold=0.5, adjustment_factor=1.5):

    aidjusted_distances = distances.copy()
    
    high_density_mask = density > density_threshold
    adjusted_distances[high_density_mask] *= adjustment_factor
    
    return adjusted_distances

def normalize(data):
    return (data - np.amin(data)) / (np.amax(data) - np.amin(data))

def my_distance(path): 
    
    img = imageio.imread(path)
    # print(img.shape)
    if img.ndim > 2:
        img = img.mean(axis=2)
    # print(img.shape)
    
    img_normalized = img / 255.0
    
    threshold = 0.95
    img_binary = (img_normalized < threshold).astype(int) # 線條部分設成1，白色部分設成0

    density = calculate_density(img_binary, sigma=7) 

    normalized_density = normalize(density) 

    distance = distance_transform_edt(1 - img_binary) 
    # distance = normalize(distance)

    distance = distance * (1 - normalized_density)*0.05 + img_normalized * normalized_density #3
    distance = normalize(distance)
    distance = np.sqrt(distance) # make all brighter

    return distance



def save_binary_image(binary_image, output_path):
    img_to_save = (binary_image) * 255
    imageio.imwrite(output_path, img_to_save.astype('uint8'))

def process_folder(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.png')  

            dist_image = my_distance(input_path)

            save_binary_image(dist_image, output_path)
            print(f"get dist: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate distance fields for images in a folder.')
    parser.add_argument('input_dir', type=str, help='Path to the input directory')
    parser.add_argument('output_dir', type=str, help='Path to the output directory')
    args = parser.parse_args()
    input_dir = os.path.normpath(args.input_dir)
    output_dir = os.path.normpath(args.output_dir)

    process_folder(input_dir, output_dir)
