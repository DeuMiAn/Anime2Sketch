"""Test script for anime-to-sketch translation
Example:
    python3 test.py --dataroot /your_path/dir --load_size 512
    python3 test.py --dataroot /your_path/img.jpg --load_size 512
"""

import torch
import os
from data import get_image_list, read_img_path, tensor_to_img, save_image
from model import create_model
import argparse
from tqdm.auto import tqdm
from kornia.enhance import equalize_clahe
from torchvision.transforms import GaussianBlur
import torch.nn.functional as F
import cv2
import numpy as np

def apply_gaussian_blur(img_tensor, kernel_size=5):
    blur = GaussianBlur(kernel_size=kernel_size)
    return blur(img_tensor)

def apply_sharpening(img_tensor, alpha=1.5):
    blur = GaussianBlur(kernel_size=3)
    blurred = blur(img_tensor)
    return torch.clamp(img_tensor + alpha * (img_tensor - blurred), -1, 1)

def apply_clahe(img_tensor, clip_limit=2.0):
    img_tensor = (img_tensor + 1) / 2.0
    img_tensor = equalize_clahe(img_tensor, clip_limit=clip_limit)
    img_tensor = (img_tensor * 2.0) - 1
    return img_tensor

def apply_canny_edge_detection(img_tensor, threshold1=100, threshold2=200):
    """
    Apply Canny edge detection to the input image tensor.

    Parameters:
        img_tensor (torch.Tensor): Input image tensor with shape (1, C, H, W).
        threshold1 (int): First threshold for the hysteresis procedure.
        threshold2 (int): Second threshold for the hysteresis procedure.

    Returns:
        torch.Tensor: Image tensor with edges detected.
    """
    # Convert to grayscale by averaging the color channels
    img_gray = img_tensor.mean(dim=1, keepdim=True).squeeze().cpu().numpy()
    
    # Convert image to [0, 255] range
    img_numpy = (img_gray * 0.5 + 0.5) * 255
    img_numpy = img_numpy.astype(np.uint8)

    # Apply Canny edge detection
    edges = cv2.Canny(img_numpy, threshold1, threshold2)
    
    # Convert back to [-1, 1] range
    edges = edges / 255.0 * 2.0 - 1.0

    # Expand edges to match img_tensor's 3-channel format
    edges_tensor = torch.tensor(edges).unsqueeze(0).repeat(3, 1, 1)
    edges_tensor = edges_tensor.unsqueeze(0)  # Add batch dimension
    return edges_tensor.to(img_tensor.device)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anime-to-sketch test options.')
    parser.add_argument('--dataroot','-i', default='test_samples/', type=str)
    parser.add_argument('--load_size','-s', default=512, type=int)
    parser.add_argument('--output_dir','-o', default='results/', type=str)
    parser.add_argument('--gpu_ids', '-g', default=[], help="gpu ids: e.g. 0 0,1,2 0,2.")
    parser.add_argument('--model', default="default", type=str, help="variant of model to use. you can choose from ['default','improved']")
    parser.add_argument('--clahe_clip', default=-1, type=float, help="clip threshold for CLAHE, set to -1 to disable")
    parser.add_argument('--canny_threshold1', default=100, type=int, help="First threshold for Canny edge detection.")
    parser.add_argument('--canny_threshold2', default=200, type=int, help="Second threshold for Canny edge detection.")
    opt = parser.parse_args()

    # create model
    gpu_list = ','.join(str(x) for x in opt.gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    device = torch.device('cuda' if len(opt.gpu_ids) > 0 else 'cpu')
    model = create_model(opt.model).to(device)      # create a model given opt.model and other options
    model.eval()

    # get input data
    if os.path.isdir(opt.dataroot):
        test_list = get_image_list(opt.dataroot)
    elif os.path.isfile(opt.dataroot):
        test_list = [opt.dataroot]
    else:
        raise Exception("{} is not a valid directory or image file.".format(opt.dataroot))
    
    # save outputs
    save_dir = opt.output_dir
    os.makedirs(save_dir, exist_ok=True)
    
    for test_path in tqdm(test_list):
        basename = os.path.basename(test_path)
        aus_path = os.path.join(save_dir, basename)
        img, aus_resize = read_img_path(test_path, opt.load_size)


        # 1. Apply CLAHE
        if opt.clahe_clip > 0:
            img = apply_clahe(img, clip_limit=opt.clahe_clip)

        # 2. Apply Gaussian Blur
        img = apply_gaussian_blur(img, kernel_size=5)

        # 3. Apply Canny Edge Detection
        edges = apply_canny_edge_detection(img, threshold1=opt.canny_threshold1, threshold2=opt.canny_threshold2)

        # 4. Combine original image with edges to emphasize contours
        α = 0.133  # Adjust α to control the strength of edges
        img = torch.clamp((1 - α) * img + α * edges, -1, 1)

        # 5. Apply Sharpening
        img = apply_sharpening(img, alpha=1.5)

        # 6. Convert the img tensor to float32 type
        img = img.to(torch.float32)
      

        aus_tensor = model(img.to(device))
        aus_img = tensor_to_img(aus_tensor)
        save_image(aus_img, aus_path, aus_resize)