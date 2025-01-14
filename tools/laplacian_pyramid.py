import numpy as np
import torch
import cv2
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import rgb_to_grayscale
from easydict import EasyDict
from isp.sharpen_torch_2_0 import unsharp_mask, adjust_sharpness

import matplotlib.pyplot as plt
from PIL import Image


def gaussian_blur_(image, kernel_size=5, sigma=1.0):
    # Create a Gaussian kernel for blurring the image
    grid = torch.arange(kernel_size).float() - kernel_size // 2
    gaussian_kernel = torch.exp(-(grid ** 2) / (2 * sigma ** 2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    gaussian_kernel = gaussian_kernel.view(1, 1, -1)  # Reshape to (1, 1, kernel_size)

    # Apply 1D convolution in both x and y directions (separable filter)
    image = F.conv2d(image, gaussian_kernel.unsqueeze(0), padding=kernel_size // 2)
    image = F.conv2d(image, gaussian_kernel.unsqueeze(1), padding=kernel_size // 2)
    return image

def downsample_avgpool(image, H, W):
    ds_func = nn.AdaptiveAvgPool2d((H, W))
    return ds_func(image)

def get_gaussian_kernel1d(kernel_size: int, sigma: float):
    # Create a 1D Gaussian kernel
    x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    gaussian = torch.exp(-0.5 * (x / sigma)**2)
    gaussian /= gaussian.sum()
    return gaussian

def gaussian_blur(input_tensor: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    B, C, H, W = input_tensor.shape
    
    # Create the 1D Gaussian kernel
    kernel_1d = get_gaussian_kernel1d(kernel_size, sigma).to(input_tensor.device)
    
    # Construct the 2D Gaussian kernel by computing the outer product of the 1D kernel
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    
    # Reshape the kernel for depthwise convolution (so it applies the same kernel to each channel)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    
    # Reshape the kernel for depthwise convolution
    kernel_2d = kernel_2d.expand(C, 1, kernel_size, kernel_size)
    
    # Apply the Gaussian blur in 2D
    padding = kernel_size // 2
    input_padded = F.pad(input_tensor, pad=(padding, padding, padding, padding), mode='reflect')

    blurred = F.conv2d(input_padded, kernel_2d, padding=0, groups=C)
    
    return blurred

def downsample(image):
    ds_func = nn.AdaptiveAvgPool2d((image.shape[2] // 2, image.shape[3] // 2))
    return ds_func(image)

def upsample(image, target_size):
    return F.interpolate(image, size=target_size, mode='bilinear', align_corners=False)

def laplacian_pyramid_layers(image, num_downsamples):
    # input image of size B x C x H x W
    image = rgb_to_grayscale(image)
    original_size = image.shape[2:]  # Store original image size (height, width)
    current_image = image.clone()
    lap_layers = []
    
    # Build the Laplacian pyramid
    for _ in range(num_downsamples):
        blurred_image = gaussian_blur(current_image)
        downsampled_image = downsample(blurred_image)
        
        # Upsample to the size of the current image
        upsampled_image = upsample(downsampled_image, current_image.shape[2:])
        
        # Compute the Laplacian layer (high-frequency information)
        laplacian_layer = current_image - upsampled_image
        lap_layers.append((0.5 + 0.5 * laplacian_layer).clamp(0, 1))
        # Update the current image to the downsampled version
        current_image = downsampled_image
    
    return lap_layers

def get_laplacian_first(image, num_downsamples):
    # input image of 1 3 256 256, num_downsamples = 3, output laplacian of 1 1 64 64
    lap_layers = laplacian_pyramid_layers(image, num_downsamples)
    H_last = lap_layers[-1].shape[2]
    W_last = lap_layers[-1].shape[3]
    return downsample_avgpool(lap_layers[0], H_last, W_last)

def get_laplacian_last(image, num_downsamples):
    # input image of 1 3 256 256, num_downsamples = 3, output laplacian of 1 1 64 64
    lap_layers = laplacian_pyramid_layers(image, num_downsamples)
    return lap_layers[-1]

def get_laplacian_all(image, num_downsamples):
    # input image of 1 3 256 256, num_downsamples = 3, output laplacian of 1 3 64 64
    lap_layers = laplacian_pyramid_layers(image, num_downsamples)
    H_last = lap_layers[-1].shape[2]
    W_last = lap_layers[-1].shape[3]
    downsampled_lap_layers = []
    for lap_layer in lap_layers:
        downsampled_lap_layers.append(downsample_avgpool(lap_layer, H_last, W_last))
    return torch.cat(downsampled_lap_layers, dim=1)

def get_laplacian_option(image, num_downsamples, option):
    if option == 0:
        return None
    elif option == 1:
        return get_laplacian_first(image, num_downsamples)
    elif option == 2:
        return get_laplacian_last(image, num_downsamples)
    elif option == 3:
        return get_laplacian_all(image, num_downsamples)
    else:
        raise NotImplementedError("unsupported laplacian pyramid option")

image_path = '/ailab/user/wujiarui/data/custom_isp_data/multi_isp/examples/high_shad/ExpertC0019-0-Best-Input.jpg'
image = Image.open(image_path)
transform_gray = T.Compose([
    T.ToTensor(),   # Convert image to tensor
    T.Resize((256, 256))  # Resize to 256x256
])
gray_image = transform_gray(image).unsqueeze(0)  # Add batch dimension
B, C, H, W = gray_image.shape
print(B,C,H,W)
num_downsamples = 3  # 256 -> 128 -> 64


print(get_laplacian_first(gray_image, num_downsamples).shape)
print(get_laplacian_last(gray_image, num_downsamples).shape)
print(get_laplacian_all(gray_image, num_downsamples).shape)
print("..........")



laplacian_pyramids = laplacian_pyramid_layers(gray_image, num_downsamples)
for lap in laplacian_pyramids:
    print(lap.min(), lap.max())


last_laplacian = laplacian_pyramids[0]
print(last_laplacian.shape)
# Convert the Laplacian layer to a numpy array and display it
last_laplacian = downsample_avgpool(last_laplacian, H//4, W//4)
output_image = last_laplacian.squeeze(0).permute(1, 2, 0).detach().numpy()
# Display the result
plt.imshow(output_image, cmap='gray')
plt.title("Laplacian Pyramid Last Layer")
plt.axis('off')
plt.savefig("/ailab/user/wujiarui/data/custom_isp_data/multi_isp/examples/high_shad/ExpertC0019-0_lap.jpg")


last_laplacian = laplacian_pyramids[1]
print(last_laplacian.shape)
# Convert the Laplacian layer to a numpy array and display it
last_laplacian = downsample_avgpool(last_laplacian, H//4, W//4)
output_image = last_laplacian.squeeze(0).permute(1, 2, 0).detach().numpy()
# Display the result
plt.imshow(output_image, cmap='gray')
plt.title("Laplacian Pyramid Last Layer")
plt.axis('off')
plt.savefig("/ailab/user/wujiarui/data/custom_isp_data/multi_isp/examples/high_shad/ExpertC0019-1_lap.jpg")


last_laplacian = laplacian_pyramids[-1]
print(last_laplacian.shape)
# Convert the Laplacian layer to a numpy array and display it
output_image = last_laplacian.squeeze(0).permute(1, 2, 0).detach().numpy()
# Display the result
plt.imshow(output_image, cmap='gray')
plt.title("Laplacian Pyramid Last Layer")
plt.axis('off')
plt.savefig("/ailab/user/wujiarui/data/custom_isp_data/multi_isp/examples/high_shad/ExpertC0019-last_lap.jpg")