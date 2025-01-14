import torch
import torch.nn.functional as F
import numpy as np


def rgb_to_yuv(image):
    """ Convert an RGB image to YUV space. """
    matrix = torch.tensor([[0.299, 0.587, 0.114],
                           [-0.14713, -0.28886, 0.436],
                           [0.615, -0.51499, -0.10001]]).to(image.device)
    return torch.matmul(image.permute(1, 2, 0), matrix.T).permute(2, 0, 1)


def batch_rgb_to_yuv(images):
    """ Convert a batch of RGB images to YUV space. """
    matrix = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.14713, -0.28886, 0.436],
        [0.615, -0.51499, -0.10001]
    ], dtype=torch.float32, device=images.device)
    # The permute rearranges the dimensions to apply matmul correctly
    return torch.matmul(images.permute(0, 2, 3, 1), matrix.T).permute(0, 3, 1, 2)


def batch_histogram(images, bins, yuv=False):
    """ Compute histograms for a batch of images in all channels, individually per image. """
    if yuv:
        images = batch_rgb_to_yuv(images)
        # Adjust U and V components to the range [0, 1]
        images[:, 1:, :, :] += 0.5
    B, C, H, W = images.shape
    histograms = torch.zeros((B, C, bins), device=images.device)
    # Flatten the images for histogram computation
    for b in range(B):
        for c in range(C):
            histograms[b, c] = torch.histc(images[b, c].flatten(), bins=bins, min=0, max=1)
    # Normalize the histograms to avoid division by zero
    histograms /= (histograms.sum(dim=-1, keepdim=True) + 1e-10)
    return histograms.reshape(B, C * bins)


def batch_histograms1(data_tensor, num_classes, yuv=False):
    """
    Compute histograms for each image in a batch independently using fully parallel operations.
    Input images: tensor @ [B, 3, H, W]
    """
    if yuv:
        data_tensor = batch_rgb_to_yuv(data_tensor)
        # Adjust U and V components to the range [0, 1]
        data_tensor[:, 1:, :, :] += 0.5
    B, C, H, W = data_tensor.shape
    data_tensor = data_tensor.view(B, C, -1)
    data_tensor = torch.round(data_tensor * (num_classes - 1)).long()
    nc = num_classes
    maxd = data_tensor.max()
    hist = torch.zeros((*data_tensor.shape[:-1], nc), dtype=data_tensor.dtype, device=data_tensor.device)
    ones = torch.tensor(1, dtype=hist.dtype, device=hist.device).expand(data_tensor.shape)
    # hist.scatter_add_(-1, ((data_tensor * nc) // (maxd+1)).long(), ones)
    hist.scatter_add_(-1, torch.clamp(data_tensor, 0, num_classes - 1), ones)

    hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-10)
    return hist.reshape(B, C * num_classes)


def soft_histogram(image, bins, sigma=0):
    """ Compute soft histogram for all channels. """
    components = image.reshape(3, -1)
    components[1:] += 0.5  # Translate U and V components
    histograms = []
    for component in components:
        histogram = torch.histc(component, bins=bins, min=0, max=1)
        histogram /= (histogram.sum() + 1e-10)  # Normalize the histogram
        histograms.append(histogram)
    return histograms


def histogram_matching_loss(img1, img2, bins=64, sigma=0.05):
    """
    Calculate the histogram matching loss for all Y, U, V channels.
    input image: tensor @ [3, H, W]
    """
    yuv1 = rgb_to_yuv(img1)
    yuv2 = rgb_to_yuv(img2)

    hist1 = soft_histogram(yuv1, bins, sigma)
    hist2 = soft_histogram(yuv2, bins, sigma)

    l2_losses = []
    intersections = []
    for h1, h2 in zip(hist1, hist2):
        l2_losses.append(F.mse_loss(h1, h2))
        intersections.append(torch.min(h1, h2).sum())

    return l2_losses[0], (l2_losses[1] + l2_losses[2])