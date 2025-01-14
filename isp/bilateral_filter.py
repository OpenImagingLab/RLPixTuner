"""
Example script: Simple bilateral filtering for image denoising.

Author: Fabian Wagner
Contact: fabian.wagner@fau.de
"""
import matplotlib.pyplot as plt
import torch
from bilateral_filter_layer import BilateralFilter3d
import time
from skimage.data import camera
import torch.nn.functional as F
import numpy as np

def bilter_filter_3d_gray():
    #############################################################
    ####             PARAMETERS (to be modified)             ####
    #############################################################
    # Set device.
    use_gpu = True
    # Filter parameters.
    sigma_x = 1.5
    sigma_y = 1.5
    sigma_z = 1.0
    sigma_r = 0.9
    # Image parameters.
    downsample_factor = 2
    n_slices = 1
    #############################################################

    if use_gpu:
        dev = "cuda"
    else:
        dev = "cpu"

    # Initialize filter layer.
    layer_BF = BilateralFilter3d(sigma_x, sigma_y, sigma_z, sigma_r, use_gpu=use_gpu)

    # Load cameraman image.
    image = camera()[::downsample_factor, ::downsample_factor]
    tensor_gt = torch.tensor(image).unsqueeze(2).repeat(1, 1, n_slices).unsqueeze(0).unsqueeze(0)
    tensor_gt = tensor_gt / torch.max(tensor_gt)

    # Prepare noisy input.
    noise = 0.1 * torch.randn(tensor_gt.shape)
    tensor_in = (tensor_gt + noise).to(dev)
    tensor_in.requires_grad = True
    print("Input shape: {}".format(tensor_in.shape))

    # Forward pass.
    start = time.time()
    prediction = layer_BF(tensor_in)
    end = time.time()
    print("Runtime forward pass: {} s".format(end - start))

    # Backward pass.
    loss = prediction.mean()
    start = time.time()
    loss.backward()
    end = time.time()
    print("Runtime backward pass: {} s".format(end - start))

    # Visual results.
    vmin_img = 0
    vmax_img = 1
    idx_center = int(tensor_in.shape[4] / 2)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7, 3))
    axes[0].imshow(tensor_in[0, 0, :, :, idx_center].detach().cpu(), vmin=vmin_img, vmax=vmax_img, cmap='gray')
    axes[0].set_title('Noisy input', fontsize=14)
    axes[0].axis('off')
    axes[1].imshow(prediction[0, 0, :, :, idx_center].detach().cpu(), vmin=vmin_img, vmax=vmax_img, cmap='gray')
    axes[1].set_title('Filtered output', fontsize=14)
    axes[1].axis('off')
    axes[2].imshow(tensor_gt[0, 0, :, :, idx_center].detach().cpu(), vmin=vmin_img, vmax=vmax_img, cmap='gray')
    axes[2].set_title('Ground truth', fontsize=14)
    axes[2].axis('off')
    plt.show()


@torch.no_grad()
def getGaussianKernel(ksize, sigma=0):
    if sigma <= 0:
        # 根据 kernelsize 计算默认的 sigma，和 opencv 保持一致
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center) # 元素与矩阵中心的横向距离
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2)) # 计算一维卷积核
    # 根据指数函数性质，利用矩阵乘法快速计算二维卷积核
    kernel = kernel1d[..., None] @ kernel1d[None, ...]
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum() # 归一化
    return kernel


def GaussianBlur(batch_img, ksize, sigma=None):
    kernel = getGaussianKernel(ksize, sigma) # 生成权重
    B, C, H, W = batch_img.shape # C：图像通道数，group convolution 要用到
    # 生成 group convolution 的卷积核
    kernel = kernel.view(1, 1, ksize, ksize).repeat(C, 1, 1, 1)
    pad = (ksize - 1) // 2 # 保持卷积前后图像尺寸不变
    # mode=relfect 更适合计算边缘像素的权重
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')
    weighted_pix = F.conv2d(batch_img_pad, weight=kernel, bias=None,
                           stride=1, padding=0, groups=C)
    return weighted_pix


def bilateralFilter(batch_img, ksize, sigmaColor=None, sigmaSpace=None):
    device = batch_img.device
    if sigmaSpace is None:
        sigmaSpace = 0.15 * ksize + 0.35  # 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    if sigmaColor is None:
        sigmaColor = sigmaSpace

    pad = (ksize - 1) // 2
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')

    # batch_img 的维度为 BxcxHxW, 因此要沿着第 二、三维度 unfold
    # patches.shape:  B x C x H x W x ksize x ksize
    patches = batch_img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    patch_dim = patches.dim()  # 6
    # 求出像素亮度差
    diff_color = patches - batch_img.unsqueeze(-1).unsqueeze(-1)
    # 根据像素亮度差，计算权重矩阵
    weights_color = torch.exp(-(diff_color ** 2) / (2 * sigmaColor ** 2))
    # 归一化权重矩阵
    weights_color = weights_color / weights_color.sum(dim=(-1, -2), keepdim=True)

    # 获取 gaussian kernel 并将其复制成和 weight_color 形状相同的 tensor
    weights_space = getGaussianKernel(ksize, sigmaSpace).to(device)
    weights_space_dim = (patch_dim - 2) * (1,) + (ksize, ksize)
    weights_space = weights_space.view(*weights_space_dim).expand_as(weights_color)

    # 两个权重矩阵相乘得到总的权重矩阵
    weights = weights_space * weights_color
    # 总权重矩阵的归一化参数
    weights_sum = weights.sum(dim=(-1, -2))
    # 加权平均
    weighted_pix = (weights * patches).sum(dim=(-1, -2)) / weights_sum
    return weighted_pix


def gen_image_pair(image_path, device, sigma):
    import skimage
    # :return: two tensors with shape (1, 3, H, W) in [0, 1] range
    clean_rgb = skimage.io.imread(image_path).astype(np.float32) / 255.0
    import cv2
    clean_rgb = cv2.resize(clean_rgb, (512, 512))
    # additive white gaussian noise
    awgn = np.random.normal(0, scale=sigma, size=clean_rgb.shape).astype(np.float32)
    noisy_rgb = np.clip(clean_rgb + awgn, 0, 1)

    clean_rgb = torch.from_numpy(clean_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    noisy_rgb = torch.from_numpy(noisy_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return clean_rgb, noisy_rgb


if __name__ == "__main__":
    from PIL import Image
    import cv2
    import rawpy

    dev = torch.device('cuda')
    # raw = rawpy.imread("images/LRM_20191230_052456_1.dng")
    # # raw.raw_pattern[:] = np.array([[0, 1], [2, 1]], np.uint8)
    # # raw.color_desc = b"RGGB"
    # image = raw.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD, output_color=rawpy.ColorSpace.sRGB,
    #                         gamma=(1, 1), no_auto_bright=True, output_bps=8, use_camera_wb=False, use_auto_wb=False,
    #                         user_wb=(1, 1, 1, 1))
    # image = np.array(image).astype(np.float32) / 255.
    # image = cv2.resize(image, (1080, 720))
    # noisy_rgb = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(dev)
    # noisy_rgb = torch.cat([noisy_rgb, noisy_rgb], dim=0)

    clean_rgb, noisy_rgb = gen_image_pair('images/000000581781.jpg', device=dev, sigma=0.05)

    noisy_rgb = torch.tensor(noisy_rgb, requires_grad=True)
    denoised_rgb = bilateralFilter(noisy_rgb, ksize=5)
    denoised_rgb.mean().backward()
    print(noisy_rgb.grad)
    exit()

    from util import show
    # show(clean_rgb.detach().cpu().numpy(), title="clean", format="CHW", is_finish=False)
    show(noisy_rgb.detach().cpu().numpy(), title="noisy", format="CHW", is_finish=False)
    show(denoised_rgb.detach().cpu().numpy()[0], title="denoise", format="CHW", is_finish=True)

