import cv2
import rawpy
from PIL import Image
import imageio
import numpy as np


def rawpy_fun():
    raw = rawpy.imread("/home/PJLAB/wangyujin/HDD/datasets/ISP Pipeline Design/reconfigISP/OnePlusRawDetection/train50/image/LRM_20191230_052425.dng")
    img = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=True, output_bps=8)
    bgr_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    denoised_image = cv2.fastNlMeansDenoisingColored(bgr_image, None, 10, 10, 7, 21)
    denoised_rgb_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)

    cv2.imshow('rgb', denoised_rgb_image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def get_bayer_indices(pattern):
    """
    Get (x_start_idx, y_start_idx) for R, Gr, Gb, and B channels
    in Bayer array, respectively
    """
    return {'gbrg': ((0, 1), (1, 1), (0, 0), (1, 0)),
            'rggb': ((0, 0), (1, 0), (0, 1), (1, 1)),
            'bggr': ((1, 1), (0, 1), (1, 0), (0, 0)),
            'grbg': ((1, 0), (0, 0), (1, 1), (0, 1))}[pattern.lower()]


def reconstruct_bayer(raw, bayer_pattern):
    """
    Inverse implementation of split_bayer: reconstruct a Bayer array from a list of
        R, Gr, Gb, and B channel sub-arrays
    :param raw: 4-element list of R, Gr, Gb, and B channel sub-arrays, each np.ndarray(H/2, W/2)
    :param bayer_pattern: 'gbrg' | 'rggb' | 'bggr' | 'grbg'
    :return: np.ndarray(H, W)
    """
    rggb_indices = get_bayer_indices(bayer_pattern)
    sub_arrays = [raw[::, ::, 0], raw[::, ::, 1], raw[::, ::, 2], raw[::, ::, 3]]
    height, width = sub_arrays[0].shape
    bayer_array = np.empty(shape=(2 * height, 2 * width), dtype=sub_arrays[0].dtype)

    for idx, sub_array in zip(rggb_indices, sub_arrays):
        x0, y0 = idx
        bayer_array[y0::2, x0::2] = sub_array

    return bayer_array



def demosaicing(raw, demosaicing_method="Malvar2004", pattern="RGGB"):
    import colour_demosaicing

    demosaicing_funcs = dict(
        DDFAPD=colour_demosaicing.demosaicing_CFA_Bayer_DDFAPD,
        Menon2007=colour_demosaicing.demosaicing_CFA_Bayer_DDFAPD,
        Malvar2004=colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004,
        bilinear=colour_demosaicing.demosaicing_CFA_Bayer_bilinear,
    )
    demosaicing_func = demosaicing_funcs[demosaicing_method]
    rgb = demosaicing_func(raw, pattern)
    return rgb


def convert_rgbraw_to_tiff():
    # filename = '/home/PJLAB/wangyujin/HDD/projects/DynamicISP/images/000000581781.jpg'
    # image = Image.open(filename)
    # if image.mode != 'RGB':
    #     image = image.convert("RGB")

    img = cv2.imread('/home/PJLAB/wangyujin/HDD/datasets/ISPPipelineDesign/COCO/coco2014v1/train2014/COCO_train2014_000000000009.png')
    print(img)
    img = np.sum(img, axis=-1).astype(np.uint8)
    # cv2.imwrite('images_raw/COCO_train2014_000000000009.tiff', img)
    print(img.shape)
    print(img)
    # h, w = img.shape
    # grbg = np.empty(shape=(h, w, 4), dtype=img.dtype)
    # grbg[::, ::, 0] = img[::2, ::2]
    # grbg[::, ::, 1] = img[::2, ::2]
    # grbg[::, ::, 2] = img[::2, ::2]
    # grbg[::, ::, 3] = img[::2, ::2]
    img = demosaicing(np.array(img, np.float) / 255, pattern='GRBG')
    print(img)

    cv2.imshow('rgb', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # convert_rgbraw_to_tiff()
    import scipy, torchvision, torch
    a = np.arange(50, step=2).reshape((1, 1, 5, 5))
    print(scipy.ndimage.gaussian_filter(a, sigma=1))
    f = torchvision.transforms.functional.gaussian_blur(torch.tensor(a), (3, 3), sigma=1)
    print(f)