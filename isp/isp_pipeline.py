import numpy as np
import cv2
import colour_demosaicing
from PIL import Image


def demosaicing(raw, demosaicing_method="Malvar2004", pattern="RGGB"):

    demosaicing_funcs = dict(
        DDFAPD=colour_demosaicing.demosaicing_CFA_Bayer_DDFAPD,
        Menon2007=colour_demosaicing.demosaicing_CFA_Bayer_DDFAPD,
        Malvar2004=colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004,
        bilinear=colour_demosaicing.demosaicing_CFA_Bayer_bilinear,
    )
    demosaicing_func = demosaicing_funcs[demosaicing_method]
    rgb = demosaicing_func(raw, pattern)
    return rgb


def bm3d(x, filter_strength, noise_types, noise_var, lambda_thr3d, mu2):
    # filter_strength = [0, 0.1, ..., 3.0]
    # noise_types = ['gw', 'g0', 'g1', 'g2', 'g3', 'g4', 'g1w', 'g2w', 'g3w', 'g4w']
    # noise_var = [0.01, 0.02, 1.0]
    # lambda_thr3d = [1.0, 2.0, 100.0]
    # mu2 = [0.1, 0.2, ..., 100.0]
    return x


def get_bayer_indices(pattern):
    """
    Get (x_start_idx, y_start_idx) for R, Gr, Gb, and B channels
    in Bayer array, respectively
    """
    return {'gbrg': ((0, 1), (1, 1), (0, 0), (1, 0)),
            'rggb': ((0, 0), (1, 0), (0, 1), (1, 1)),
            'bggr': ((1, 1), (0, 1), (1, 0), (0, 0)),
            'grbg': ((1, 0), (0, 0), (1, 1), (0, 1))}[pattern.lower()]


def split_bayer(bayer_array, bayer_pattern):
    """
    Split R, Gr, Gb, and B channels sub-array from a Bayer array
    :param bayer_array: np.ndarray(H, W)
    :param bayer_pattern: 'gbrg' | 'rggb' | 'bggr' | 'grbg'
    :return: 4-element list of R, Gr, Gb, and B channel sub-arrays, each is an np.ndarray(H/2, W/2)
    """
    rggb_indices = get_bayer_indices(bayer_pattern)

    sub_arrays = []
    for idx in rggb_indices:
        x0, y0 = idx
        sub_arrays.append(
            bayer_array[y0::2, x0::2]
        )
    return sub_arrays


def reconstruct_bayer(sub_arrays, bayer_pattern):
    """
    Inverse implementation of split_bayer: reconstruct a Bayer array from a list of
        R, Gr, Gb, and B channel sub-arrays
    :param sub_arrays: 4-element list of R, Gr, Gb, and B channel sub-arrays, each np.ndarray(H/2, W/2)
    :param bayer_pattern: 'gbrg' | 'rggb' | 'bggr' | 'grbg'
    :return: np.ndarray(H, W)
    """
    rggb_indices = get_bayer_indices(bayer_pattern)

    height, width = sub_arrays[0].shape
    bayer_array = np.empty(shape=(2 * height, 2 * width), dtype=sub_arrays[0].dtype)

    for idx, sub_array in zip(rggb_indices, sub_arrays):
        x0, y0 = idx
        bayer_array[y0::2, x0::2] = sub_array

    return bayer_array


def white_balance(bayer, r_gain, g_gain, b_gain, pattern):
    assert pattern.upper() == "GRBG"
    sub_arrays = split_bayer(bayer, pattern)
    gains = (r_gain, g_gain, g_gain, b_gain)
    wb_sub_arrays = []
    for sub_array, gain in zip(sub_arrays, gains):
        wb_sub_arrays.append(gain * sub_array)
    wb_bayer = reconstruct_bayer(wb_sub_arrays, pattern)
    wb_bayer = np.clip(wb_bayer, 0, 1.0)
    return wb_bayer


def digital_gain(x, gain):
    x = x * gain
    x = np.clip(x, 0, 1.0)
    return x


def color_space_transform(x, ccm):
    matrix = np.array(ccm, dtype=np.float32).T   # (3, 3) right-matrix
    bias = np.zeros((1, 1, 3), np.float32)
    print(matrix.shape)
    rgb_image = x.astype(np.float32)
    ccm_rgb_image = rgb_image @ matrix + bias
    ccm_rgb_image = np.clip(ccm_rgb_image, 0, 1.0)
    return ccm_rgb_image


def sharpen(x, amount, sigma):
    # from DFEImageLib import *
    # #	Set	sharpening	parameters
    # amount = 1.0  # amount	of	sharpening
    # sigma = 0.8  # amount	of	blurring	for	generating	the	mask
    # #	Load	image
    # (h, w, img) = loadPNG("image.png")
    # #	Execute	the	kernel	on	the	DFE
    # (mask, sharp_img) = DFEImageLib_sharpen(h, w, amount, sigma, img)
    pass


def run(x, r_gain, g_gain, b_gain, d_gain, ccm, pattern='GRBG'):
    x = white_balance(x, r_gain, g_gain, b_gain, pattern)
    x = digital_gain(x, d_gain)
    x = demosaicing(x, pattern=pattern)
    img = Image.fromarray(np.uint8(x * 256))
    img.show('demosaic')

    x = color_space_transform(x, ccm)
    img = Image.fromarray(np.uint8(x*256))
    img.show('ccm')


def remosaic(raw_png, pattern='GRBG'):
    assert pattern.upper() == 'GRBG'
    h, w, _ = raw_png.shape
    bayer = np.zeros((h, w), dtype=np.uint16)
    # g r b g
    bayer[::2, ::2] = raw_png[::2, ::2, 1]
    bayer[::2, 1::2] = raw_png[::2, 1::2, 0]
    bayer[1::2, ::2] = raw_png[1::2, ::2, 2]
    bayer[1::2, 1::2] = raw_png[1::2, 1::2, 1]
    return bayer


def main():
    raw_path = '/home/PJLAB/wangyujin/HDD/projects/approx-vision/test/outputs/output_u16.png'
    raw_png = cv2.imread(raw_path, cv2.IMREAD_ANYDEPTH)
    raw_png = cv2.cvtColor(raw_png, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(np.uint8(raw_png / 65535 * 256))
    img.show("raw")

    pattern = 'GRBG'
    bayer = remosaic(raw_png, pattern)

    bayer = np.uint16(bayer / 65535 * 2**12) / 2**12
    # Image.fromarray(bayer).save("./test.tiff")
    # import rawpy
    # raw = rawpy.imread("./test.tiff")
    # rgb = raw.postprocess() # gamma=(1,1), no_auto_bright=True, output_bps=16
    # rgb = Image.fromarray(rgb)
    # rgb.show()

    r_gain, g_gain, b_gain = 2.0, 1.0, 2.0
    d_gain = 1.0
    ccm = [[1.578706, -0.509131, -0.026729], [-0.129182, 1.430037, -0.268639], [0.026785, -0.554129, 1.556603]]

    run(bayer, r_gain, g_gain, b_gain, d_gain, ccm, pattern)


if __name__ == "__main__":
    main()