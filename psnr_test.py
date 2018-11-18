from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import numpy as np


def test_quality(gt, pred):
    qr = psnr(gt.astype(np.uint8), pred.astype(np.uint8))
    sm = ssim(gt.astype(np.uint8), pred.astype(np.uint8), multichannel = True)
    return {'psnr': qr, 'ssim': sm}