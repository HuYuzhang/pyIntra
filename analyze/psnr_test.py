from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import numpy as np

# gt and pred can be one sample or many sample
def test_quality(gt, pred):
    shape = gt.shape
    if len(shape) == 3:
        psnr_s = []
        ssim_s = []
        for i in range()
    else if len(shape) == 2:
        qr = psnr(gt.astype(np.uint8), pred.astype(np.uint8))
        sm = ssim(gt.astype(np.uint8), pred.astype(np.uint8), multichannel = True)
        return {'psnr': qr, 'ssim': sm}