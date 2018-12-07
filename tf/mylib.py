# gt and pred can be one sample or many sample
def test_quality(gt, pred):
    shape = gt.shape
    if len(shape) == 3:
        psnr_s = []
        ssim_s = []
        for i in range(shape[0]):
            qr = psnr(gt[i,:,:].astype(np.uint8), pred[i,:,:].astype(np.uint8))
            sm = ssim(gt[i,:,:].astype(np.uint8), pred[i,:,:].astype(np.uint8), multichannel = True)
            psnr_s.append(qr)
            ssim_s.append(sm)
        # Here we will return the mean of the psnrs and ssims
        return np.mean(psnr_s), np.mean(ssim_s)

    elif len(shape) == 2:
        qr = psnr(gt.astype(np.uint8), pred.astype(np.uint8))
        sm = ssim(gt.astype(np.uint8), pred.astype(np.uint8), multichannel = True)
        return qr, sm

def img2input(img):
    img = img / 255.0
    ret = np.zeros([3072])
    gt = np.zeros([1024])
    ret[:2048] = img[:32,:64].reshape([2048])
    ret[2048:] = img[32:,:32].reshape([1024])
    gt = img[32:,32:].reshape([1024])
    return ret, gt