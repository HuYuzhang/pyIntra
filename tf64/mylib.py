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

class h5Handler(object):
    def __init__(self, h5_path):
        self.h5_path = h5_path

    def read(self, key, start, end, step):
        fid = h5py.File(self.h5_path, 'r')
        ret = fid[key][start:end:step]
        fid.close()
        return ret

    # right now very bad way to assign 3072 and 1024, but not a big problem
    # assume that datas and labels are of size [n, c, h, w]
    def write(self, datas, labels, create=True):
        if create:
            f = h5py.File(self.h5_path, 'w')
            f.create_dataset('data', data=datas, maxshape=(None, 96, 96), chunks=True, dtype='float32')
            f.create_dataset('label', data=labels, maxshape=(None, 32, 32), chunks=True, dtype='float32')
            f.close()
        else:
            # append mode
            f = h5py.File(self.h5_path, 'a')
            h5data = f['data']
            h5label = f['label']
            cursize = h5data.shape
            addsize = datas.shape

            # # --------------for debug------------------
            # print('-------now begin to add data------')
            # print(cursize)
            # # --------------for debug------------------

            h5data.resize([cursize[0] + addsize[0], 96, 96])
            h5label.resize([cursize[0] + addsize[0], 32, 32])
            h5data[-addsize[0]:,:,:] = datas
            h5label[-addsize[0]:,:,:] = labels
            f.close()