import cv2
import h5py
import numpy as np
import sys
import os
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

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


# Assume that our video is I420
def read_frame(filename, idx, _height, _width, mode=0):
    pixel_num = _height * _width
    byte_num = int(pixel_num * 3 / 2)
    # print(byte_num)
    with open(filename, 'rb') as f:
        f.seek(idx * byte_num, 0)
        # only luma mode
        if mode == 0:
                data = np.fromfile(f, dtype=np.uint8, count=pixel_num)
                return data.reshape([_height, _width])

        else:
                # Three color mode
                dataY = np.fromfile(f, dtype=np.uint8, count=pixel_num)
                dataU = np.fromfile(f, dtype=np.uint8, count=int(pixel_num / 4))
                dataV = np.fromfile(f, dtype=np.uint8, count=int(pixel_num / 4))
                img = np.zeros([3, _height, _width])
                img[0,:,:] = dataY.reshape([_height, _width])
                img[1,0::2,0::2] = dataU.reshape([int(_height / 2), int(_width / 2)])
                img[1,0::2,1::2] = img[1,0::2,0::2]
                img[1,1::2,0::2] = img[1,0::2,0::2]
                img[1,1::2,1::2] = img[1,0::2,0::2]
                img[2,0::2,0::2] = dataV.reshape([int(_height / 2), int(_width / 2)])
                img[2,0::2,1::2] = img[2,0::2,0::2]
                img[2,1::2,0::2] = img[2,0::2,0::2]
                img[2,1::2,1::2] = img[2,0::2,0::2]
                img = img.astype(np.uint8)
                img = img.transpose(1,2,0)
                print(img.dtype)
                print('---', img.shape)
                img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
                return img