import cv2
import numpy as np
import sys
import os
from mylib import read_frame
import h5py
        

# path = '../../raw_data/dec.yuv'
# height = 1024
# width = 1792
# idx = int(sys.argv[1])

# img = read_frame(path, idx, height, width, mode=1)
# cv2.imwrite('../../img/' + sys.argv[1] + '.png', img)

h5_path = '../../train/planar.h5'
f = h5py.File(h5_path, 'r')
data = f['/data']
label = f['/label']
input = data[2000,:,:,:] * 255.0
gt = label[2000,:,:,:] * 255.0
img = np.zeros([64,64])
img[:32,:64] = input[:2048,:,:].reshape([32,64])
img[32:64,:32] = input[2048:,:,:].reshape([32,32])
img[32:,32:] = gt.reshape([32,32])
cv2.imwrite('../../img/valid.png', img)
