import numpy as np
import cv2 as cv
import sys
from h5 import h5Handler
from mylib import read_frame
import h5py

cache_size = 2000

cu_size = int(sys.argv[1])
scale = int(sys.argv[2])
f_cnt = int(sys.argv[3])
dec_name = sys.argv[4]
gt_name = sys.argv[5]
height = int(sys.argv[6])
width = int(sys.argv[7])
h5_name = "s" + str(cu_size) + "_m" + str(scale) + '.h5'


cu_pixel = cu_size * cu_size
input_size = cu_pixel * (2 * scale - 1)
label_size = cu_pixel
middle_size  = cu_pixel * scale
input_cache = np.zeros([input_size, 1, 1])
label_cache = np.zeros([label_size, 1, 1])
cache_cnt = 0

f = h5py.File(h5_name, 'r')
data = f['/data']
label = f['/label']
for i in range(f_cnt):
    Y = read_frame(dec_name, i, height, width)
    YY = read_frame(gt_name, i, height, width)
    for lx in range(0,width,cu_size):
        for ly in range(0,height,cu_size):
            rx = lx + cu_size * scale
            ry = ly + cu_size * scale
            if rx >= width or ry >= height:
                continue
            # import IPython
            # IPython.embed()
            input_cache[0:middle_size, 0, 0]     = Y[ly:ly+cu_size, lx:lx+cu_size*scale].reshape([-1])
            input_cache[middle_size:input_size, 0, 0] = Y[ly+cu_size:ly+cu_size*scale, lx:lx+cu_size].reshape([-1])
            label_cache[:, 0, 0] = YY[ly+cu_size:ly+cu_size*2, lx+cu_size:lx+cu_size*2].reshape([-1])
            
            input_h5 = data[cache_cnt]
            label_h5 = label[cache_cnt]
            for k in range(input_size):
                if int(input_cache[k, 0, 0]) != int(input_h5[k, 0, 0]):
                    print('label error in i: %d, lx: %d, ly: %d, k: %d'%(i, lx, ly, k))
                    print(input_cache[k, 0, 0], input_h5[k, 0, 0])
                    exit(0)
            for k in range(label_size):
                if int(label_cache[k, 0, 0]) != int(label_h5[k, 0, 0]):
                    print('label error in i: %d, lx: %d, ly: %d, k: %d'%(i, lx, ly, k))
                    print(input_cache[k, 0, 0], input_h5[k, 0, 0])
                    exit(0)
            cache_cnt = cache_cnt + 1


    print("Finish test data from frame: %d"%(i))                
