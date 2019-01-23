import numpy as np
import cv2 as cv
import sys
from h5 import h5Handler
from mylib import read_frame

cache_size = 2000

cu_size = int(sys.argv[1])
scale = int(sys.argv[2])
f_cnt = int(sys.argv[3])
dec_name = sys.argv[4]
gt_name = sys.argv[5]
height = int(sys.argv[6])
width = int(sys.argv[7])
h5_name = "../../train/s" + str(cu_size) + "_m" + str(scale) + '.h5'


cu_pixel = cu_size * cu_size
input_size = cu_pixel * (2 * scale - 1)
label_size = cu_pixel
middle_size  = cu_pixel * scale
input_cache = np.zeros([cache_size, cu_size * scale, cu_size * scale])
label_cache = np.zeros([cache_size, cu_size, cu_size])
cache_cnt = 0

fid = 0
h5er = h5Handler(h5_name)

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
            input_cache[cache_cnt, :, :] = Y[ly:ly+cu_size * scale, lx:lx+cu_size*scale]
            label_cache[cache_cnt, :, :] = YY[ly+cu_size:ly+cu_size*2, lx+cu_size:lx+cu_size*2]

            cache_cnt = cache_cnt + 1
            if cache_cnt == cache_size:
                input_cache = input_cache / 255.0
                label_cache = label_cache / 255.0
                if fid == 0: # create mode
                    h5er.write(input_cache, label_cache, create=True)
                    fid = 1
                else:
                    h5er.write(input_cache, label_cache, create=False)
                cache_cnt = 0
    print("Finish getting data from frame: %d"%(i))                
