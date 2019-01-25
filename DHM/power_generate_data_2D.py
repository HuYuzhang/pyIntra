import numpy as np
import cv2
import sys
from mylib import read_frame, h5Handler
import random
import math


####### Configuration ##########
cache_size = 2000
random_flag = True
random_num = 500
align_flag  = False
mask_mean = True
####### Configuration ##########


cu_size = int(sys.argv[1])
scale = int(sys.argv[2])
f_cnt = int(sys.argv[3])
dec_name = sys.argv[4]
gt_name = sys.argv[5]
height = int(sys.argv[6])
width = int(sys.argv[7])
#h5_name = "../../train/data/s" + str(cu_size) + "_m" + str(scale) + '.h5'
h5_name = sys.argv[8]

cu_pixel = cu_size * cu_size
input_size = cu_pixel * (2 * scale - 1)
label_size = cu_pixel
middle_size  = cu_pixel * scale
input_cache = np.zeros([cache_size, cu_size * scale, cu_size * scale])
label_cache = np.zeros([cache_size, cu_size, cu_size])
cache_cnt = 0


####### Configuration ##########
fid = 0
h5er = h5Handler(h5_name)
x_up = math.floor(width / cu_size - 1)
y_up = math.floor(height / cu_size - 1)
####### Configuration ##########
print("Input size: ", input_size)
if not random_flag:
    for i in range(f_cnt):
        Y = read_frame(dec_name, i, height, width)
        YY = read_frame(gt_name, i, height, width)
        cv2.imwrite('dec.png', Y)
        cv2.imwrite('gt.png', YY)
        for lx in range(0,width,cu_size):
            for ly in range(0,height,cu_size):
                rx = lx + cu_size * scale
                ry = ly + cu_size * scale
                if rx >= width or ry >= height:
                    continue
                # import IPython
                # IPython.embed()
                input_cache[cache_cnt, :, :] = Y[ly:ly+cu_size * scale, lx:lx+cu_size*scale]

                if mask_mean:
                    input_cache[cache_cnt, cu_size:, cu_size:] = 0
                    mean = np.sum(input_cache[cache_cnt, :, :]) / float(input_size)
                    input_cache[cache_cnt, cu_size:, cu_size:] = mean
                else:
                    input_cache[cache_cnt, cu_size:, cu_size:] = 0

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
        # print("Finish getting data from frame: %d"%(i))
else: # we will random get some picture from the picture
    for i in range(f_cnt):
        tmp_num = 0
        Y = read_frame(dec_name, i, height, width)
        YY = read_frame(gt_name, i, height, width)
        while tmp_num < random_num:
            if align_flag:
                lx = random.randint(0, x_up) * cu_size
                ly = random.randint(0, y_up) * cu_size
            else:
                lx = random.randint(0, width - 1)
                ly = random.randint(0, height - 1)

            rx = lx + cu_size * scale
            ry = ly + cu_size * scale
            if rx >= width or ry >= height:
                continue
            # import IPython
            # IPython.embed()
            input_cache[cache_cnt, :, :] = Y[ly:ly+cu_size * scale, lx:lx+cu_size*scale]

            if mask_mean:
                input_cache[cache_cnt, cu_size:, cu_size:] = 0
                mean = np.sum(input_cache[cache_cnt, :, :]) / float(input_size)
                input_cache[cache_cnt, cu_size:, cu_size:] = mean
            else:
                input_cache[cache_cnt, cu_size:, cu_size:] = 0
            
            label_cache[cache_cnt, :, :] = YY[ly+cu_size:ly+cu_size*2, lx+cu_size:lx+cu_size*2]
            tmp_num = tmp_num + 1
            cache_cnt = cache_cnt + 1
            if cache_cnt == cache_size:
                # print("-------->Cache fill, frame: %d, tmpnum: %d"%(i, tmp_num))
                input_cache = input_cache / 255.0
                label_cache = label_cache / 255.0
                if fid == 0: # create mode
                    h5er.write(input_cache, label_cache, create=True)
                    fid = 1
                else:
                    h5er.write(input_cache, label_cache, create=False)
                cache_cnt = 0
        # print("Finish getting data from frame: %d"%(i))
