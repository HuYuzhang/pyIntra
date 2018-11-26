import cv2 as cv
import numpy as np
import os
mode = 'angle'
path = '/home/hyz/lab/intra/valid/' + mode
target_path = '/home/hyz/lab/intra/weekly/'
file_list = os.listdir(path)
dump_cnt = 5
img = np.zeros([dump_cnt * 64, 3 * 64])

col = {'planar': 0, 'dc': 1, 'angle': 2}

for f in file_list:
    file_name = f
    f = f[:-4].split('_')
    row = int(f[0])
    img[64*row: 64*(row + 1), 64*(col[f[1]]): 64*(col[f[1]] + 1)] = cv.imread(path + '/' + file_name)[:,:,0]
    
cv.imwrite(target_path + mode + '.png', img)
