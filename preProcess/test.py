import cv2
import numpy as np
import sys
import os
from mylib import read_frame
        

path = '../../raw_data/dec.yuv'
height = 1024
width = 1792
idx = int(sys.argv[1])

img = read_frame(path, idx, height, width, mode=1)
cv2.imwrite('../../img/a.png', img)
