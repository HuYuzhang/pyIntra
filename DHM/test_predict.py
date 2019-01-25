import tensorflow as tf
import  numpy as np
import sys
import h5py
import cv2
from mylib import test_quality


fc = cv2.imread('pred_fc1.png', -1)
rnn = cv2.imread('pred_fc2.png', -1)
gt = cv2.imread('gt.png', -1)

print(test_quality(fc[16:,16:], gt[16:,16:]))
print(test_quality(rnn[16:,16:], gt[16:,16:]))