import caffe
import numpy as np
import cv2 as cv
from psnr_test import test_quality

deploy_path ='../network/deploy.prototxt'
dcmodel_path = '../network/dc.caffemodel'
planarmodel_path = '../network/planar.caffemodel'
anglemodel_path = '../network/angle.caffemodel'

img = np.zeros([64, 64])
dcnet = caffe.Net(deploy_path, dcmodel_path, caffe.TEST)
planarnet = caffe.Net(deploy_path, planarmodel_path, caffe.TEST)
anglenet = caffe.Net(deploy_path, anglemodel_path, caffe.TEST)

one_block = np.ones([32,32])
input = np.zeros([1,3072,1,1])

for i in range(8, 16):
    for j in range(8, 56):
        img[i][j] = 1

for i in range(48, 56):
    for j in range(8, 56):
        img[i][j] = 1

for i in range(8, 56):
    for j in range(8, 16):
        img[i][j] = 1

for i in range(8, 56):
    for j in range(48, 56):
        img[i][j] = 1

# img[32:,:32] = one_block
# img[:32,32:] = one_block
tmp_img = img.copy()
tmp_img *= 255.0


input[:,:2048,:,:] = img[:32,:].reshape([1,2048,1,1], order='F')
input[:,2048:,:,:] = img[32:,:32].reshape([1,1024,1,1], order='F')
dcnet.blobs['data'].data[...] = input
planarnet.blobs['data'].data[...] = input
anglenet.blobs['data'].data[...] = input
img *= 255.0

vec = dcnet.forward()
vec = vec['efc4']
img[32:,32:] = vec.reshape([32,32], order='F') * 255.0
cv.imwrite('../test/dc.png', img)
print(test_quality(tmp_img, img))

vec = planarnet.forward()
vec = vec['efc4']
img[32:,32:] = vec.reshape([32,32], order='F') * 255.0
cv.imwrite('../test/planar.png', img)
print(test_quality(tmp_img, img))

vec = anglenet.forward()
vec = vec['efc4']
img[32:,32:] = vec.reshape([32,32], order='F') * 255.0
cv.imwrite('../test/angle.png', img)
print(test_quality(tmp_img, img))

cv.imwrite('../test/gt.png', tmp_img)