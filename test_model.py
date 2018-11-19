import caffe
import numpy as np
import cv2 as cv
deploy_path ='../network/deploy.prototxt'
dcmodel_path = '../network/planar.caffemodel'

img = np.zeros([64, 64])
net = caffe.Net(deploy_path, dcmodel_path, caffe.TEST)

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

input[:,:2048,:,:] = img[:32,:].reshape([1,2048,1,1])
input[:,2048:,:,:] = img[32:,:32].reshape([1,1024,1,1])
net.blobs['data'].data[...] = input
vec = net.forward()
vec = vec['efc4']
print(vec)
img[32:,32:] = vec.reshape([32,32])
img *= 255.0
cv.imwrite('dst.png', img)