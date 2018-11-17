import caffe
import numpy as np
import cv2
import os
from h5 import h5Handler

root_path = '/home/hyz/lab/intra/'
deploy_path ='network/deploy.prototxt'
model_path = 'network/angle.caffemodel'
# img_path = 'img/'
# target_path = 'target/'
gt_path = '../gt/'
pred_path = '../pred/'
h5_path = '/home/hyz/angle_test.h5'

# initilization
deploy_path = root_path + deploy_path
model_path = root_path + model_path
# img_path = root_path + img_path
net = caffe.Net(deploy_path, model_path, caffe.TEST)
# finish initilization

#--------------- Abandoned, for img test in a folder-------------------
# for file_name in os.listdir(img_path):
#     img = cv2.imread(img_path + file_name)
#     img = img[:,:,0]
#     vec = np.zeros([1, 3072, 1, 1], dtype=float)
#     vec[:,:2048,:,:] = img[0:32,0:64].reshape([1,2048,1,1])
#     vec[:,2048:,:,:] = img[32:,0:32].reshape([1,1024,1,1])
#     vec = vec / 255.0
#     net.blobs['data'].data[...] = vec
#     out = net.forward()
#     out = out['efc4'].reshape([32,32], order='F') * 255.0
#     img[32:,32:] = out[...]
#     cv2.imwrite(target_path + file_name, img)
#--------------- Abandoned, for img test in a folder-------------------

# 1-sampe refer to one picture
st_id = 0
ed_id = 10000
sample_number = ed_id - st_id
stride = 10
handler = h5Handler(h5_path)
datas = handler.read('data', st_id, ed_id, stride)
labels = handler.read('label', st_id, ed_id, stride)

img = np.zeros([64, 64])
for i in range(sample_number):
    data = datas[i:i+1, :, :, :]
    label = labels[i:i+1, :, :, :]
    img[:32, :] = data[:, :2048, :, :].reshape([32, 64], order='F') * 255.0
    img[32:64, :32] = data[:, 2048:, :, :].reshape([32, 32], order='F') * 255.0
    img[32:64, 32:64] = label.reshape([32, 32], order='F') * 255.0
    # write img to the gt directory
    # for normal data, we consider nothing
    print(i)
    if np.var(img) > 200:
        cv2.imwrite(gt_path + str(i) + '.png', img)
        net.blobs['data'].data[...] = data
        vec = net.forward()
        vec = vec['efc4'].reshape([32, 32], order='F') * 255.0
        img[32:, 32:] = vec[...]
        # write pred_img to the pred directory
        cv2.imwrite(pred_path + str(i) + '.png', img)
    
