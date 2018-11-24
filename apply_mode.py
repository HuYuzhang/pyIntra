import caffe
import numpy as np
import cv2
import os
from h5 import h5Handler
from psnr_test import test_quality

root_path = '/home/hyz/lab/intra/'
deploy_path = root_path + 'network/deploy.prototxt'
dcmodel_path = root_path + 'network/dc.caffemodel'
planarmodel_path = root_path + 'network/planar.caffemodel'
anglemodel_path = root_path + 'network/angle.caffemodel'
img_path = root_path + 'img/'
# target_path = 'target/'
gt_path = root_path + 'gt/'
pred_path = root_path + 'pred/'
h5_path = root_path + 'data_set/dc_test.h5'
valid_path = root_path + 'valid/' + 'dc/'# when changing this, remenber to change the h5_path's name

# initilization
dcnet = caffe.Net(deploy_path, dcmodel_path, caffe.TEST)
planarnet = caffe.Net(deploy_path, planarmodel_path, caffe.TEST)
anglenet = caffe.Net(deploy_path, anglemodel_path, caffe.TEST)
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
# We store the gt picture in traget_path, and predicted picture in pred_path
st_id = 0
ed_id = 1000
sample_number = ed_id - st_id
stride = 1
handler = h5Handler(h5_path)
datas = handler.read('data', st_id, ed_id, stride)
labels = handler.read('label', st_id, ed_id, stride)

img = np.zeros([64, 64])

# for weekly report dump
max_dump = 5
cnt_dump = 0
# for weekly report dump


# 0, 1, 2 respond to dc, planar, angle
psnrs = [[], [], []]
ssims = [[], [], []]
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
        # print(str(i) + '_frame var bigger than 200')
        # cv2.imwrite(gt_path + str(i) + '.png', img)
        label = label.reshape([32, 32], order='F') * 255.0
        dcnet.blobs['data'].data[...] = data
        planarnet.blobs['data'].data[...] = data
        anglenet.blobs['data'].data[...] = data

        # test for dcnet
        vec = dcnet.forward()
        vec = vec['efc4'].reshape([32, 32], order='F') * 255.0
        tmp_stat = test_quality(label, vec)
        psnrs[0].append(tmp_stat['psnr'])
        ssims[0].append(tmp_stat['ssim'])
        
        # for weekly report dump
        if cnt_dump < max_dump:
                print('begin_dump')
                img[32:, 32:] = vec
                dump_path = valid_path + str(cnt_dump) + '_dc' + '.png'
                print(dump_path)
                cv2.imwrite(dump_path, img)
        # for weekly report dump

        # test for planarnet
        vec = planarnet.forward()
        vec = vec['efc4'].reshape([32, 32], order='F') * 255.0
        tmp_stat = test_quality(label, vec)
        psnrs[1].append(tmp_stat['psnr'])
        ssims[1].append(tmp_stat['ssim'])

        # for weekly report dump
        if cnt_dump < max_dump:
                img[32:, 32:] = vec
                dump_path = valid_path + str(cnt_dump) + '_planar' + '.png'
                cv2.imwrite(dump_path, img)
        # for weekly report dump

        # test for anglenet
        vec = anglenet.forward()
        vec = vec['efc4'].reshape([32, 32], order='F') * 255.0
        tmp_stat = test_quality(label, vec)
        psnrs[2].append(tmp_stat['psnr'])
        ssims[2].append(tmp_stat['ssim'])

        # for weekly report dump
        if cnt_dump < max_dump:
                img[32:, 32:] = vec
                dump_path = valid_path + str(cnt_dump) + '_angle' + '.png'
                cv2.imwrite(dump_path, img)
                cnt_dump = cnt_dump + 1
        # for weekly report dump

        


print(np.mean(psnrs[0]), np.mean(psnrs[1]), np.mean(psnrs[2]))
print(np.mean(ssims[0]), np.mean(ssims[1]), np.mean(ssims[2]))

# now test without filter
psnrs = [[], [], []]
ssims = [[], [], []]
for i in range(sample_number):
    print(i)
    data = datas[i:i+1, :, :, :]
    label = labels[i:i+1, :, :, :]
    img[:32, :] = data[:, :2048, :, :].reshape([32, 64], order='F') * 255.0
    img[32:64, :32] = data[:, 2048:, :, :].reshape([32, 32], order='F') * 255.0
    img[32:64, 32:64] = label.reshape([32, 32], order='F') * 255.0
    # write img to the gt directory
    # for normal data, we consider nothing
    # print(i)
    # if np.var(img) > 200:
        # cv2.imwrite(gt_path + str(i) + '.png', img)
    label = label.reshape([32, 32], order='F') * 255.0
    dcnet.blobs['data'].data[...] = data
    planarnet.blobs['data'].data[...] = data
    anglenet.blobs['data'].data[...] = data
    # test for dcnet
    vec = dcnet.forward()
    vec = vec['efc4'].reshape([32, 32], order='F') * 255.0
    tmp_stat = test_quality(label, vec)
    psnrs[0].append(tmp_stat['psnr'])
    ssims[0].append(tmp_stat['ssim'])
    # test for planarnet
    vec = planarnet.forward()
    vec = vec['efc4'].reshape([32, 32], order='F') * 255.0
    tmp_stat = test_quality(label, vec)
    psnrs[1].append(tmp_stat['psnr'])
    ssims[1].append(tmp_stat['ssim'])
    # test for anglenet
    vec = anglenet.forward()
    vec = vec['efc4'].reshape([32, 32], order='F') * 255.0
    tmp_stat = test_quality(label, vec)
    psnrs[2].append(tmp_stat['psnr'])
    ssims[2].append(tmp_stat['ssim'])
    # img[32:, 32:] = vec[...]
    # write pred_img to the pred directory
    # cv2.imwrite(pred_path + str(i) + '.png', img)
print(np.mean(psnrs[0]), np.mean(psnrs[1]), np.mean(psnrs[2]))
print(np.mean(ssims[0]), np.mean(ssims[1]), np.mean(ssims[2]))
    
