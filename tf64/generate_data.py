import numpy as np
import cv2 as cv
from mylib import h5Handler
from mylib import read_frame

dump_path = 'dump_dir.txt'
dec_path = '../../raw_data/dec.yuv'
gt_path = '../../raw_data/video.yuv'
target_path = '../img/'

planar_h5_path = '../../train64/planar.h5'
dc_h5_path = '../../train64/dc.h5'
angle_h5_path = '../../train64/angle.h5'

height = 1024
width = 1792
block_size = 32

# define 3 mode
planar_mode = 0
dc_mode = 1
angle_mode = 2



def filter_sample(img, threshold):
    # var = np.var(img)
    # if var > threshold:
    #     return True
    # else:
    #     return False
    pass

input = np.zeros([96, 96])
label = np.zeros([32, 32])
test_img = np.zeros([64,64])
dcvars = []
planarvars = []
anglevars = []

planarinput = np.zeros([2000, 96, 96])
planarlabel = np.zeros([2000, 32, 32])
dcinput =np.zeros([2000, 96, 96])
dclabel = np.zeros([2000, 32, 32])
angleinput = np.zeros([2000, 96, 96])
anglelabel = np.zeros([2000, 32, 32])

planar_handler = h5Handler(planar_h5_path)
dc_handler = h5Handler(dc_h5_path)
angle_handler = h5Handler(angle_h5_path)
# # --------------for debug------------------
# cnt = 0
# datas = np.zeros([20, 3072, 1, 1])
# labels = np.zeros([20, 1024, 1, 1])
# flag = True   
# handler = h5Handler('/home/hyz/lab/intra/train/train.h5')     
# # --------------for debug------------------

with open(dump_path) as f:
    while True:
        line = f.readline()
        if line == '':
            break
        [f_id, y, x, mode] = line.split()
        y = int(y)
        x = int(x)
        f_id = int(f_id)
        mode = int(mode)
        # --------------for debug------------------
       	#if f_id > 5:
        #   break
        # --------------for debug------------------

        if y == 0 and x == 0:
            pc = 0
            dc = 0
            ac = 0

            gt_img = read_frame(gt_path, f_id, height, width)
            dec_img = read_frame(dec_path, f_id, height, width)
            print(f_id)
        # Abort the most outsize row and column
        if x == 0 or y == 0 or x == 992 or y == 1760:
            continue
        # print([x, y])
        input  = dec_img[x-block_size:x+2*block_size,y-block_size:y+2*block_size] / 255.0
        label = gt_img[x:x+block_size,y:y+block_size] / 255.0


        # --------------for debug------------------(write data to hdf5 file)
        # datas[cnt:cnt + 1, :, :, :] = input / 255.0
        # labels[cnt:cnt + 1, :, :, :] = label / 255.0
        # cnt += 1
        # if cnt == 20:
        #     if flag:
        #         handler.write(datas, label, create=True)
        #         print('$$$$$$$$$$$ new h5 file constructed $$$$$$$$$$$$')
        #         flag = False
        #     else:
        #         handler.write(datas, labels, create=False)
        #         print('$$$$$$$$$$$ add data to existed h5 file constructed $$$$$$$$$$$$')
        #     cnt = 0
        # --------------for debug------------------(write data to hdf5 file)

        if mode == planar_mode:
            planarinput[pc, :, :] = input
            planarlabel[pc, :, :] = label
            pc = pc + 1
            planarvars.append(np.var(test_img))
        elif mode == dc_mode:
            dcinput[dc, :, :] = input
            dclabel[dc, :, :] = label
            dc = dc + 1
            dcvars.append(np.var(test_img))
        else:
            angleinput[ac, :, :] = input
            anglelabel[ac, :, :] = label
            ac = ac + 1
            anglevars.append(np.var(test_img))
        # cv.imwrite(target_path + str(y) + '_' + str(x) + '.png', test_img)

        # Then check if we have arrive the final block of this frame
        if y == 1728 and x == 960:
            # now begin to write data to the h5 file
            if f_id == 0:
                # create mode
                planar_handler.write(planarinput[:pc,:,:], planarlabel[:pc,:,:], create=True)
                dc_handler.write(dcinput[:dc,:,:], dclabel[:dc,:,:], create=True)
                angle_handler.write(angleinput[:ac,:,:], anglelabel[:ac,:,:], create=True)
            else:
                # append mode
                planar_handler.write(planarinput[:pc,:,:], planarlabel[:pc,:,:], create=False)
                dc_handler.write(dcinput[:dc,:,:], dclabel[:dc,:,:], create=False)
                angle_handler.write(angleinput[:ac,:,:], anglelabel[:ac,:,:], create=False)
            print('In this frame %d: planar: %d, dc: %d, angle: %d'%(f_id, pc, dc, ac))


print('------------ print statistic data -------------')
print('planar var: ', np.mean(planarvars), len(planarvars))
print('dc var: ', np.mean(dcvars), len(dcvars))
print('angle var: ', np.mean(anglevars), len(anglevars))


