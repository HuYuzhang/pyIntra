import numpy as np
import cv2 as cv

dump_path = 'dump_dir.txt'
dec_path = '../raw_data/dec.yuv'
gt_path = '../raw_data/video.yuv'
target_path = '../img/'
height = 1024
width = 1792
block_size = 32

# Assume that our video is I420
def read_frame(filename, idx, _height, _width):
    pixel_num = _height * _width
    byte_num = int(pixel_num * 3 / 2)
    # print(byte_num)
    with open(filename, 'rb') as f:
        f.seek(idx * byte_num, 0)
        data = np.fromfile(f, dtype=np.uint8, count=pixel_num)
        return data.reshape([_height, _width])


input = np.zeros([1, 3072, 1, 1])
label = np.zeros([1, 1024, 1, 1])
test_img = np.zeros([64,64])
dcvars = []
planarvars = []
anglevars = []
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
        if y == 0 and x == 0:
            gt_img = read_frame(gt_path, f_id, height, width)
            dec_img = read_frame(dec_path, f_id, height, width)
            print(f_id)
        if x == 0 or y == 0:
            continue
        # print([x, y])
        input[:,:2048,:,:] = dec_img[x-block_size:x, y-block_size:y+block_size].reshape([1,2048,1,1])
        input[:,2048:,:,:] = dec_img[x:x+block_size,y-block_size:y].reshape([1,1024,1,1])
        label[...] = gt_img[x:x+block_size,y:y+block_size].reshape([1,1024,1,1])
        test_img[:32,:64] = input[:,:2048,:,:].reshape([32,64])
        test_img[32:,:32] = input[:,2048:,:,:].reshape([32,32])
        test_img[32:,32:] = label.reshape([32,32])
        if mode == 0:
            planarvars.append(np.var(test_img))
        elif mode == 1:
            dcvars.append(np.var(test_img))
        else:
            anglevars.append(np.var(test_img))
        # cv.imwrite(target_path + str(y) + '_' + str(x) + '.png', test_img)


print(np.mean(planarvars))
print(np.mean(dcvars))
print(np.mean(anglevars))


