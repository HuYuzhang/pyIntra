import cv2
import numpy as np

# Assume that our video is I420
def read_frame(filename, idx, _height, _width, mode=0):
    pixel_num = _height * _width
    byte_num = int(pixel_num * 3 / 2)
    # print(byte_num)
    with open(filename, 'rb') as f:
        f.seek(idx * byte_num, 0)
        # only luma mode
        if mode == 0:
                data = np.fromfile(f, dtype=np.uint8, count=pixel_num)
                return data.reshape([_height, _width])

        else:
                # Three color mode
                dataY = np.fromfile(f, dtype=np.uint8, count=pixel_num)
                dataU = np.fromfile(f, dtype=np.uint8, count=int(pixel_num / 4))
                dataV = np.fromfile(f, dtype=np.uint8, count=int(pixel_num / 4))
                img = np.zeros([3, _height, _width])
                img[0,:,:] = dataY.reshape([_height, _width])
                img[1,0::2,0::2] = dataU.reshape([int(_height / 2), int(_width / 2)])
                img[1,0::2,1::2] = img[1,0::2,0::2]
                img[1,1::2,0::2] = img[1,0::2,0::2]
                img[1,1::2,1::2] = img[1,0::2,0::2]
                img[2,0::2,0::2] = dataV.reshape([int(_height / 2), int(_width / 2)])
                img[2,0::2,1::2] = img[2,0::2,0::2]
                img[2,1::2,0::2] = img[2,0::2,0::2]
                img[2,1::2,1::2] = img[2,0::2,0::2]
                img = img.astype(np.uint8)
                img = img.transpose(1,2,0)
                print(img.dtype)
                print('---', img.shape)
                img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
                return img