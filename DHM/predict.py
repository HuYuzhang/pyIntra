import tensorflow as tf
import  numpy as np
import sys
import h5py
import cv2
from mylib import read_frame

cu_size = int(sys.argv[1])
scale = int(sys.argv[2])
f_cnt = int(sys.argv[3])
dec_name = sys.argv[4]
gt_name = sys.argv[5]
height = int(sys.argv[6])
width = int(sys.argv[7])

cu_pixel = cu_size * cu_size
input_cache = np.zeros([1, cu_size * scale, cu_size * scale])
label_cache = np.zeros([1, cu_size, cu_size])
input_size = cu_pixel * (2 * scale - 1)
mask_mean = True

result = np.zeros([height, width])
rnn_flag = False
hfc = False
print(input_cache.shape)
with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    if rnn_flag:
        with open("graph_m2_s8_rev7.pb", "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")
    else:
        if hfc:
            with open("graph_m2_s8_FC.pb", "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")    
        else:
            with open("graph_m2_s8.pb", "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        s = str(sess.graph_def)
        with open('graph.log', 'w') as f:
            f.write(s)
            f.close()
        input_x = sess.graph.get_tensor_by_name("Placeholder:0")
        if rnn_flag:
            pred = sess.graph.get_tensor_by_name("main_full/conv11/BiasAdd:0")
        else:
            if hfc:
                pred = sess.graph.get_tensor_by_name("main_full/Reshape_1:0")
            else:
                pred = sess.graph.get_tensor_by_name("main_full/4_dim_out_pixel:0")

        
        
        Y = read_frame(dec_name, 0, height, width)
        YY = read_frame(gt_name, 0, height, width)
        for lx in range(0,width,cu_size):
            for ly in range(0,height,cu_size):
                rx = lx + cu_size * scale
                ry = ly + cu_size * scale
                if rx >= width or ry >= height:
                    continue
                input_cache[0, :, :] = Y[ly:ly+cu_size * scale, lx:lx+cu_size*scale] / 255.0

                if mask_mean:
                    input_cache[0, cu_size:, cu_size:] = 0
                    mean = np.sum(input_cache[0, :, :]) / float(input_size)
                    input_cache[0, cu_size:, cu_size:] = mean
                else:
                    input_cache[0, cu_size:, cu_size:] = 0

                recon = sess.run(pred, feed_dict={input_x: input_cache.reshape([1, cu_size * scale, cu_size * scale, 1])})
                recon = np.clip(recon, 0, 1).reshape([cu_size, cu_size]) * 255.0
                result[ly+cu_size:ly+cu_size*2, lx+cu_size:lx+cu_size*2] = recon
        result = result.astype(np.uint8)
if rnn_flag:
    cv2.imwrite('pred_rnn.png', result)
else:
    if hfc:
        cv2.imwrite('pred_hfc.png', result)
    else:
        cv2.imwrite('pred_myfc.png', result)
            

