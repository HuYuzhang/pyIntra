# Data augmented
from __future__ import print_function
import tensorflow as tf
import cv2
import h5py
import numpy as np
import sys
import os
import subprocess as sp
from tensorflow.python.framework.graph_util import convert_variables_to_constants

batch_size = 1 
epochs = 1000


def tf_build_model(module_name, weights_name, params, input_tensor, target_tensor):
    with tf.variable_scope('main_full', reuse=tf.AUTO_REUSE):
        model_module = __import__(module_name)
        satd_loss, mse_loss, recon = model_module.build_model(input_tensor, target_tensor, params=params, test=True)
        # model_module.build_model(
        #     input_tensor, output_tensor, params, mode=mode)
        return satd_loss, mse_loss, recon


def drive():
    model_module_name = sys.argv[2]
    block_size = int(sys.argv[3])
    scale = int(sys.argv[4])
    weights_name = sys.argv[5]
    

    inputs = tf.placeholder(tf.float32, [batch_size, block_size*scale, block_size*scale, 1])
    targets = tf.placeholder(tf.float32, [batch_size, block_size, block_size, 1])

    # build model
    satd_loss, mse_loss, recon = tf_build_model(model_module_name,
                                       weights_name,
                                       {'learning_rate': 0.0001,
                                           'batch_size': batch_size,
                                           'block_size': block_size,
                                           'scale':scale},
                                       inputs,
                                       targets)
    # tensorboard_dir = 'tensorboard'
    # if not os.path.exists(tensorboard_dir):
    #     os.makedirs(tensorboard_dir)

    # writer = tf.summary.FileWriter(tensorboard_dir)
    saver = tf.train.Saver()
    # checkpoint_dir = './ckpt/'
    with tf.Session() as sess:
        saver.restore(sess, weights_name)
        # import IPython
        # IPython.embed()
        graph = convert_variables_to_constants(sess, sess.graph_def, ['main_full/4_dim_out_pixel'])
        #graph = convert_variables_to_constants(sess, sess.graph_def, ['main_full/Reshape_1'])
        tf.train.write_graph(graph,'../../pb','graph_m%d_s%d.pb' % (scale, block_size),as_text=False)



if __name__ == '__main__':
    tasks = {'train': drive}
    task = sys.argv[1]
    tasks[task]()
