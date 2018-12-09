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


def tf_build_model(module_name, weights_name, params, input_tensor, target_tensor, mode):
    with tf.variable_scope('main_full', reuse=tf.AUTO_REUSE):
        model_module = __import__(module_name)
        satd_loss, mse_loss, recon = build_model(input_tensor, target_tensor, params=params, freq=True, test=True)
        # model_module.build_model(
        #     input_tensor, output_tensor, params, mode=mode)
        return satd_loss, mse_loss, recon


def drive():
    block_size = 8
    model_module_name = sys.argv[2]
    model_type = sys.argv[3]
    weights_name = None
    mode = int(sys.argv[4])
    num_scale = int(sys.argv[5])
    if len(sys.argv) == 7:
        weights_name = sys.argv[6]
    print(weights_name)
    # load data

    # hf = None
    # if mode == 3:
    #     hf = h5py.File('./Diverse_dataset_8_full.h5')
    # else:
    #     hf = h5py.File('./Diverse_dataset_8_partial.h5')
    

    inputs = tf.placeholder(tf.float32, [batch_size, num_scale*mode, num_scale*mode, 1])
    targets = tf.placeholder(tf.float32, [batch_size, num_scale, num_scale, 1])

    # build model
    satd_loss, mse_loss, recon = tf_build_model(model_module_name,
                                       weights_name,
                                       {'learning_rate': 0.0001,
                                           'batch_size': batch_size,
                                           'num_scale':num_scale},
                                       inputs,
                                       targets,mode)
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
        graph = convert_variables_to_constants(sess, sess.graph_def, ['main_full/mul_idct1'])
        #graph = convert_variables_to_constants(sess, sess.graph_def, ['main_full/Reshape_1'])
        tf.train.write_graph(graph,'.','graph_m%d_s%d_%s.pb' % (mode, num_scale, model_id),as_text=False)



if __name__ == '__main__':
    tasks = {'train': drive}
    task = sys.argv[1]
    tasks[task]()
