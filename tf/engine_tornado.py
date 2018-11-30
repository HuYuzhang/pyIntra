# Data augmented
from __future__ import print_function
import tensorflow as tf
import cv2
import h5py
import numpy as np
import sys
import os
import subprocess as sp

batch_size = 1024 
epochs = 1000


def tf_build_model(module_name, weights_name, params, input_tensor, output_tensor):
    with tf.variable_scope('main_full', reuse=tf.AUTO_REUSE):
        model_module = __import__(module_name)
        train_op, satd_op, mse_op = model_module.build_model(
            input_tensor, output_tensor, params)
        return train_op, satd_op, mse_op


def drive():
    print(sys.argv)
    global batch_size
    block_size = 8
    model_module_name = sys.argv[2]
    weights_name = None
    train_mode = sys.argv[3]
    init_lr = float(sys.argv[4])
    batch_size = int(sys.argv[5])
    if len(sys.argv) == 7:
        weights_name = sys.argv[6]
    print(weights_name)

    h5_path = '../../train/' + train_mode + '.h5'
    # load data

    hf = None
    # if mode == 3:
    #     hf = h5py.File('./General_dataset_%d_full.h5' % (num_scale))
    #     print('./General_dataset_%d_full.h5' % (num_scale))
    # else:
    #     hf = h5py.File('./General_dataset_%d_partial.h5' % (num_scale))
    #     print('./General_dataset_%d_partial.h5' % (num_scale))
    
    hf = h5py.File(h5_path)

    with tf.Session() as sess:
        pass
        
    print("Loading data")
    x = np.array(hf['data'], dtype=np.float32)
    y = np.array(hf['label'], dtype=np.float32)

    length = x.shape[0]
    array_list = list(range(0, length))
    np.random.shuffle(array_list)
    bar = int(length*0.95)

    train_data = x[array_list[:bar], :, :, :]
    val_data = x[array_list[bar:], :, :, :]
    train_label = y[array_list[:bar], :, :, :]
    val_label = y[array_list[bar:], :, :, :]
    # train_data = train_data.transpose([0,2,3,1])
    # val_data = val_data.transpose([0,2,3,1])

    # train_label = train_label.transpose([0,2,3,1])
    # val_label = val_label.transpose([0,2,3,1])
    print(bar)

    def train_generator():
        while True:
            for i in range(0, bar, batch_size)[:-1]:
                yield train_data[i:i+batch_size, :, :, :], train_label[i:i+batch_size, :, :, :]
            # np.random.shuffle(train_data)

    def val_generator():
        for i in range(0, length-bar, batch_size)[:-1]:
            yield val_data[i:i+batch_size, :, :, :], val_label[i:i+batch_size, :, :, :]

    # inputs = tf.placeholder(tf.float32, [batch_size, 3072, 1, 1])
    # targets = tf.placeholder(tf.float32, [batch_size, 1024, 1, 1])
    inputs = tf.placeholder(tf.float32, [None, 3072, 1, 1])
    targets = tf.placeholder(tf.float32, [None, 1024, 1, 1])

    # build model
    train_op, satd_loss, mse_loss = tf_build_model(model_module_name,
                                       weights_name,
                                       {'learning_rate': init_lr,
                                           'batch_size': batch_size
                                        },
                                       inputs,
                                       targets)
    
    tensorboard_train_dir = '../../tensorboard/' + train_mode + '/train'
    tensorboard_valid_dir = '../../tensorboard/' + train_mode + '/valid'
    if not os.path.exists(tensorboard_train_dir):
        os.makedirs(tensorboard_train_dir)
    if not os.path.exists(tensorboard_valid_dir):
        os.makedirs(tensorboard_valid_dir)

    saver = tf.train.Saver(max_to_keep=30)
    checkpoint_dir = '../../model/' + train_mode + '/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with tf.Session() as sess:
        if weights_name is not None:
            saver.restore(sess, weights_name)
        else:
            sess.run(tf.global_variables_initializer())
        total_var = 0
        for var in tf.trainable_variables():
            shape = var.get_shape()
            par_num = 1
            for dim in shape:
                par_num *= dim.value
            total_var += par_num
        print("Number of total variables: %d" %(total_var))
        options = tf.RunOptions()  # trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        data_gen = train_generator()
        interval = 500
        metrics = np.zeros((interval,3))

        # --------------- part for tensorboard----------------
        train_writer = tf.summary.FileWriter(tensorboard_train_dir, sess.graph)
        valid_writer = tf.summary.FileWriter(tensorboard_valid_dir, sess.graph)
        tf.summary.scalar('SATD loss', satd_loss)
        tf.summary.scalar('MSE loss', mse_loss)
        merged = tf.summary.merge_all()
        # --------------- part for tensorboard----------------

        for i in range(60000):
            if i % interval == 0:
                val_satd_s = []
                val_mse_s = []
                val_gen = val_generator()
                for v_data, v_label in val_gen:
                    val_satd, val_mse = sess.run([satd_loss, mse_loss], feed_dict={
                                                 inputs: v_data, targets: v_label})
                    val_satd_s.append(float(val_satd))
                    val_mse_s.append(float(val_mse))

                val_satd, val_mse, rs = sess.run([satd_loss, mse_loss, merged], feed_dict={
                                                inputs: val_data, targets: val_label})
                test_writer.add_summary(rs, i)
                # ------------ Now try to write data to the test_writer
                
                # ------------ end writing --------------------

                # # ----------------- for test-------------------
                # for v_data, v_label in val_gen:
                #     val_mse = sess.run(mse_loss, feed_dict={
                #                                  inputs: v_data, targets: v_label})
                #     # val_satd_s.append(float(val_satd))
                #     val_mse_s.append(float(val_mse))
                # #----------------------------------------------

                # print(val_satd_s)
                print("Model name: %s, step %8d, Train SATD %.4f, Train MSE %.4f, Val SATD %.4f, Val MSE %.6f" % (
                    model_module_name, i, np.mean(metrics[:,0]), np.mean(metrics[:,1]), np.mean(val_satd_s), np.mean(val_mse_s)))
                
            # ------------------- Here is the training part ---------------
            iter_data, iter_label = next(data_gen)
            # print(iter_data.shape)
            feed_dict = {inputs: iter_data, targets: iter_label}
            _, satd, mse, rs = sess.run([train_op, satd_loss, mse_loss, merged],
                                    feed_dict=feed_dict,
                                    options=options,
                                    run_metadata=run_metadata)

            train_writer.add_summary(rs, i)

            metrics[i%interval,0] = satd
            metrics[i%interval,1] = mse
            
            if i % 10000 == 0:
                save_path = saver.save(sess, os.path.join(
                    checkpoint_dir, "%s_%06d.ckpt" % (model_module_name,i)))


if __name__ == '__main__':
    tasks = {'train': drive}
    task = sys.argv[1]
    tasks[task]()
