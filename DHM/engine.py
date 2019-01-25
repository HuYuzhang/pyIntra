# Data augmented
from __future__ import print_function
import tensorflow as tf
# import cv2
import h5py
import numpy as np
import sys
import os
import subprocess as sp
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from mylib import test_quality

batch_size = 1024
epochs = 1000


def tf_build_model(module_name, input_tensor, output_tensor, test=False, params=None, _weights_name=None):
    with tf.variable_scope('main_full', reuse=tf.AUTO_REUSE):
        model_module = __import__(module_name)
        if test:
            satd_loss, mse_loss, pred = model_module.build_model(
                input_tensor, output_tensor, params=params, test=test)
            return satd_loss, mse_loss, pred
        else:
            train_op, satd_loss, mse_loss, pred = model_module.build_model(
                input_tensor, output_tensor, params=params, test=test)
            return train_op, satd_loss, mse_loss, pred


def drive():
    if len(sys.argv) < 7:
        # This is --help mode
        print(
            "Usage: model_module_name scale block_size init_lr batch_size [weights_name]")
        exit(0)
    print(sys.argv)
    model_module_name = sys.argv[2]
    scale = int(sys.argv[3])
    block_size = int(sys.argv[4])
    init_lr = float(sys.argv[5])
    batch_size = int(sys.argv[6])
    weights_name = None
    #if len(sys.argv) == 8:
    #    weights_name = sys.argv[7]
    print(weights_name)
    prefix = 's' + str(block_size) + '_m' + str(scale)
    #h5_path = '../../train/data/' + prefix + '.h5'
    # load data
    h5_path = sys.argv[7]
    hf = None

    hf = h5py.File(h5_path)

    print("Loading data")
    x = np.array(hf['data'], dtype=np.float32)
    y = np.array(hf['label'], dtype=np.float32)

    length = x.shape[0]
    array_list = list(range(0, length))
    np.random.shuffle(array_list)
    bar = int(length*0.95)
    print('-------print the length of bar: %d, and length %d' % (bar, length))
    train_data = x[array_list[:bar], :, :]
    val_data = x[array_list[bar:], :, :]
    train_label = y[array_list[:bar], :, :]
    val_label = y[array_list[bar:], :, :]

    def train_generator():
        while True:
            for i in range(0, bar, batch_size)[:-1]:
                yield train_data[i:i+batch_size, :, :], train_label[i:i+batch_size, :, :]
            # np.random.shuffle(train_data)

    def val_generator():
        for i in range(0, length-bar, batch_size)[:-1]:
            yield val_data[i:i+batch_size, :, :], val_label[i:i+batch_size, :, :]

    inputs = tf.placeholder(
        tf.float32, [batch_size, block_size * scale, block_size * scale])
    targets = tf.placeholder(tf.float32, [batch_size, block_size, block_size])

    # build model
    train_op, satd_loss, mse_loss, pred = tf_build_model(model_module_name,
                                                                              inputs,
                                                                              targets,
                                                                              test=False,
                                                                              params={'learning_rate': init_lr,
                                                                                      'batch_size': batch_size,
                                                                                      'scale': scale,
                                                                                      'block_size': block_size
                                                                                      },
                                                                              _weights_name=weights_name
                                                                              )

    tensorboard_train_dir = '../../tensorboard/' + \
        prefix + '/train'
    tensorboard_valid_dir = '../../tensorboard/' + \
        prefix + '/valid'
    checkpoint_dir = '../../model/' + prefix + '/'
    if not os.path.exists(tensorboard_train_dir):
        os.makedirs(tensorboard_train_dir)
    if not os.path.exists(tensorboard_valid_dir):
        os.makedirs(tensorboard_valid_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver = tf.train.Saver(max_to_keep=30)

    with tf.Session() as sess:
        if weights_name is not None:
            saver.restore(sess, weights_name)
            print('-----------Sucesfully restoring weights from: ', weights_name)
        else:
            sess.run(tf.global_variables_initializer())
            print('-----------No weights defined, run initializer')
        total_var = 0
        for var in tf.trainable_variables():
            shape = var.get_shape()
            par_num = 1
            for dim in shape:
                par_num *= dim.value
            total_var += par_num
        print("----------------Number of total variables: %d" % (total_var))
        options = tf.RunOptions()  # trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        data_gen = train_generator()
        interval = 500
        metrics = np.zeros((interval, 3))

        # --------------- part for tensorboard----------------
        train_writer = tf.summary.FileWriter(tensorboard_train_dir, sess.graph)
        valid_writer = tf.summary.FileWriter(tensorboard_valid_dir, sess.graph)
        train_satd_summary = tf.summary.scalar(
            'SATD loss', satd_loss)
        train_mse_summary = tf.summary.scalar(
            'MSE loss', mse_loss)
        merged = tf.summary.merge(
            [train_satd_summary, train_mse_summary])

        # sub1--------------------------------here for valid mean
        valid_size = int(len(range(0, length - bar, batch_size)[:-1]))
        print(valid_size)
        valid_mse_input = tf.placeholder(tf.float32, [valid_size])
        valid_satd_input = tf.placeholder(tf.float32, [valid_size])

        valid_mse_mean = tf.reduce_mean(valid_mse_input)
        valid_satd_mean = tf.reduce_mean(valid_satd_input)

        valid_mse_summary = tf.summary.scalar(
            'MSE loss', valid_mse_mean)
        valid_satd_summary = tf.summary.scalar(
            'SATD loss', valid_satd_mean)

        valid_merged = tf.summary.merge(
            [valid_satd_summary, valid_mse_summary])
        # sub1--------------------------------for valid mean

        # --------------- part for tensorboard----------------

        for i in range(200000):
            if i % interval == 0:
                val_satd_s = []
                val_mse_s = []
                val_gen = val_generator()
                psnr_s = []
                ssim_s = []
                for v_data, v_label in val_gen:
                    val_satd, val_mse, recon = sess.run([satd_loss, mse_loss, pred], feed_dict={
                        inputs: v_data, targets: v_label})
                    val_mse_s.append(float(val_mse))
                    val_satd_s.append(float(val_satd))
                    tmp_psnr, tmp_ssim = test_quality(np.clip(v_label,0,1).reshape(
                        [-1, block_size, block_size])[0] * 255.0, np.clip(recon,0,1).reshape([-1, block_size, block_size])[0] * 255.0)
                    psnr_s.append(tmp_psnr)
                    ssim_s.append(tmp_ssim)
                    # print('#########tmp: ', tmp_psnr, tmp_ssim)

                # Here is about the tensorboard
                rs = sess.run(valid_merged, feed_dict={
                    valid_satd_input: val_satd_s, valid_mse_input: val_mse_s
                })
                valid_writer.add_summary(rs, i)
                # Here is about the tensorboard

                # now test for psnr
                print('------------->now show the info of PSNR and SSIM')
                print('PSNR is: %f, SSIM is: %f' %
                      (np.mean(psnr_s), np.mean(ssim_s)))

                # print(val_satd_s)
                print("Model name: %s, step %8d, Train SATD %.6f, Train pixel MSE %.6f, Train freq MSE %.6f, Val SATD %.6f, Val freq_MSE %.6f, Val pixel_MSE %.6f" % (
                    model_module_name, i, np.mean(metrics[:, 0]), np.mean(metrics[:, 1]), np.mean(metrics[:, 2]), np.mean(val_satd_s), np.mean(val_mse_s), np.mean(val_mse_s)))

            # ------------------- Here is the training part ---------------
            iter_data, iter_label = next(data_gen)
            # print(iter_data.shape)
            feed_dict = {inputs: iter_data, targets: iter_label}
            _, satd, mse, rs = sess.run([train_op, satd_loss, mse_loss, merged],
                                                        feed_dict=feed_dict,
                                                        options=options,
                                                        run_metadata=run_metadata)
            if i % interval == 0:
                train_writer.add_summary(rs, i)

            metrics[i % interval, 0] = satd
            metrics[i % interval, 1] = mse
            metrics[i % interval, 2] = mse

            if i % 10000 == 0:
                save_path = saver.save(sess, os.path.join(
                    checkpoint_dir, "%s_%06d.ckpt" % (model_module_name, i)))


def run_test():
    if len(sys.argv) == 2:
        # This is --help mode
        print(
            "Usage: model_module_name train_mode scale block_size init_lr batch_size [weights_name]")
    print(sys.argv)
    model_module_name = sys.argv[2]
    scale = int(sys.argv[3])
    block_size = int(sys.argv[4])
    init_lr = float(sys.argv[5])
    batch_size = int(sys.argv[6])
    #weights_name = None
    #if len(sys.argv) == 8:
    weights_name = sys.argv[7]
    print(weights_name)

    inputs = tf.placeholder(
        tf.float32, [batch_size, block_size * scale, block_size * scale])
    targets = tf.placeholder(tf.float32, [batch_size, block_size, block_size])

    prefix = 's' + str(block_size) + '_m' + str(scale)
    #h5_path = '../../train/data/' + prefix + '.h5'
    h5_path = sys.argv[8]
    hf = None

    hf = h5py.File(h5_path)

    print("Loading data")
    x = np.array(hf['data'], dtype=np.float32)
    y = np.array(hf['label'], dtype=np.float32)

    length = x.shape[0]
    print("Finishing loading data and begin to build network from: ", model_module_name)
    satd_loss, mse_loss, pred = tf_build_model(model_module_name,
                                                                    inputs,
                                                                    targets,
                                                                    test=True,
                                                                    params={'learning_rate': init_lr,
                                                                            'batch_size': batch_size,
                                                                            'scale': scale,
                                                                            'block_size': block_size
                                                                            },
                                                                    _weights_name=weights_name
                                                                    )
    print('finish build network')

    def val_generator():
        for i in range(0, length, batch_size)[:-1]:
            yield x[i:i+batch_size, :, :], y[i:i+batch_size, :, :]

    saver = tf.train.Saver()

    with tf.Session() as sess:
        if weights_name is None:
            print('error!, no weights_name')
            exit(0)
        else:
            saver.restore(sess, weights_name)
            print('Successfully restore weights from file: ', weights_name)
        # Fore debug
        #import IPython
        # IPython.embed()
        # For debug
        val_satd_s = []
        val_mse_s = []
        val_gen = val_generator()
        psnr_s = []
        ssim_s = []
        val_cnt = 0
        for v_data, v_label in val_gen:
            val_satd, val_mse, recon = sess.run([satd_loss, mse_loss, pred], feed_dict={
                inputs: v_data, targets: v_label})
            val_mse_s.append(float(val_mse))
            val_satd_s.append(float(val_satd))
            val_psnr, val_ssim = test_quality(v_label.reshape(
                [-1, block_size, block_size])[0] * 255.0, recon.reshape([-1, block_size, block_size])[0] * 255.0)
            psnr_s.append(val_psnr)
            ssim_s.append(val_ssim)
            val_cnt = val_cnt + batch_size
            print('-----------> tmp data, now %d sample tested, %d in total, psnr: %f, ssim: %f, pixel mse loss: %f, freq mse loss: %f, satd_loss: %f<------------' %
                  (val_cnt, length, val_psnr, val_ssim, val_mse, val_mse, val_satd))
        print('Finish testing, now psnr is: %f, and ssim is: %f, pixel mse loss: %f, freq mse loss: %f, satd_loss: %f' %
              (np.mean(psnr_s), np.mean(ssim_s), np.mean(val_mse_s), np.mean(val_mse_s), np.mean(val_satd_s)))


        # for v_data, v_label in val_gen:
        #     val_satd, val_mse, recon = sess.run([satd_loss, mse_loss, pred], feed_dict={
        #         inputs: v_data, targets: v_label})
        #     val_cnt = val_cnt + batch_size
        #     recon = recon.reshape([-1, 32, 32]) * 255.0
        #     gt = v_label.reshape([-1, 32, 32]) * 255.0
        #     val_psnr, val_ssim = test_quality(gt, recon)

        #     print('-----------> tmp data, now %d sample tested, %d in total, psnr: %f, ssim: %f, mse loss: %f, satd_loss: %f<------------' %
        #           (val_cnt, length, val_psnr, val_ssim, np.mean(val_mse), np.mean(val_satd)))
        #     psnr_s.append(val_psnr)
        #     ssim_s.append(val_ssim)
        # print('Finish testing, now psnr is: %f, and ssim is: %f' %
        #       (np.mean(psnr_s), np.mean(ssim_s)))


def dump_img(filename, targetpath):
    pass
    # model_module_name = sys.argv[2]
    # weights_name = sys.argv[3]
    # filename = sys.argv[4]
    # print(weights_name, model_module_name, filename)

    # img = skimage.imread(filename) / 255.0
    # input, gt = img2input(filename)

    # inputs = tf.placeholder(tf.float32, [1, 3072, 1, 1])
    # targets = tf.placeholder(tf.float32, [1, 1024, 1, 1])
    # satd_loss, mse_loss, pred = tf_build_model(model_module_name,
    #                                    inputs,
    #                                    targets,
    #                                    test=True,
    #                                    freq=False,
    #                                    _weights_name=weights_name
    #                                    )

    # saver = tf.train.Saver()

    # with tf.Session() as sess:
    #     if weights_name is None:
    #         print('error!, no weights_name')
    #         exit(0)
    #     else:
    #         saver.restore(sess, weights_name)
    #         print('Successfully restore weights from file: ', weights_name)

    #     recon = sess.run(pred, feed_dict={inputs: input.reshape(1,3072,1,1), targets: gt.reshape(1,1024,1,1)})
    #     img[32:,32:] = recon.reshape([32,32]) * 255.0
    #     skimage.imwrite(targetpath, img)


if __name__ == '__main__':
    tasks = {'train': drive, 'test': run_test, 'dump': dump_img}
    task = sys.argv[1]
    print('-------------begin task', task)
    tasks[task]()
