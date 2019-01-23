import h5py
import sys
import numpy as np
import tensorflow as tf
from mylib import h5Handler
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

if __name__ == '__main__':
    h5_name = sys.argv[1]
    model_module_name = sys.argv[2]
    block_size = int(sys.argv[3])
    scale = int(sys.argv[4])

    weights_name = sys.argv[5]
    thres = float(sys.argv[6])


    f = h5py.File(h5_name, 'r')
    batch_size = 1

    x = np.array(f['data'], dtype=np.float32)
    y = np.array(f['label'], dtype=np.float32)
    length = x.shape[0]
    print("The total length is : %d"%(length))
    inputs = tf.placeholder(
        tf.float32, [batch_size, block_size * scale, block_size * scale])
    targets = tf.placeholder(tf.float32, [batch_size, block_size, block_size])

    ##################### Cache Part ##########################
    filter_h5_name = "filter_" + h5_name
    cache_size = 2000
    input_cache = np.zeros([cache_size, block_size * scale, block_size * scale])
    label_cache = np.zeros([cache_size, block_size, block_size])
    cache_cnt = 0
    fid = 0
    h5er = h5Handler(filter_h5_name)
    ##################### Cache Part ##########################

    satd_loss, mse_loss, pred = tf_build_model(model_module_name,
                                                                        inputs,
                                                                        targets,
                                                                        test=True,
                                                                        params={'learning_rate': 0,
                                                                                'batch_size': batch_size,
                                                                                'scale': scale,
                                                                                'block_size': block_size
                                                                                },
                                                                        _weights_name=weights_name
                                                                        )
    print('finish build network')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        if weights_name is None:
            print('error!, no weights_name')
            exit(0)
        else:
            saver.restore(sess, weights_name)
            print('Successfully restore weights from file: ', weights_name)

        for i in range(100):
        #for i in range(length):
            val_satd, val_mse, recon = sess.run([satd_loss, mse_loss, pred], feed_dict={
                    inputs: x[i:i+1,:,:], targets: y[i:i+1,:,:]})
            print("Print loss in sampe i: %d, loss is %d", val_mse)
            if val_mse > thres:
                input_cache[cache_cnt,:,:] = x[i,:,:]
                label_cache[cache_cnt,:,:] = y[i,:,:]
                cache_cnt = cache_cnt + 1

                if cache_cnt >= cache_size:
                    if fid == 0: # create mode
                        h5er.write(input_cache, label_cache, create=True)
                        fid = 1
                    else:
                        h5er.write(input_cache, label_cache, create=False)
                    cache_cnt = 0
