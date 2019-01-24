import tensorflow as tf
import numpy as np
# refined srnn based on rev5 lite
# try shrink the network


def build_model(input_tensor, target_tensor, params=None, test=False):

    print(input_tensor.shape)

    # in fact, in test stage, params is always None
    if params is not None:
        batch_size = params['batch_size']
        lr = params['learning_rate']
        block_size = params['block_size']
        scale = params['scale']
    else:
        batch_size = input_tensor.shape[0]
        lr = 0.0001
        scale = 2
        block_size = 32

    # we assume that input_tensor is of size (batch_size * 96 * 96)
    # our target is to (batch_size * 32 * 32)
    input_tensor = tf.reshape(
        input_tensor, (-1, block_size * scale, block_size * scale))
    target_tensor = tf.reshape(target_tensor, (-1, block_size, block_size))
    # so first slice
    inputs = []
    if scale == 2:  # scale 2
        inputs.append(tf.reshape(tf.slice(input_tensor, [0, 0, 0], [
                      batch_size, block_size, 2 * block_size]), [-1, block_size * block_size * 2]))
        inputs.append(tf.reshape(tf.slice(input_tensor, [0, block_size, 0], [
                      batch_size, block_size, block_size]), [-1, block_size * block_size]))
    else:  # scale 3
        inputs.append(tf.reshape(tf.slice(input_tensor, [0, 0, 0], [
                      batch_size, block_size, 3 * block_size]), [-1, block_size * block_size * 3]))
        inputs.append(tf.reshape(tf.slice(input_tensor, [0, block_size, 0], [
                      batch_size, 2 * block_size, block_size]), [-1, 2 * block_size * block_size]))
    input_layer = tf.concat(inputs, 1)
    # print('---------->For debug, ', inputs[0].shape, inputs[1].shape)
    print('----------> Here is in the model building function, the input_layer size is(after slice and concat): ', input_layer.shape)
    # now the input_payer is of size [batch_size, 5120]
    # For the number of hidden state, we keep same with 3072 input

    _fc1 = tf.layers.dense(input_layer, 3072, name='fc1')

    fc1 = tf.nn.elu(_fc1, name='relu1')

    _fc2 = tf.layers.dense(fc1, 3072, name='fc2')

    fc2 = tf.nn.elu(_fc2, name='relu2')

    _fc3 = tf.layers.dense(fc2, 3072, name='fc3')

    fc3 = tf.nn.elu(_fc3, name='relu3')

    _fc4 = tf.layers.dense(fc3, block_size*block_size, name='fc4')

    fc4 = tf.nn.elu(_fc4, name='relu4')

    def SATD(y_true, y_pred):
            H_8x8 = np.array(
                [[1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
                 [1., -1.,  1., -1.,  1., -1.,  1., -1.],
                 [1.,  1., -1., -1.,  1.,  1., -1., -1.],
                 [1., -1., -1.,  1.,  1., -1., -1.,  1.],
                 [1.,  1.,  1.,  1., -1., -1., -1., -1.],
                 [1., -1.,  1., -1., -1.,  1., -1.,  1.],
                 [1.,  1., -1., -1., -1., -1.,  1.,  1.],
                 [1., -1., -1.,  1., -1.,  1.,  1., -1.]],
                dtype=np.float32
            )
            H_target = np.zeros((1, 32, 32), dtype=np.float32)
            H_target[0, 0:8, 8:16] = H_8x8
            H_target[0, 8:16, 0:8] = H_8x8
            H_target[0, 8:16, 8:16] = H_8x8

            H_target[0, 16:32, 0:16] = H_target[:, 0:16, 0:16]
            H_target[0, 0:16, 16:32] = H_target[:, 0:16, 0:16]
            H_target[0, 16:32, 16:32] = H_target[:, 0:16, 0:16]

            TH0 = tf.constant(H_target[:, :block_size, :block_size])

            TH1 = tf.tile(TH0, (input_tensor.shape[0], 1, 1))

            diff = tf.reshape(y_true - y_pred, (-1, block_size, block_size))

            return tf.reduce_mean(tf.sqrt(tf.square(tf.matmul(tf.matmul(TH1, diff), TH1)) + 0.0001))

        # prediction in pixel domain
    recon = tf.reshape(fc4, (-1, block_size, block_size), name='3_dim_raw_output_pixel')
    mse_loss = tf.reduce_mean(tf.square((target_tensor-recon)))
    satd_loss = SATD(target_tensor, recon)
    # loss = satd_loss
    loss = mse_loss
        
    recon = tf.reshape(recon, (-1, block_size, block_size, 1), name='4_dim_out_pixel')
    if test:
        return satd_loss, mse_loss, recon

    # global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.exponential_decay(params['learning_rate'], global_step=global_step, decay_steps = 10000, decay_rate=0.7)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return train_op, satd_loss, mse_loss, recon
