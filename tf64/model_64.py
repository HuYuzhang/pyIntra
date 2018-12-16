import tensorflow as tf
import numpy as np
# refined srnn based on rev5 lite
# try shrink the network

def stacked_RNN(input_tensor, num, scope, units, batch_size, num_scale):
    with tf.variable_scope(scope):
        cells = [tf.contrib.rnn.GRUCell(num_units=units,name='cell_%d'% (i) )for i in range(num)]
        states = [it.zero_state(batch_size, dtype=tf.float32) for i,it in enumerate(cells)]
        last = input_tensor
        for i in range(num):
            last, _ = tf.nn.static_rnn(cells[i], last, initial_state=states[i], sequence_length=[num_scale,]*batch_size, scope='rnn_%d' % (i))
        return last

def inter_rnn(input_tensor, num, scope, batch_size, channels=8, units=8):
    with tf.variable_scope(scope):
        # split input tensor to lines
        shaped_conv1 = tf.reshape(input_tensor, [batch_size, num, num, channels], name="conv1")

        vertical_form = tf.reshape(shaped_conv1, [batch_size, num, num*channels], name='vertical_form')
        horizontal_form1 = tf.transpose(shaped_conv1, [0, 2, 1, 3], name='horizontal_form1')
        horizontal_form =tf.reshape(horizontal_form1, [batch_size, num, num*channels], name='horizontal_form')

        vertical_split = tf.unstack(
            vertical_form,
            num=num,
            axis=1,
            name="vertical_split"
        )

        horizontal_split = tf.unstack(
            horizontal_form,
            num=num,
            axis=1,
            name="horizontal_split"
        )

        vr4 = stacked_RNN(vertical_split, 1, 'vrnn', num*channels, batch_size, num)
        hr4 = stacked_RNN(horizontal_split, 1, 'hrnn', num*channels, batch_size, num)

        stack_h_ = tf.stack(hr4, axis=1, name='from_h')

        stack_v_ = tf.stack(vr4, axis=1, name='from_v')

        _stack_h = tf.reshape(stack_h_, [batch_size, num, num, channels], name='stack_shape_h')
        stack_v = tf.reshape(stack_v_, [batch_size, num, num, channels], name='stack_shape_v')

        stack_h = tf.transpose(_stack_h, [0,2,1,3], name='h_stack_back')

        concat2 = tf.concat([stack_v, stack_h], axis=3)

        _connect = tf.layers.conv2d(
            inputs=concat2,
            filters=units,
            kernel_size=[1, 1],
            strides=[1,1],
            padding="VALID",
            name="connect"
        )

        connect = tf.keras.layers.PReLU(shared_axes=[1,2], name='relu_con')(_connect)


        return connect


def build_model(input_tensor, target_tensor, params=None, freq=False, test=False):

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
    input_tensor = tf.reshape(input_tensor,(-1, block_size * scale, block_size * scale))
    target_tensor = tf.reshape(target_tensor, (-1,block_size,block_size))
    # so first slice
    inputs = []
    if scale == 2: # scale 2
        inputs.append(tf.reshape(tf.slice(input_tensor, [0,0,0],[batch_size,block_size,2 * block_size]), [-1,block_size * block_size * 2]))
        inputs.append(tf.reshape(tf.slice(input_tensor, [0,block_size,0],[batch_size,block_size,block_size]), [-1,block_size * block_size]))
    else:# scale 3
        inputs.append(tf.reshape(tf.slice(input_tensor, [0,0,0],[batch_size,block_size,3 * block_size]), [-1,block_size * block_size * 3]))
        inputs.append(tf.reshape(tf.slice(input_tensor, [0,block_size,0],[batch_size,2 * block_size,block_size]), [-1,2 * block_size * block_size]))
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

    _fc4 = tf.layers.dense(fc3, 1024, name='fc4')

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
            H_target = np.zeros((1,32,32), dtype=np.float32)
            H_target[0,0:8,8:16] = H_8x8
            H_target[0,8:16,0:8] = H_8x8
            H_target[0,8:16,8:16] = H_8x8


            H_target[0,16:32,0:16] = H_target[:, 0:16, 0:16]
            H_target[0,0:16,16:32] = H_target[:, 0:16, 0:16]
            H_target[0,16:32,16:32] = H_target[:, 0:16, 0:16]

            TH0 = tf.constant(H_target)

            TH1 = tf.tile(TH0, (input_tensor.shape[0], 1, 1))

            diff = tf.reshape(y_true - y_pred, (-1, 32, 32))

            return tf.reduce_mean(tf.sqrt(tf.square(tf.matmul(tf.matmul(TH1, diff), TH1)) + 0.0001))


    if freq:
        dct = np.zeros((1,block_size,block_size), dtype=np.float32)
        for i in range(0,block_size):
            for j in range(0,block_size):
                a = 0.0
                if i == 0:
                    a = np.sqrt(1/float(block_size))
                else:
                    a = np.sqrt(2/float(block_size))
                dct[0,i,j] = a * np.cos(np.pi * (j + 0.5) * i / float(block_size))
        idct = dct.transpose([0, 2, 1])
        tf_dct = tf.constant(dct, name='dct')
        tf_idct = tf.constant(idct, name='idct')
        # ------------------ finish initilize the dct matrix -----------------
        freq_tensor = tf.reshape(fc4, (-1, block_size, block_size), name='3_dim_raw_output_freq')
        batch_dct = tf.tile(tf_dct, [tf.shape(input_tensor)[0],1,1],name='title_dct')
        batch_idct = tf.tile(tf_idct, [tf.shape(input_tensor)[0],1,1],name='title_idct')
        freq_gt = tf.matmul(tf.matmul(batch_dct, target_tensor, name='freq_gt0'), batch_idct, name='freq_gt1')
        recon = tf.matmul(tf.matmul(batch_idct, freq_tensor, name='mul_dct1'), batch_dct, name='mul_idct1')
        # Note that here recon is of size "batch_size * 32 * 32"
        freq_mse_loss = tf.reduce_mean(tf.square((freq_gt-freq_tensor)))
        pixel_mse_loss = tf.reduce_mean(tf.square((target_tensor-recon)))
        satd_loss = SATD(recon, target_tensor)
        # loss = satd_loss
        loss = freq_mse_loss
        # now we just end the function because we don't need the train_op
        recon = tf.reshape(recon, (-1, block_size, block_size, 1), name='4_dim_out_freq')
        if test:
            return satd_loss, freq_mse_loss, pixel_mse_loss, recon
        
        # for training, we need the train_op
        global_step = tf.Variable(0, trainable=False)
        # learning_rate = tf.train.exponential_decay(params['learning_rate'], global_step=global_step, decay_steps = 10000, decay_rate=0.7)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return train_op, satd_loss, freq_mse_loss, pixel_mse_loss, recon


    else:
        # prediction in pixel domain
        recon = tf.reshape(fc4, (-1, block_size, block_size), name='3_dim_raw_output_pixel')
        mse_loss = tf.reduce_mean(tf.square((target_tensor-recon)))
        satd_loss = SATD(recon, target_tensor)
        # loss = satd_loss
        loss = mse_loss
        
        recon = tf.reshape(recon, (-1, block_size, block_size, 1), name='4_dim_out_pixel')
        if test:
            return satd_loss, mse_loss, recon

        global_step = tf.Variable(0, trainable=False)
        # learning_rate = tf.train.exponential_decay(params['learning_rate'], global_step=global_step, decay_steps = 10000, decay_rate=0.7)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return train_op, satd_loss, mse_loss, recon
