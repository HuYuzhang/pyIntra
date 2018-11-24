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



def build_model(input_tensor, target_tensor, params, mode=3):

    print("mode : %d" % (mode))
    print(input_tensor.shape)
    batch_size = params['batch_size']

    num_scale = params['num_scale']

    input_layer = tf.reshape(input_tensor, [-1, num_scale*mode * num_scale*mode])

    _fc1 = tf.layers.dense(input_layer, 1024, name='fc1')

    fc1 = tf.keras.layers.PReLU(shared_axes=[1], name='relu1')(_fc1)

    _fc2 = tf.layers.dense(fc1, 1024, name='fc2')

    fc2 = tf.keras.layers.PReLU(shared_axes=[1], name='relu2')(_fc2)

    _fc3 = tf.layers.dense(fc2, 1024, name='fc3')

    fc3 = tf.keras.layers.PReLU(shared_axes=[1], name='relu3')(_fc3)

    _fc4 = tf.layers.dense(fc3, 1024, name='fc4')

    fc4 = tf.keras.layers.PReLU(shared_axes=[1], name='relu4')(_fc4)

    _fc5 = tf.layers.dense(fc4, 1024, name='fc5')

    fc5 = tf.keras.layers.PReLU(shared_axes=[1], name='relu5')(_fc5)

    _fc6 = tf.layers.dense(fc5, 1024, name='fc6')

    fc6 = tf.keras.layers.PReLU(shared_axes=[1], name='relu6')(_fc6)

    _fc7 = tf.layers.dense(fc6, 1024, name='fc7')

    fc7 = tf.keras.layers.PReLU(shared_axes=[1], name='relu7')(_fc7)

    _fc8 = tf.layers.dense(fc7, 1024, name='fc8')

    fc8 = tf.keras.layers.PReLU(shared_axes=[1], name='relu8')(_fc8)

    fco = tf.layers.dense(fc8, 64, name='fco')

    conv11 = tf.reshape(fco, (-1,8,8,1))

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

            TH0 = tf.constant(H_8x8.reshape([1,8,8]))

            TH1 = tf.tile(TH0, (batch_size, 1, 1))

            diff = tf.reshape(y_true - y_pred, (-1, 8, 8))

            return tf.reduce_mean(tf.sqrt(tf.square(tf.matmul(tf.matmul(TH1, diff), TH1)) + 0.0001))


    mse_loss = tf.reduce_mean(tf.square((target_tensor-conv11)))
    satd_loss = SATD(conv11, target_tensor)
    loss = satd_loss

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(params['learning_rate'], global_step=global_step, decay_steps = 10000, decay_rate=0.7)
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())


    return train_op, satd_loss, mse_loss
