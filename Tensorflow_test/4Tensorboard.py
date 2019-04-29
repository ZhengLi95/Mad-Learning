import tensorflow as tf
import numpy as np


def add_layer(inputs, inputs_size, outputs_size,
              n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer  # define a new var
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal([inputs_size, outputs_size]), name='W')
            tf.summary.histogram('/weights', weights)

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, outputs_size]) + 0.1, name='b')
            tf.summary.histogram('/biases', biases)

        with tf.name_scope('Wx_plus_b'):
            outputs_init = tf.add(tf.matmul(inputs, weights), biases)

        if activation_function is None:
            outputs = outputs_init
        else:
            outputs = activation_function(outputs_init)

        tf.summary.histogram('/outputs', outputs)

    return outputs


# 构建数据
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

# 输入输出都只有1个神经元
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# 搭建单隐藏层NN，prediction_layer为输出层
layer1 = add_layer(x_data, 1, 10, n_layer=1, activation_function=tf.nn.relu)
prediction_layer = add_layer(layer1, 10, 1, n_layer=2, activation_function=None)

with tf.name_scope('loss'):
    # 计算误差损失
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction_layer),
                                        reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    # 优化器
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 必不可少的变量初始化
init = tf.global_variables_initializer()
sess = tf.Session()
# 将所有的summaries合并在一起
merged = tf.summary.merge_all()
# 将图保存到文件中，才能在浏览器中查看
writer = tf.summary.FileWriter("D:/Anaconda/pkgs/tensorboard-1.9.0-py36he025d50_0/Scripts/logs/", sess.graph)
sess.run(init)

# 开始训练
for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        rs = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(rs, i)
