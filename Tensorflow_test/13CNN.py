import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


# 初始化权重参数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 初始化偏差参数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


'''
定义卷积
x是图片的所有参数，W是此卷积层的权重
定义步长strides=[1,1,1,1]值，strides[0]和strides[3]的两个1是默认值，中间两个1代表padding时在x方向运动一步，y方向运动一步
'''


def conv2d(x, W):
    # tf.nn.conv2d函数是tensoflow里面的二维卷积函数
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


'''
定义池化
池化的作用: padding时是一次一步的, 图像尺寸没有改变, 为了压缩图像减少参数
           用以稀疏化参数
池化由两种: 最大池化, 平均池化
池化的核函数大小为2x2，因此ksize=[1,2,2,1]，步长为2，因此strides=[1,2,2,1]:
'''


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)  # dropout

# 改变xs的形状, -1代表不考虑输入图像的数量
# 1代表通道数, 黑白图为1, RGB为3
x_image = tf.reshape(xs, [-1, 28, 28, 1])

'''建立卷积层1+池化层1'''
# 设定深度为32
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# 卷积层1输出为: 28x28x32
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 池化层1输出为: 14x14x32
h_pool1 = max_pool_2x2(h_conv1)

'''建立卷积层2+池化层2'''
# 设定深度为64
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
# 卷积层2输出为: 14x14x64
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# 池化层2输出为: 7x7x64
h_pool2 = max_pool_2x2(h_conv2)

'''建立全连接层'''
# 将上面的三维数据展成一维
# [n_samples,7,7,64]->>[n_samples,7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
'''全连接层1'''
# 将深度继续扩展到1024
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
'''全连接层2'''
# 输出为十类
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+ b_fc2)

'''利用交叉熵定义损失函数'''
cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(ys * tf.log(prediction),
                   reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000]))
