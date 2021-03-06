import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, inputs_size, outputs_size, activation_function=None):
    weights = tf.Variable(tf.random_normal([inputs_size, outputs_size]))
    biases = tf.Variable(tf.zeros([1, outputs_size]) + 0.1)

    outputs_init = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = outputs_init
    else:
        outputs = activation_function(outputs_init)
    return outputs


# 构建数据
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

# 输入输出都只有1个神经元
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 搭建单隐藏层NN，prediction_layer为输出层
layer1 = add_layer(x_data, 1, 10, activation_function=tf.nn.relu)
prediction_layer = add_layer(layer1, 10, 1, activation_function=None)

# 计算误差损失
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction_layer),
                                    reduction_indices=[1]))

# 优化器
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 必不可少的变量初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion() # 连续画图
plt.show()

# 开始训练
for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to see the step improvement
        # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction_layer, feed_dict={xs: x_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.1)