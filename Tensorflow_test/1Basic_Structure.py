import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

### create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# 权重随机数列生成，[1]代表一维，范围-1~1
biases = tf.Variable(tf.zeros([1]))

# 预测的y
y = Weights * x_data + biases

# 平方损失函数
loss = tf.reduce_mean(tf.square(y - y_data))

# 创建优化器，来减少损失, 0.5为learning rate, 训练方法为GradientDescentOptimizer
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 上述步骤已经构建好了图的结构，下面进行初始化
init = tf.global_variables_initializer()
### create tensorflow structure start ###

# 创建会话，用Session执行初始化
sess = tf.Session()
sess.run(init)  # 激活初始化

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases), sess.run(loss))
