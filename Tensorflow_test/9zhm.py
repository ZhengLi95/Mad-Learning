import tensorflow as tf
import numpy as np

x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*100+50

#create tensorflow structure#

weights=tf.Variable(tf.random_uniform([1],-100,200))
biases=tf.Variable(tf.zeros([1]))
y=weights*x_data+biases
loss=tf.reduce_mean(tf.square(y-y_data))
optimizer=tf.train.GradientDescentOptimizer(0.7)
train=optimizer.minimize(loss)
init=tf.initialize_all_variables()
#create tensorflow structure end#

sess=tf.Session()
sess.run(init)

for step in range(200):
    sess.run(train)
    if step%20==0:
        print(step,sess.run(weights),sess.run(biases))
