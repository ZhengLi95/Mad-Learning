import tensorflow as tf
import numpy as np


def add_layer(inputs, inputs_size, outputs_size, activation_function=None):
    weights = tf.Variable(tf.random_normal([inputs_size, outputs_size]))
    biases = tf.Variable(tf.zeros([1, outputs_size]) + 0.1)

    outputs_init = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = outputs_init
    else:
        outputs = activation_function(outputs_init)
    return outputs

x_data=np.linspace(-1,1,300,dtype=np.float32)