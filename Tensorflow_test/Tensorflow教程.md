## Tensorflow教程

### 基本结构

![img](./Tensorflow教程.assets/tensors_flowing.gif)

> TensorFlow是采用数据流图（data flow graphs）来计算, 所以首先要创建一个数据流图， 然后再将数据（数据以张量(tensor)的形式存在）放在数据流图中计算。节点（Nodes）在图中表示数学操作，图中的线（edges）则表示在节点间相互联系的多维数据数组， 即张量（tensor)。 训练模型时tensor会不断的从数据流图中的一个节点flow到另一节点, 这就是TensorFlow名字的由来。
>
> [Code](./1Basic_Structure.py)

#### 张量（Tensor）

- 张量有多种，零阶张量为 纯量或标量 (scalar) 也就是一个数值. 比如 `[1]`
- 一阶张量为 **向量 (vector)**, 比如 一维的 `[1, 2, 3]`
- 二阶张量为 **矩阵 (matrix)**, 比如 二维的 `[[1, 2, 3],[4, 5, 6],[7, 8, 9]]`
- 以此类推, 还有 三阶 三维的 …

#### 会话（Session）

`Session` 是 Tensorflow 为了控制和输出文件的执行的语句。运行 `session.run()` 可以获得你要得知的运算结果，或者是你所要运算的部分。

```python
import tensorflow as tf

# create two matrixes

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1,matrix2) 
# product并没有实际进行计算，需要sess.run()来激活

# method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()
# [[12]]

# method 2
# 自动close
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)
# [[12]]
```

#### 变量（Variable）

变量定义语法： `state = tf.Variable()`

常量定义语法：`one = tf.constant(1)`

[Code](./2Variable.py)

#### 占位符（Placeholder）

用于暂时存储变量，要想从外部传入data，就必须使用`tf.placeholder()`，然后以 `sess.run(***, feed_dict={input: **})`形式传输数据。

```python
import tensorflow as tf

#在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# mul = multiply 是将input1和input2 做乘法运算，并输出为 output 
ouput = tf.multiply(input1, input2)
```

接下来, 传值的工作交给了 `sess.run()` , 需要传入的值放在了`feed_dict={}` ，注意此处是传入字典。

```python
with tf.Session() as sess:
    print(sess.run(ouput, feed_dict={input1: [7.], input2: [2.]}))
# [ 14.]
```

#### 激励函数（Activation Function)

用于非线性映射，激活神经网络中某一部分神经元，将激活信息向后传入下一层的神经系统。



### 建立神经网络

[Code](./3Build_a_NN.py)

```python
# 用添加层的方式构建网络
def add_layer(inputs, inputs_size, outputs_size, activation_function=None):
    weights = tf.Variable(tf.random_normal([inputs_size, outputs_size]))
    biases = tf.Variable(tf.zeros([1, outputs_size]) + 0.1)

    outputs_init = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = outputs_init
    else:
        outputs = activation_function(outputs_init)
    return outputs
```

使用matplotlib实现训练过程的可视化

1. 创建图像

   ```python
   fig = plt.figure()
   ax = fig.add_subplot(1, 1, 1)
   ax.scatter(x_data, y_data)
   plt.ion() # 连续画图
   plt.show()
   ```

2. 训练时循环绘制预测的曲线

   ```python
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
   ```

   

### 优化函数详解 (Speed Up Training)

优化算法作用就是，用来调节模型参数，进而使得损失函数最小化，（此时的模型参数就是那个最优解）目前优化算法求解损失函数的最优解一般都是迭代的方式。

- 如果是凸优化问题，如果数据量特别大，那么计算梯度非常耗时，因此会选择使用迭代的方法求解，迭代每一步计算量小，且比较容易实现。
- 对于非凸问题，只能通过迭代的方法求解，每次迭代目标函数值不断变小，不断逼近最优解。
- 因此优化问题的重点是使用何种迭代方法进行迭代，即求迭代公式。

目前优化算法有以下几种：

- Stochastic Gradient Descent (SGD)
- Momentum
- AdaGrad
- RMSProp
- Adam

各方法速度如下：

![img](./Tensorflow教程.assets/1.gif)

各方法在鞍点的表现如下：

![优化函数2.gif](D:\Github\Mad-Learning\Tensorflow_test\Tensorflow教程.assets\2.gif)

一般线性回归函数的假设函数为：
$$
h_θ=∑^n_{j=0}θ_jx_j
$$
对应的能量函数（损失函数）形式为：
$$
J_{train}(\theta)=\dfrac{1}{2m}\sum^m_{i=1}(h_{\theta}(x^{(i)})-y^{(i)})^2
$$

此处，$\frac{1}{2}$为方便求导之后的化简

#### Gradient Descent (GD)

梯度下降（GD）是最小化风险函数、损失函数的一种常用方法。一般而言，GD代表批量梯度下降Batch Gradient Descent(BGD)，它的具体思路是在更新每一参数时都使用所有的样本来进行更新，其数学形式如下：
$$
\begin{align*}\\
&\dfrac{\partial J(\theta)}{\partial \theta_j}=-\dfrac{1}{m}\sum^m_{i=1}(y^i-h_{\theta}(x^i))x^i_j \\
repeat\{  &\\
&\theta_j'=\theta_j+\dfrac{\alpha}{m}\sum^m_{i=1}(y^i-h_{\theta}(x^i))x^i_j\\
&(for\  every \ j=0,\dots,n)\\
&\}\\
\end{align*}
$$

　　**优点：**全局最优解；易于并行实现；

　　**缺点：**当样本数目很多时，训练过程会很慢。

```python
import numpy as np
import pylab
from sklearn.datasets.samples_generator import make_regression


def bgd(alpha, x, y, numIterations):
    """Copied from Internet"""
    m = x.shape[0]  # number of samples
    theta = np.ones(2)
    J_list = []

    x_transpose = x.transpose()
    for iter in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = y - hypothesis
        J = np.sum(loss ** 2) / (2 * m)  # cost
        J_list.append(J)
        print("iter %s | J: %.3f" % (iter, J))

        gradient = np.dot(x_transpose, loss) / m
        theta += alpha * gradient  # update

    pylab.plot(range(numIterations), J_list, "k-")
    return theta
```

#### Stochastic Gradient Descent (SGD)

随机梯度下降法是为了解决批量梯度下降法样本数目过多时，训练过程很慢的问题。
$$
\begin{align*}
repeat\{  &\\
&for \ i=1,\dots,m\{ &\\
& &\theta_j'=\theta_j+\dfrac{\alpha}{m}\sum^m_{i=1}(y^i-h_{\theta}(x^i))x^i_j\\
&&(for\  every \ j=0,\dots,n)\\
\ &\}\\
&\}\\
\end{align*}
$$
　　**优点：**训练速度快；支持在线学习；

　　**缺点：**准确度下降，非全局最优；不易于并行实现。

```python
def sgd(alpha, x, y, num_iter):
    """Writtern by kissg"""
    m = x.shape[0]  # number of samples
    theta = np.ones(2)
    J_list = []

    # 随机化序列
    idx = np.random.permutation(y.shape[0])
    x, y = x[idx], y[idx]

    for j in range(num_iter):

        for i in idx:
            single_hypothesis = np.dot(x[i], theta)
            single_loss = y[i] - single_hypothesis
            gradient = np.dot(x[i].transpose(), single_loss)
            theta += alpha * gradient  # update

        hypothesis = np.dot(x, theta)
        loss = y - hypothesis
        J = np.sum(loss ** 2) / (2 * m)  # cost
        J_list.append(J)
        print("iter %s | J: %.3f" % (j, J))

    pylab.plot(range(num_iter), J_list, "r-")
    return theta
```

```python
import tensorflow as tf

optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
```



#### Mini-Batch Gradient Descent (MBGD)（常用）

MBGD 是为解决 BGD 与 SGD 各自缺点而发明的折中算法，或者说它利用了 BGD 和 SGD 各自优点。其基本思想是: *每次更新参数时，使用 n 个样本，既不是全部，也不是 1。*(SGD 可以看成是 n=1 的 MBGD 的一个特例)
$$
\begin{align*}
&Say\ b=10, m=1000 \\
&repeat\{ \\
&for \ i=1,11,21\dots,991\{ \\
&&\theta_j'=\theta_j+\dfrac{\alpha}{10}\sum^{i+9}_{k=i}(y^k-h_{\theta}(x^k))x^k_j\\
&&(for\  every \ j=0,\dots,n)\\
&\}\\
&\}\\
\end{align*}
$$

```python
def mbgd(alpha, x, y, num_iter, minibatches):
    """Writtern by kissg"""
    m = x.shape[0]  # number of samples
    theta = np.ones(2)
    J_list = []

    for j in range(num_iter):

        idx = np.random.permutation(y.shape[0])
        x, y = x[idx], y[idx]
        mini = np.array_split(range(y.shape[0]), minibatches)

        for i in mini:
            mb_hypothesis = np.dot(x[i], theta)
            mb_loss = y[i] - mb_hypothesis
            gradient = np.dot(x[i].transpose(), mb_loss) / minibatches
            theta += alpha * gradient  # update

        hypothesis = np.dot(x, theta)
        loss = y - hypothesis
        J = np.sum(loss ** 2) / (2 * m)  # cost
        J_list.append(J)
        print("iter %s | J: %.3f" % (j, J))

    pylab.plot(range(num_iter), J_list, "y-")
    return theta
```

![梯度下降, 成本-时间曲线](D:\Github\Mad-Learning\Tensorflow_test\Tensorflow教程.assets\gradient_descent_cost-iteration.png)

- 黑色是 BGD 的图像，是一条光滑的曲线， 因为 BGD 每一次迭代求得的都是全局最优解;
- 红色是 SGD 的图像， 可见抖动很剧烈，有不少局部最优解;
- 黄色是 MBGD 的图像，相对 SGD 的成本-时间曲线平滑许多，但仔细看，仍然有抖动.

上述三个方法面临的主要挑战如下：

- 选择适当的学习率α \alphaα 较为困难。太小的学习率会导致收敛缓慢，而学习速度太块会造成较大波动，妨碍收敛。
- 目前可采用的方法是在训练过程中调整学习率大小，例如模拟退火算法：预先定义一个迭代次数m，每执行完m次训练便减小学习率，或者当cost function的值低于一个阈值时减小学习率。然而迭代次数和阈值必须事先定义，因此无法适应数据集的特点。
- 上述方法中, 每个参数的 learning rate 都是相同的，这种做法是不合理的：如果训练数据是稀疏的，并且不同特征的出现频率差异较大，那么比较合理的做法是对于出现频率低的特征设置较大的学习速率，对于出现频率较大的特征数据设置较小的学习速率。
- 近期的的研究表明，深层神经网络之所以比较难训练，并不是因为容易进入local minimum。相反，由于网络结构非常复杂，在绝大多数情况下即使是 local minimum 也可以得到非常好的结果。而之所以难训练是因为学习过程容易陷入到马鞍面中，即在坡面上，一部分点是上升的，一部分点是下降的。而这种情况比较容易出现在平坦区域，在这种区域中，所有方向的梯度值都几乎是 0。


#### Momentum

Momentum算法借用了物理中的动量概念，它模拟的是物体运动时的惯性，即更新的时候在一定程度上保留之前更新的方向，同时利用当前batch的梯度微调最终的更新方向。这样一来，可以在一定程度上增加稳定性，从而学习地更快，并且还有一定摆脱局部最优的能力：![这里写图片描述](D:\Github\Mad-Learning\Tensorflow_test\Tensorflow教程.assets\20170728165011954.jfif)
$$
\begin{align}
\theta_{a}&=\theta-\alpha\nabla a\\
v_t&=\gamma v_{t-1}+\alpha\nabla b\\
\theta_{b}&=\theta-v_t
\end{align}
$$
其中，$v_{t-1}$表示之前所有步骤所累积的动量和。

![这里写图片描述](D:\Github\Mad-Learning\Tensorflow_test\Tensorflow教程.assets\20170728172055239.png)![这里写图片描述](D:\Github\Mad-Learning\Tensorflow_test\Tensorflow教程.assets\20170728172411270.png)

```python
optimizer=tf.train.MomentumOptimizer(learning_rate,momentum,use_locking=False,name='Momentum',use_nesterov=False).minimize(loss)
```

#### AdaGrad 

**特点：**变学习率
$$
\begin{align}
G_{i,t}&=G_{i,t-1}+\nabla_{\theta_{i,t}}J(\theta)\\
\theta_{i,t+1}&=\theta_{i,t}-\dfrac{\eta}{\sqrt{G_{i,t}+\epsilon}}\nabla_{\theta_{i,t}}J(\theta)
\end{align}
$$
容易看出，随着算法不断的迭代，$G_t$ 会越来越大，整体的学习率会越来越小。所以一般来说adagrad算法一开始是激励收敛，到了后面就慢慢变成惩罚收敛，速度越来越慢。

t代表每一次迭代。 $\epsilon$ 一般是一个极小值，作用是防止分母为0 。$G_{i,t} $ 表示了前 $t$ 步参数 $\theta_i$梯度的累加

#### RMSProp

RMSProp=Momentum+AdaGrad 

同时拥有 Momentum 的惯性原则和AdaGrad的对错误方向的阻力

#### Adam 

```python
optimizer=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,name='Adam').minimize(loss)
```

