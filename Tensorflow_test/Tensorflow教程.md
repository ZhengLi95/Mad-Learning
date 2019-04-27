## Tensorflow教程

### 基本结构

![img](./Tensorflow教程.assets/tensors_flowing.gif)

> TensorFlow是采用数据流图（data flow graphs）来计算, 所以首先要创建一个数据流图， 然后再将数据（数据以张量(tensor)的形式存在）放在数据流图中计算。节点（Nodes）在图中表示数学操作，图中的线（edges）则表示在节点间相互联系的多维数据数组， 即张量（tensor)。 训练模型时tensor会不断的从数据流图中的一个节点flow到另一节点, 这就是TensorFlow名字的由来。
>
> [Code](./1Basic_Structure.py)

**张量（Tensor）：**

- 张量有多种，零阶张量为 纯量或标量 (scalar) 也就是一个数值. 比如 `[1]`
- 一阶张量为 **向量 (vector)**, 比如 一维的 `[1, 2, 3]`
- 二阶张量为 **矩阵 (matrix)**, 比如 二维的 `[[1, 2, 3],[4, 5, 6],[7, 8, 9]]`
- 以此类推, 还有 三阶 三维的 …

**会话（Session）**：

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

**变量（Variable）：**

变量定义语法： `state = tf.Variable()`

常量定义语法：`one = tf.constant(1)`

[代码](./2Variable.py)

**占位符（Placeholder）：**

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

