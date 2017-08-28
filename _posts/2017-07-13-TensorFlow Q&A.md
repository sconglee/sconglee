---
title: TensorFlow Q&A
date: 2017-07-13 17:40:52
math: y
---

#### 1. TensorFlow供给数据（Feeding）错误，具体信息如下图：

```python
def train():
    mnist = getMnist()
    img, label = read_tfrecord()
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                   batch_size=100,
                                                   num_threads=2,
                                                   capacity=1000,
                                                   min_after_dequeue=700)
    init = tf.initialize_all_variables()
    
    with sess.as_default():
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(10000):
            img_np, label_np = sess.run([img_batch, label_batch])
            sess.run(train_step, feed_dict={x:img_np, y_:label_np})
            if (i % 100 == 0):
                print("step %4d " %i)
    print("Accuracy on testdata:", sess.run(accuracy, feed_dict={x:mnist.test.images, y_:tf.cast(tf.arg_max(mnist.test.labels, 1), tf.int32)}))
    coord.request_stop()
    coord.join(threads)
```

![error1](/images/figures/2017-07-13-01.png)

报错意思是被feed进的值不能是一个张量，而应该是Python scalars，strings， lists， 或是arrays中的一种，定位问题出现在倒数第三行的print语句的tf.cast(tf.arg_max(mnist.test.labels, 1), tf.int32)，该值是个tensor，可以改为以下的方式注入参数：

```python
ys = sess.run(tf.cast(tf.arg_max(mnist.test.labels, 1), "int32"))       
print("Accuracy on testdata:", sess.run(accuracy, feed_dict={x:mnist.test.images, y_:ys}))
```
tips：Tensorflow的数据供给机制允许在运算图中将数据注入到任一张量中，但必须通过run()或者eval()函数输入feed_dict参数，才可以启动运算过程。而且设计placeholder节点的意图就是为了提供数据供给的方法，该节点在声明时是未初始化的，也不包含数据，所以如果没有给它feed数据，则Tensorflow运算的时候会产生错误。

#### 2. 计算图的误用

```python
# num_to_char()是自定义的函数，用于将给定的value转化为key
preValue = num_to_char(tf.arg_max(tf.nn.softmax(y), 1))
print "The prediction value is:", sess.run(preValue, feed_dict={x:testPicArr})
```
![error2](/images/figures/2017-07-13-02.png)

可以看到报错提示是无效的参数类型，该处run()中的preValue应该是一个tensor。而且num_to_char(tf.arg_max(tf.nn.softmax(y), 1))也不是一个有效的值，因为tf.arg_max(tf.nn.softmax(y), 1)也是tensor，没有在session中run之前，它仅仅是一个计算图的节点符号，没有实际值的意义，所以num_to_char()返回的是None。可以做如下修改

```python
# num_to_char()是自定义的函数，用于将给定的value转化为key
preValue = tf.arg_max(tf.nn.softmax(y), 1)
value = sess.run(preValue, feed_dict={x:testPicArr})
print "index is: " , value, "The prediction value is:", num_to_char(value)
```
tips：Tensorflow是一个编程系统，仅仅使用图来表示计算任务，图中的每个节点有0或多个tensor，多个节点组成的图描述了计算的过程，为了进行计算，该图必须在会话里启动，从而返回对应的值类型。

#### 3. tf.nn.conv2d()中参数padding_method的使用

这里引用Stack Overflow中的一个解释：

SAME：means that the output feature map has the same spatial dimensions as the input feature map. Zero padding is introduced to make the shapes match as needed, equally on every side of the input map.

VALID: means no padding.

追踪源码至tensorflow/tensorflow/python/ops/nn_ops.py
```python
def convolution(input, filter, padding, strides=None, dilation_rate=None, 
                           name=None, data_format=None):
                           
    Args:
      padding: A string, either `"VALID"` or `"SAME"`. The padding algorithm.
      dilation_rate: Optional, Specifies the filter upsampling/input downsampling rate, usually dilation_rate=1.
     
    Returns:
        If padding == "SAME":
            output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])
        If padding == "VALID":
            output_spatial_shape[i] =ceil((input_spatial_shape[i] -
              (spatial_filter_shape[i]-1) * dilation_rate[i]) / strides[i]).
```
整理下就是，对于“VALID”，输出的形状计算如下：

$$new\_shape_{i}=\lceil\dfrac{(W_{i}-F_{i}+1)}{S_{i}}\rceil$$

对于“SAME”，输出的形状计算如下：

$$new\_shape_{i}=\lceil\dfrac{W_{i}}{S_{i}}\rceil$$

其中，$i$是对应的维度，$W$为输入的size，$F$是filter的size，$S$是步长，$\lceil\rceil$是向上取整符号。

#### 4. softmax_cross_entropy_with_logits和sparse_softmax_cross_entropy_with_logits的区别

官方给出的解释还是简单明了的

- For sparse_softmax_cross_entropy_with_logits, labels must have the shape [batch_size] and the dtype int32 or int64. Each label is an int in range [0, num_classes-1].
- For softmax_cross_entropy_with_logits, labels must have the shape [batch_size, num_classes] and dtype float32 or float64.
- Labels used in softmax_cross_entropy_with_logits are the one hot version of labels used in sparse_softmax_cross_entropy_with_logits.

两个函数返回值是一样的，但都不是一个数，而是一个向量，如果要求loss，则要做一步tf.reduce_mean()操作，即对向量求均值。
