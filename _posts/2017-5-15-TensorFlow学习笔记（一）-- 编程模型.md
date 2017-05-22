---
title: TensorFlow学习笔记（一）-- 编程模型
date: 2017-05-15 17:50:11
---
TensorFlow中涉及到三种模型，计算模型、数据模型和运行模型，通过这三个角度对TensorFlow开始学习，可以对其工作原理有一个整体的理解。

### TensorFlow计算模型——计算图
计算图是TensorFlow中最基本的一个概念，TensorFlow的所有计算都会被转化为计算图上的节点
#### 计算图的概念
其实TensorFlow的名字已经说明了最重要的两个概念——tensor和flow，tensor就是张量，可以被简单的理解为多维数组，也就是它的数据结构（后续介绍），而flow则体现了它的计算模型，flow意为“流”，很直观的表达了张量之间通过计算相互转化的过程。TensorFlow是一个通过计算图的方式来表述计算的编程系统，它的每一个计算都是计算图上的一个节点，而节点之间的边描述了计算之间的依赖关系。
#### 计算图的使用
TensorFlow程序一般可分为两个阶段，在第一个阶段需要定义计算图中所有的计算，第二个阶段为执行阶段（后续介绍），以下是计算定义阶段的示例：
```python
import tensorflow as tf
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a + b
```
在这个过程中，TensorFlow会自动将定义的计算转化为计算图上的节点。在TensorFlow中，系统会自动维护一个默认的计算图，通过tf.get_default_graph函数可以获取当前默认的计算图，当然TensorFlow也支持通过tf.Graph函数来生成新的计算图，但不同计算图的上的张量和运算都不会共享。

TensorFlow中的计算图不仅可以用来隔离张量和计算，还提供了管理张量和计算的机制，计算图可以通过tf.Graph.device函数来指定运行计算的设备，这也为TensorFlow使用GPU提供了机制，以下程序指定计算跑在GPU上
```python
g = tf.Graph()
with g.device('/gpu:0')
    result = a + b
```
### TensorFlow数据模型——张量
#### 张量的概念
在TensorFlow中，所有的数据都通过张量的形式来表示。从功能的角度看，张量可以简单理解为多维数据，其中零阶张量表示标量(scalar)，也就是一个数，第一阶张量为向量(vector)，也就是一个一维数组，第n阶张量可以理解为一个n维数组。但张量在TensorFlow中的实现并不是直接采用数组的形式，它只是对其运算结果的引用，在张量中并没有真正保存数字，它保存的是如何得到这些结果的计算过程，这里还是以向量加法为例，以下得到的是对结果的一个引用
```python
import tensorflow as tf
#tf.constant是一个计算，这个计算的结果为一个张量，保存在变量a中
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = tf.add(a, b, name = "add")
print result

输出：
Tensor("add:0", shape=(2,), dtype=float32)
```
从上面代码可以看出，运行结果是一个张量的结构，主要保存了三个属性：名字(name)、维度(shape)和类型(type)。

张量的第一个属性名字不仅是一个张量的唯一标识，同样也给出了这个张量是如何计算出来的。前面说计算图上的每一个节点代表了一个计算，计算的结果就保存在张量中，所以张量和计算图上所代表的计算结果是对应的，这样张量的命名就可以通过“node:src_output”的形式给出，其中node是节点的名称，src_output表示当前张量来自节点的第几个输出。如上面“add:0”就表示result这个张量是计算节点“add”输出的第一个结果。

张量的维度(shape)，这个属性描述了一个张量的维度信息。还有类型(type)属性，每一个张量会有唯一的类型，TensorFlow会对参与运算的所有张量进行类型的检查，当发现类型不匹配时会报错，如把以上代码改为：
```python
a = tf.constant([1, 2], name="a")

就会报错：
valueError: Tensor conversion requested dtype int32 for Tensor with dtype float32: 'Tensor("b:0", shape=(2,), dtype=float32)'
```
TensorFlow支持的数据类型包括，实数(tf.float32、tf.float64)、整数(tf.int8、tf.int16、tf.int32、tf.int64、tf.uint8)、布尔型(tf.bool)和复数(tf.complex64、tf.complex128)。
#### 张量的使用
和TensorFlow的计算模型相比，其数据模型相对比较简单。张量使用可以总结为两大类，第一类是对中间结果的引用，当计算的复杂度增加时，使用张量来引用中间结果可以大大提高代码的可读性，这样也可以方便的获取中间结果，比如在卷积神经网络中，卷积层或者池化层可能改变张量的维度，通过result.get_shape函数可以获取张量的维度信息，以免去人工计算的麻烦；使用张量的另一类情况是当计算图构造完成后，通过张量(tf.Session().run(result))可以获得计算结果，也就是真是的数字。
### TensorFlow运行模型——会话
TensorFlow使用会话(Session)来执行定义好的运算，会话拥有并管理TensorFlow程序运行时的所有资源。当所有计算完成后需要关闭会话来帮助系统回收资源，否则就可能出现资源泄漏的问题。TensorFlow使用会话的模式一般有两种，第一种模式需要明确调用、关闭会话，如下所示:
```python
# 创建一个会话
sess = tf.Session()
# 使用这个会话来得到期望的结果
sess.run(result)
# 关闭会话
sess.close()
```
使用这种模式时，当程序因为异常而退出时，关闭会话的函数就不会被执行而导致资源泄漏，为了解决这问题，TensorFlow可以通过python的上下文管理器来使用会话，以下代码展示了这种方式：
```python
# 通过python的上下文机制，只要将所有的计算放在“with”内部就可以了
with tf.Session() as sess:
    sess.run()
# 不需要调用sess.close()来关闭会话
# 当上下文退出时会话关闭和资源释放也就自动完成了
```
前面有说过TensorFlow可以会自动生成一个默认的计算图，如果没有特殊指定，运算会自动加入这个计算图中。而会话也有类似的机制，但不会自动生成默认的会话，而是需要手动指定，如下所示：
```python
sess = tf.Session()
with sess.as_default():
    print(result.eval())
```
以下代码有类似的功能
```python
sess = tf.Session()
print(result.eval(session=sess)
sess.close()
```
另外在交互式环境下，TensorFlow也提供了一种直接构建默认会话的方式，下面为其用法：
```python
# 该函数会自动将生成的会话注册为默认的会话
sess = tf.InteractiveSession()
print(result.eval())
sess.close()
```
无论使用哪种方法都可以通过ConfigProto Protocol Buffer来配置需要生成的会话
```python
config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=True)
sess1 = tf.InteractiveSession(config=config) # or sess2 = tf.Session(config=config)
```
参数allow_soft_placement，其值为布尔型，当为True时，在以下任一一个条件成立的时候，GPU上的运算可以放到CPU上进行：

1、运算无法在GPU上执行

2、没有GPU资源(比如运算被指定到第二个GPU上，但机器上只有一个GPU)

3、运算输入包含对CPU计算结果的引用

这个参数的默认值是False，但是为了使得代码的可移植性更强，在有GPU的情况下，这个参数一般会设置为True，保证程序在拥有不同数量GPU的机器上顺利运行。

参数log_device_placement，当值为True时日志中将会记录每个节点被安排在了哪个设备上，以方便调试，不过生产环境中最好置为False以减少产生日志。
