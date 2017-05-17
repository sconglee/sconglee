---
layout: post
title: "TensorFlow学习笔记（一）--编程模型"
date: 2017-05-17 17:50:11.000000000 +09:00
categories: [python]
---
TensorFlow中涉及到三种模型，计算模型、数据模型和运行模型，通过这三个角度对TensorFlow开始学习，可以对其工作原理有一个整体的理解。

## TensorFlow计算模型——计算图
计算图是TensorFlow中最基本的一个概念，TensorFlow的所有计算都会被转化为计算图上的节点
### 计算图的概念
其实TensorFlow的名字已经说明了最重要的两个概念——tensor和flow，tensor就是张量，可以被简单的理解为多维数组，也就是它的数据结构（后续介绍），而flow则体现了它的计算模型，flow意为“流”，很直观的表达了张量之间通过计算相互转化的过程。TensorFlow是一个通过计算图的方式来表述计算的编程系统，它的每一个计算都是计算图上的一个节点，而节点之间的边描述了计算之间的依赖关系。
### 计算图的使用
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
