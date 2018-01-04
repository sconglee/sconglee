---
title: New checkpoint format in TensorFlow
date: 2018-01-04 11:40:42
math: y
---

从版本1.2.0之后，TensorFlow模型的checkpoint文件格式发生了变化，使restore之前老版本模型文件时报错。

##### According to the TensorFlow v1.2.0 RC0’s release note:

New checkpoint format becomes the default in tf.train.Saver. Old V1 checkpoints continue to be readable; controlled by the write_version argument, tf.train.Saver now by default writes out in the new V2 format. It significantly reduces the peak memory required and latency incurred during restore.

可以看出，tf.train.Saver函数有一个参数write_version，用于指定的新的checkpoint文件格式，新版本是V2，老版本是V1。保存的模型文件如下：

old format(v1) | new format(v2) 
:--:|:--:|:--:
model-1000.ckpt|model-1000.index
model-1000.meta|model-1000.meta
|model-1000.data

有些时候我们要使用旧版本的模型，该怎么办尼，查看[Saver API](https://www.tensorflow.org/api_docs/python/tf/train/Saver)可以得知tf.train.Saver 的定义如下：

Saver类用于保存和恢复变量

Methods
```python
__init__(
    var_list=None,
    reshape=False,
    sharded=False,
    max_to_keep=5,  # 要保留的最近checkpoint文件的最大数
    keep_checkpoint_every_n_hours=10000.0, # 训练时每N小时保留一个检查点文件
    name=None,
    restore_sequentially=False,
    saver_def=None,
    builder=None,
    defer_build=False,
    allow_empty=False,
    write_version=tf.train.SaverDef.V2, # 指定保存模型版本
    pad_step_number=False,
    save_relative_paths=False,
    filename=None
)
```
所以，我们可以通过传指定参数来保存旧版本的模型：

```python
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
...
saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V1)
saver.save(sess, './model.ckpt', global_step=step)
...
```
但是这样保存模型会有提示警告信息：

```python
WARNING:tensorflow:*******************************************************
WARNING:tensorflow:TensorFlow's V1 checkpoint format has been deprecated.
WARNING:tensorflow:Consider switching to the more efficient V2 format:
WARNING:tensorflow: `tf.train.Saver(write_version=tf.train.SaverDef.V2)`
WARNING:tensorflow:now on by default.
WARNING:tensorflow:*******************************************************
```
whatever, 但如果想获得更好的性能，还是推荐使用新的模型文件格式V2。

***