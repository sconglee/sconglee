---
title: TensorFlow学习笔记（五）-- 深度神经网络（DNN）反向传播算法（BP）
date: 2017-06-27 20:44:52
math: y
---
#### DNN反向传播算法 for what ?
在了解DNN的反向传播算法前，我们要先搞清楚反向传播算法是用来解决什么问题的，也就是说，什么时候我们需要使用这个反向传播算法。

这里有个监督学习的一般问题，假设我们有m个训练样本：$\{(x_1,y_1),(x_2,y_2),...,(x_m,y_m)\}$，其中$x$ 是输入向量，特征维度为$$n\_in$$，而$y$是输出向量，特征维度是$$n\_out$$。现在我们需要利用这个样本训练出一个模型，当有一个新的测试样本$(x_{test},?)$，我们可以预测$y_{test}$向量的输出。

如果我们采用DNN的模型，即输入层有$$n\_in$$个神经元，输入层有$$n\_out$$个神经元，再加上一些含有若干神经元的隐藏层。此时我们还需要找到合适的所有隐藏层和输出层对应的线性系数矩阵$W$，偏置向量$b$，让所有输入的训练样本计算出的输出尽可能等于或很接近样本的输出。那么问题就来了，怎么找到合适的参数尼？

了解传统机器学习算法的优化过程的话，就会很容易想到可以用一个合适的损失函数来度量训练样本的输出损失，接着对这个损失函数进行优化求最小化的极值，对应的一系列线性系数矩阵$W$，以及偏置向量$b$即为我们的最终结果。在DNN中，损失函数优化极值求解的过程最常见的一般是通过梯度下降法来一步步迭代完成的，当然也可以是其他的迭代方法比如牛顿法与拟牛顿法。

可以知道，**对DNN的损失函数用梯度下降法进行迭代优化求极值的过程就是我们所说的反向传播算法**。
#### DNN反向传播算法的基本思路
为便于通俗直观的理解该算法，这里我定义了个三层网络，第0层输入层，第一层隐藏层，第二层输出层，并且每个节点没有偏置（有偏置原理完全一样），激活函数为sigmod，其中使用到的符号说明如下：

| 符号 | 描述|
| :--: | :--: |
|  $W_{ab}$  | 代表的是节点a到节点b的权重 |
|  $y_a$  | 代表的是节点a的输出值 |
|  $Z_a$  | 代表的是节点a的输入值 |
|  $C$  | 最终损失函数 |
|  $f(x)=\dfrac{1}{1+e^{-x}}$  | 节点激活函数 |
|  $W1$  | 左边字母，右边数字，代表第几层的权重 |

对应的网络如下：

![神经网络](http://i4.piimg.com/595056/bb2fa853f940a3d0.png)

 $$X=Z_0=\left[
         \begin{matrix}
         0.35 \\
         0.9 
         \end{matrix}
         \right]$$ 
         
 $$y_{out}=0.5$$ 

 $$W0=\left[
     \begin{matrix}
     w_{31} & w_{32} \\
     w_{41} & w_{42}
     \end{matrix}
     \right]
    =\left[
    \begin{matrix}
    0.1 & 0.8 \\
    0.4 & 0.6 
    \end{matrix}
    \right]$$
    
$$W1=\left[
     \begin{matrix}
     w_{53} & w_{54}
     \end{matrix}
     \right]
    =\left[
    \begin{matrix}
    0.3 & 0.9
    \end{matrix}
    \right]$$
    
首先我们先走一遍**正向传播过程**，如下推导:

$$Z_1=\left[
      \begin{matrix}
      Z_3 \\
      Z_4 
      \end{matrix}
      \right]
    =W0*X
    =\left[
     \begin{matrix}
     w_{31} & w_{32} \\
     w_{41} & w_{42}
     \end{matrix}
     \right]*
     \left[
     \begin{matrix}
     x_1 \\
     x_2
     \end{matrix}
     \right]
    =\left[
     \begin{matrix}
     w_{31}*x_1 + w_{32}*x_2 \\
     w_{41}*x_1 + w_{42}*x_2
     \end{matrix}
     \right]
    =\left[
     \begin{matrix}
     0.755 \\
     0.68
     \end{matrix}
     \right]$$

那么隐藏层的输出为：

$$y_1=\left[
     \begin{matrix}
     y_3 \\
     y_4
     \end{matrix}
     \right]
    =f(Z_1)
    =f(\left[
     \begin{matrix}
     0.755 \\
     0.68
     \end{matrix}
     \right])
    =f(\left[
     \begin{matrix}
     0.680 \\
     0.663
     \end{matrix}
     \right])$$

同理可以得到：

$$Z_2=W1*y_1
     =\left[
     \begin{matrix}
     W_{53} & W_{54}
     \end{matrix}
     \right]*
    \left[
     \begin{matrix}
     y_3 \\
     y_4
     \end{matrix}
     \right]
    =\left[
     \begin{matrix}
     0.801
     \end{matrix}
     \right]$$
     
$$y_2=f(Z_2)
     =f(\left[
     \begin{matrix}
     0.801
     \end{matrix}
     \right])
    =\left[
     \begin{matrix}
     0.690
     \end{matrix}
     \right]$$
     
那么最终的损失为：

$$C=\dfrac{1}{2}(0.690-0.5)^2=0.01805$$

对于这个损失值，我们当然是希望这个值越小越好。这也是我们进行多次训练，调节参数的目的，在这个训练的过程中就用到了我们的反向传播算法，实际上反向传播就是梯度下降法中链式法则的使用。

下面是**反向传播的推导过程**

根据公式，我们有：

$$\begin{cases}
  Z_2=W_{53}*y_3+W_{54}*y_4 \\
  y_2=f(Z_2) \\
  C=\dfrac{1}{2}(y_2-y_{out})^2 
  \end{cases}$$

这个时候我们需要求出$C$对$W$的偏导，则根据链式法则有：

$$\dfrac{\partial C}{\partial W_{53}}=\dfrac{\partial C}{\partial y_5}*\dfrac{\partial y_5}{\partial Z_5}*\dfrac{\partial Z_5}{\partial W_{53}}

=(y_5-y_{out})*f(Z_2)*(1-f(Z_2))*y_3

=0.02763$$

同理有:

$$\dfrac{\partial C}{\partial W_{54}}=\dfrac{\partial C}{\partial y_5}*\dfrac{\partial y_5}{\partial Z_5}*\dfrac{\partial Z_5}{\partial W_{54}}

=(y_5-y_{out})*f(Z_2)*(1-f(Z_2))*y_4

=0.02711$$

   

至此我们已经求出最后一层的参数偏导了，继续向前链式推导：

我们现在还需要求出： $W_{31},W_{32},W_{41},W_{42}$ 

已知：
$$\begin{cases}
  Z_3=W_{31}*x_1+W_{32}*x_2 \\
  y_3=f(Z_3) \\
  Z_5=W_{53}*y_3+W_{54}*y_4 \\
  y_5=f(Z_5) \\
  C=\dfrac{1}{2}(y_5-y_{out})^2
  \end{cases}$$

则：

$$\dfrac{\partial C}{\partial W_{31}}=\dfrac{\partial C}{\partial y_5}*\dfrac{\partial y_5}{\partial Z_5}*\dfrac{\partial Z_5}{\partial y_3}*\dfrac{\partial y_3}{\partial Z_3}*\dfrac{\partial Z_3}{\partial Z_{31}}

=(y_5-y_{out})*f(Z_5)*(1-f(Z_5))*W_{53}*f(Z_3)*(1-f(Z_3))*x_1$$
   
上面结果都是已知值，同理可得其他几个式子。

则最终的结果为：

$$\begin{cases}
  W_{31}=W_{31}-\dfrac{\partial C}{\partial W_{31}}=0.09661944 \\
  W_{32}=W_{32}-\dfrac{\partial C}{\partial W_{32}}=0.78985831 \\
  W_{41}=W_{41}-\dfrac{\partial C}{\partial W_{41}}=0.39661944 \\
  W_{42}=W_{42}-\dfrac{\partial C}{\partial W_{42}}=0.58985831 
  \end{cases}$$
  
  再按照这个权重参数进行一遍正向传播得出的Error为0.0165，这个值比原来的0.01805要小，则继续迭代，不断修正权值，使得代价函数越来越小，预测值也不段逼近0.5，直到最后得到无限逼近真实值对应的权值。
  
#### summary
从以上反向传播的推导过程可以看出，计算代价函数对最后一层（输出层）的偏导，就为计算倒数第二层的偏导提供了条件，有了倒数第二层的偏导，又可以求出倒数第三层...一直重复这些步骤，就可以将所有层的偏导都计算出来。到这里，想必你就更加清楚什么叫反向传播了吧，其实就是从输出层一层层求偏导。

其实，对每层进行求偏导，按照多元函数的链式法则操作，如果一上来就求输入层的偏导数，那么就要先求出第2层的偏导数，而第2层的偏导数，要用第3层的偏导数去表达...最后发现最关键的是要求出最后一层（输出层）的偏导数，所以尼，只好从输出层开始求，一步步向输入层方向求偏导。

最后多说一句哈，大家可能感觉到很多知识和概念都是如此莫名，也就是说科学知识的体系和人类的认知规律经常不一致。比如说在物理学上，时间的定义先于速度，而我们却是先体会到物体运动的快慢，后面才产生时间的概念，皮亚杰的发生认识论就是讲这个问题的，大家可以去看看这位认知方面的祖师爷理论，你应该会更加体会到，神经网络是多么接近人类的认知模式。

***
#### more information
[How the backpropagation algorithm works](https://tigerneil.gitbooks.io/neural-networks-and-deep-learning-zh/content/chapter2.html)

[Principles of training multi-layer neural network using backpropagation](http://galaxy.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html)