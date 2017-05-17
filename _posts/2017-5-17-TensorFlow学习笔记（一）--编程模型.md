<html lang="en"><head>
    <meta charset="UTF-8">
    <title></title>
<style id="system" type="text/css">h1,h2,h3,h4,h5,h6,p,blockquote {    margin: 0;    padding: 0;}body {    font-family: "Helvetica Neue", Helvetica, "Hiragino Sans GB", Arial, sans-serif;    font-size: 13px;    line-height: 18px;    color: #737373;    margin: 10px 13px 10px 13px;}a {    color: #0069d6;}a:hover {    color: #0050a3;    text-decoration: none;}a img {    border: none;}p {    margin-bottom: 9px;}h1,h2,h3,h4,h5,h6 {    color: #404040;    line-height: 36px;}h1 {    margin-bottom: 18px;    font-size: 30px;}h2 {    font-size: 24px;}h3 {    font-size: 18px;}h4 {    font-size: 16px;}h5 {    font-size: 14px;}h6 {    font-size: 13px;}hr {    margin: 0 0 19px;    border: 0;    border-bottom: 1px solid #ccc;}blockquote {    padding: 13px 13px 21px 15px;    margin-bottom: 18px;    font-family:georgia,serif;    font-style: italic;}blockquote:before {    content:"C";    font-size:40px;    margin-left:-10px;    font-family:georgia,serif;    color:#eee;}blockquote p {    font-size: 14px;    font-weight: 300;    line-height: 18px;    margin-bottom: 0;    font-style: italic;}code, pre {    font-family: Monaco, Andale Mono, Courier New, monospace;}code {    background-color: #fee9cc;    color: rgba(0, 0, 0, 0.75);    padding: 1px 3px;    font-size: 12px;    -webkit-border-radius: 3px;    -moz-border-radius: 3px;    border-radius: 3px;}pre {    display: block;    padding: 14px;    margin: 0 0 18px;    line-height: 16px;    font-size: 11px;    border: 1px solid #d9d9d9;    white-space: pre-wrap;    word-wrap: break-word;}pre code {    background-color: #fff;    color:#737373;    font-size: 11px;    padding: 0;}@media screen and (min-width: 768px) {    body {        width: 748px;        margin:10px auto;    }}</style><style id="custom" type="text/css"></style></head>
<body marginheight="0"><hr>
<p>layout: post
title: "TensorFlow学习笔记（一）--编程模型"
date: 
</p>
<h2>categories: [python]</h2>
<p>TensorFlow中涉及到三种模型，计算模型、数据模型和运行模型，通过这三个角度对TensorFlow开始学习，可以对其工作原理有一个整体的理解。

</p>
<h2>TensorFlow计算模型——计算图</h2>
<p>计算图是TensorFlow中最基本的一个概念，TensorFlow的所有计算都会被转化为计算图上的节点
</p>
<h3>计算图的概念</h3>
<p>其实TensorFlow的名字已经说明了最重要的两个概念——tensor和flow，tensor就是张量，可以被简单的理解为多维数组，也就是它的数据结构（后续介绍），而flow则体现了它的计算模型，flow意为“流”，很直观的表达了张量之间通过计算相互转化的过程。TensorFlow是一个通过计算图的方式来表述计算的编程系统，它的每一个计算都是计算图上的一个节点，而节点之间的边描述了计算之间的依赖关系。
</p>
<h3>计算图的使用</h3>
<p>TensorFlow程序一般可分为两个阶段，在第一个阶段需要定义计算图中所有的计算，第二个阶段为执行阶段（后续介绍），以下是计算定义阶段的示例：
</p>
<pre><code class="lang-python">import tensorflow as tf
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a + b</code></pre>
<p>在这个过程中，TensorFlow会自动将定义的计算转化为计算图上的节点。在TensorFlow中，系统会自动维护一个默认的计算图，通过tf.get_default_graph函数可以获取当前默认的计算图，当然TensorFlow也支持通过tf.Graph函数来生成新的计算图，但不同计算图的上的张量和运算都不会共享。

</p>
<p>TensorFlow中的计算图不仅可以用来隔离张量和计算，还提供了管理张量和计算的机制，计算图可以通过tf.Graph.device函数来指定运行计算的设备，这也为TensorFlow使用GPU提供了机制，以下程序指定计算跑在GPU上
</p>
<pre><code class="lang-python">g = tf.Graph()
with g.device('/gpu:0')
    result = a + b</code></pre>
<p>Edit By <a href="http://mahua.jser.me">MaHua</a></p>
</body></html>