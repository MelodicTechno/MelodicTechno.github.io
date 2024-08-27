---
title: 动作检测综述笔记
date: 2024-08-26 21:26:51
tags: ["deep learning", "action recognition"]
mathjax: true
---

> 论文：Going deeper into action recognition: A survey
## 方法

---
## 描述动作的方法

## Local representation based approaches

1. Interest point detection
2. Local descriptors
    1. Edge and motion descriptors
    2. Pixel pattern descriptors
    3. From cuboids to trajectories
3. Aggregation
   上述内容较基础且年代久远，暂不详细整理。

# 用于动作检测的深度学习网络架构

---

## 四种架构

1. Spatiotemporal networks (时空网络)
2. Multiple stream networks (多流网络)
3. Deep generative networks (深度生成网络)
4. Temporal coherency networks (时间相干网络)
   以上中文翻译来自谷歌翻译，我认为中文名并不重要，不予深究。分类则应该是论文作者个人的观点。

## Spatiotemporal networks

## 零碎的记录：

1. pooling和weight-sharing用于减少网络搜索的空间；
2. 三维卷积在卷积的基础上增加了时序信息，使用三维的卷积核。三维卷积神经网络输入的视频的帧数是预先确定的；
3. 在将时序信息输入(fusion)卷积网络的方法中，最大池化表现很好(吴恩达)；
4. slow fusion可以增强神经网络对时序的认知；在slow fusion中，相同的几个层接收几个连续的视频片段，输出再输入到全连接层，由此得以描述视频；
5. 其他的fusion方法：early fusion: 逐帧特征加入到最后一层；Karpathy提出的方法：使用两个网络，能够增加精确度，同时减少需要学习的参数，因为每支网络能接受较小的输入，![示例](/images/har1/ie1.png)在这个示例中，fovea stream能注意到视频中央的区域，利用了摄像机的偏差，即兴趣点大多出现在视频中央；
6. Tran等人的工作: 只使用$3\times3$ 的卷积核效果更好；
7. 增加输入的时间的长度，同时结合使用具有不同对时间的意识的网络，能够提高神经网络的表现；
8. 结合使用2D和1D的卷积核能减少3D卷积核对参数数量的需求；
9. Baccouche与Donahue等人：一系列卷积神经网络+LSTM，充分利用了时间信息；为了检测动作，Baccouche等人建议将三维卷积网络提取的特征输入到LSTM中；
10. Donahue等人：Long-term Recurrent Convolutional Network (LRCN)![lrcn](images/har1/ie2.png)

---

## Multiple stream networks

## 1. Simonyan与Zisserman的双流网络

结构如下：

![双流网络](images/har1/two_stream_network.png)
这是两个并行的网络。

- 使用预训练的模型
- 输入时堆叠时序信息
- 有多个classification layer，每个在不同的训练集上训练，这是一种多任务学习
  双流网络使用softmax将两个流连接起来，在中间层融合可以表现得更好，同时减少需要学习的参数；在卷积层后融合可以减少对两个流的全连接层的需求；这个网络还可以进一步拓展：使用Fisher Vector，增加第三条支流来增加音频信号。双流网络中，播放的帧是唯一与动作相关的输入，这使双流网络无法捕获持续时间长的微小动作，将网络与手动提示结合起来可以改善这个问题。
 
---

## Deep generative models

几种模型如下：
1. Dynencoder
2. LSTM autoencoder model
3. Adversarial models
4. Temporal coherency networks

### Dynencoder

最基础的版本包含三层，第一层将输入$x_t$映射到隐藏$h_t$，第二层是预测层，基于当前的 $h_t$ 预测 $\tilde {h} _ {t+1}$ ，第三层使用预测的 $\tilde {h} _ {t+1}$ 生成预测的帧 $\tilde {x} _ {t+1}$ 。在合成动态纹理方面效果不错，可以理解成一种再现视频信息的简洁方法。

## LSTM Dyencoder

构造如下：![lstm auto encoder](images/har1/lstmaotoencoder.png)

## Adversarial models

对抗网络

## Temporal coherency networks

一种弱监督学习的方法，用元组训练，判断动作是否连续。以Siamese Network为例：
![siames network](images/har1/siames.png)注意对时间上的连续性不一定意味着可靠性，比如插播广告时也是连续的，但显然广告与正片没有相关性。

Wang等人的工作：将动作划分为两个阶段来识别，将动作划分为前提（precondiction）和效果（effect），使用Siamese Network，构造如下：![two phase](images/har1/twophasecoherency.png)
Rank pooling可以用来捕捉动作序列中的时序变化。

---

本篇综述剩下的内容是对与不同网络表现的数值分析，上图：

![](images/har1/performance.png)
![](images/har1/performance2.png)
