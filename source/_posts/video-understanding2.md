---
cover: "../images/video1/timeline1.jpg"
title: 视频理解·续
date: 2024-08-30 11:36:18
tags: [deep learning, action recognition]
categories: [课程笔记, 论文笔记]
mathjax: true
excerpt: "整理李沐账号的视频理解串讲"
---

<iframe src="//player.bilibili.com/player.html?isOutside=true&aid=256025019&bvid=BV11Y411P7ep&cid=586721445&p=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"></iframe>

> [【视频理解论文串讲（下）【论文精读】】](https://www.bilibili.com/video/BV11Y411P7ep/?share_source=copy_web&vd_source=5c87dbd20cc0bcba1008ebe80abdab66)

---

## 主要内容

1. 手工 -> CNN ✔️
2. 双流网络 ✔️
3. 3维卷积
4. video transformer

> 抽取光流很慢，也很占空间。光流存储带来的IO很多。所以很多人想把这种方法换掉，使用3D CNN，希望能直接学视频。但实际上3D网络越做越大，结果也没解决性能问题。实际上光流还是好特征。

## 3D CNN

### C3D

> - Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." Proceedings of the IEEE international conference on computer vision. 2015.
>   - [link](https://arxiv.org/pdf/1412.0767)

**主要工作**: 提出简单且有效的学习视频的时空特征的方法。

结果：
![result](images/video2/result1.jpg)

模型的结构：
![model](images/video2/model1.jpg)

总共有11层，卷积核均为 $3*3*3$ 。这个网络是3D的VGG。
输入大小: $16 \times 112 \time 112$，视频帧，大小不重要
Conv2a: $16 \times 56 \times 56$
Conv3a: $8 \times 28 \times 28$
Conv4a: $4 \times 14 \times 14$
Conv4a: $2 \times 7 \times 7$
fc6: $1 \times 4096$

可以从fc6抽特征给SVM，做分类，效果更快更好。fc6抽的特征叫C3D特征

结果:
![result](images/video2/result2.jpg)

效果挺好，比2D好

![result](images/video2/result3.jpg)

上面两个效果不好，不如之前的手工特征，也不如双流。整体上效果还是不好。文章的抽特征好，提供了python的实现，可以调接口。

### I3D

> - Carreira, Joao, and Andrew Zisserman. "Quo vadis, action recognition? a new model and the kinetics dataset." proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.
>   - [link](https://arxiv.org/pdf/1705.07750)

模型:

![result](images/video2/model2.jpg)

i即膨胀。预训练时使用2d数据，通过膨胀将2d转为3d的数据。i2d与i3d结构不变，卷积核变。不用再设计网络，只要扩充成3d，还能用之前用2d训练的模型参数

结果:
![result](images/video2/result4.jpg)

意义：

1. 分别用了RGB和光流，效果很好
2. 简单方法提供表现
3. 证明2D网络迁移到3D网络的有效性

从此不用双流网络改用3D网络，数据集由UCF 101变成了K 400

ResNet -> ResNet3D
ResNext -> MFNet
SENet -> STCNet

### Non-local

> - Wang, Xiaolong, et al. "Non-local neural networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
>   - [link](https://arxiv.org/pdf/1711.07971) 

---

> 3D网络大概已经定型，需要改进：时序上怎么建模？lstm可用，注意力很好

卷积和递归都是在局部上操作，作者提出non-local算子，是一个可以建模长距离信息的模块。

non-local block:
![non-local](images/video2/non_local_block.jpg)

