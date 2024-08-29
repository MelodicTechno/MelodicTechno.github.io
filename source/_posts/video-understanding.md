---
title: 视频理解
date: 2024-08-28 22:19:24
tags: [deep learning, action recognition]
categories: [课程笔记, 论文笔记]
mathjax: true
---

## 时间线

> - 综述: A Comprehensive Study of Deep Video Action Recognition
    - [link](https://arxiv.org/pdf/2012.06567)
    - [pdf](../../../../pdf/video1/2012.06567v1.pdf)

时间线如下：

![时间线](images/video1/timeline1.jpg)

从Deep Video，纯粹的深度学习，到使用特征提取的双流网络（用光流教网络学习），之后出现了3D网络，最后来到流行Video Transformer的现在(2024年)。

## 主要内容

1. 手工 -> CNN
2. 双流网络
3. 3维卷积
4. video transformer

## CNN

> - A. Karpathy, G. Toderici, S. Shetty, T. Leung, R. Sukthankar and L. Fei-Fei, "Large-Scale Video Classification with Convolutional Neural Networks," 2014 IEEE Conference on Computer Vision and Pattern Recognition, Columbus, OH, USA, 2014, pp. 1725-1732, doi: 10.1109/CVPR.2014.223.
>   - [link](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42455.pdf) 

### fusion

![fusion](images/video1/fuse.jpg)



