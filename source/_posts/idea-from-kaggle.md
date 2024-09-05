---
cover: "../images/idea-from-kaggle/cover.jpg"
title: 来自Kaggle的灵感
date: 2024-09-05 23:13:53
tags: ["deep learning", "computer vision", "kaggle"]
categories: [深度学习实战]
excerpt: "在接触Kaggle的竞赛的过程中发现的一些方法"
---

## 数据预处理

对于一些尺寸很大，细节很多的图像，比如医学影像，可以切割成多个小块。

## 实验方法

使用C语言代码加快对图像数据处理的速率，C代码可以直接加到notebook里。
使用k折交叉验证估计模型性能

Two strategies were implemented to mitigate class imbalance:
Weighted sampling to create balanced batches for training,
Using class weights in CE loss.

## 编码技巧

重复出现的块：
~~~python
chowder_models = [Chowder(**chowder_kwargs) for _ in range(50)]

model = ModelEnsemble(chowder_models)
~~~