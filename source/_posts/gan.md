---
cover: "../images/gan/cover2.jpg"
title: GAN(Generative Adversarial Networks)
date: 2024-09-05 15:07:30
tags: [基础, deep learning]
categories: [论文笔记]
excerpt: "GAN! 干！"
mathjax: true
---

> paper: https://arxiv.org/pdf/1406.2661
> cover: いずもねる
> I'm so sad :<

## Adversarial nets

$p_z(z)$ was defined to learn $p_g$ over data $x$, then presenting a mapping to $G(z;\theta_g)$; Second mlp $D(x;\theta_d)$ outputs a single scalar. $D(x)$ means the probability that $x$ came from the data rather than $p_g$. The training, or the game between $G$ and $D$ is:
![gan](images/gan/gan.jpg)

## Train

In the earlier training, the objective function is to maximize $logD(G(z))$, the result is the same but the gradients are stronger early in learning.

## The algorithm

In the paper, the author explained the algorithm to optimize Eq1, but it is a bit deep for me. I may watch some videos explaining this later.

## Experiment

Dataset: MNIST, Toronto Face Database and CIFAR-10.

The generator nets used a mixture of rectifier linear activations and sigmoid activations.
The discriminator net used maxout activations.
Dropout was applied in training the discriminator net.
Estimate the probability with Gaussian Parzen window.