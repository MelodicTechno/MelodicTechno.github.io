---
cover: "../images/attention/cover.jpg"
title: Attention Notes
date: 2024-09-04 22:54:30
tags: [基础, deep learning]
categories: [论文笔记]
excerpt: "A brief note of Attention mercanism"
mathjax: true
---

> paper: https://arxiv.org/pdf/1706.03762
> cover: from anime "Code Geass Lelouch of the Rebellion"

## Before the paper

Basically, Attention managed to model the attention of human eyes. It turn an input into vectors, representing the features in a higher dimension.

## Attention

The operation of Attention can be depicted as the following equation:

$$
Attention(Q, K, V) = sofmax(\frac{QK^T}{\sqrt{d_k}})V
$$

Well, it is operation over the vector, Q, k, and V. Q is the query, K is the key and V are the values. Q dot-product K, the output was divided to scale its value, and was feeded into sofmax for regularization, following the multiplication with V. The overall output is the possibility of the reference of the attention, a weighted sum.

## Model

![model](images/attention/model.png)
Transformer

It has an encoder-decoder structure, with the encoder mapping symbols $(x_1, \dots x_n)$ to representations $z = (z_1, \dots, z_n)$. The decoder accepts $z$ and generates output $(y_1, \dots, y_n)$.

### Encoder and Decoder

**Encoder**: 
stack of 6 identical layers with 2 sublayers;
layer1: multi-head self-attention;
layer2: simple, position-wised fc;
each 2 layers conneced with residual connection;
output dimension: $d_{model} = 512$

**Decoder**:
stack of 6 identical layers with 2 sublayers, with a third sub-layer inserted, performing multi-head attention;
with residual connection, followed by layer normalization

### Attention 

![attention](images/attention/attention.png)

Multi-Head Attention is the parallelization of Attenion

The two most commonly used attention functions are additive attention and dot-product attention. The latter is more efficient int both the speed and space.

### Multi-Head Attention

Linearly project the queries, keys and values $h$ times with different, learned linear projections to $d_k$, $d_k$ and $d_v$ dimensions. They was operated in parallel, yielding $d_v$ dimensional output values. The outputs are concatenated and projected, resulting in the final values.

The operation can be defined as:

$$
MultiHead(Q, K, V) = Concat(head_1, \dots, head_h)W^O where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
$$

### Position-wise Feed-Forward Networks

identical fc applied to each position, with ReLU in between:

$$
FFN(x) = max(０，xW_1+b_1)W_2 + b_2
$$