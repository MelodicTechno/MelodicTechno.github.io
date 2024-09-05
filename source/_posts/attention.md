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

### Embedding and Softmax

Learned embeddings were used to convert the inputs to $d_{model}$ dimensional outputs.

### Positional Encoding

No recurrence and convolution is used. Positional encodings manage to make use of the order of the the sequence. The functions are:

$$
PE_{(pos, 2i)} = sin(pos / 10000^{2i / d_{model}})
$$

$$
PE_{(pos, 2i+1)} = cos(pos / 10000^{2i / d_{model}})
$$

## Training

- Data: WMT 2014 English-German dataset, WMT 2014 English-French dataset
- Sentences were encoded using byte-pair encoding
- Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.

## Hardware and Schedule

- Hardware: 8 NVIDIA P100 GPUs
- Time: 12 hours

## Optimizer

Adam optimizer with $\beta_1 = 0.9$, $\beta_2 = 0.98$ and $\sigma = 10^{−9}$

The learning rate changes over time:

$$
lrate = d_{model}^{-0.5} \dot min(step \_ num^{-0.5}, step \_ num \dot warmup \_ steps^{-1.5})
$$

$warmup \_ steps = 4000$

## Regularization

Three types of regularization:
1. Residual Dropout
2. Label Smoothing

~~Where is the third one?~~