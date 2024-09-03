---
cover: "../images/gru/cover.png"
title: 门控循环单元(GRU)
date: 2024-09-03 22:53:25
tags: [深度学习, 网络]
categories: [论文笔记]
mathjax: true
excerpt: "对GRU的简单记录"
---

## 单元

**RNN Encoder & Decoder**

This is the rnn encoder & decoder.
![cell1](images/gru/rnn_de.png)

the **hidden** unit
![cell1](images/gru/hidden_unit.png)

z: update gate, selects whether the hidden state would be updated by $\hat h$; r: reset gate, decides whether the previous hidden state would be ignored

The hidden unit managed to drop information that is irrelavent