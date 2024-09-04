---
cover: "../images/gru/cover.png"
title: 门控循环单元(GRU)
date: 2024-09-03 22:53:25
tags: [深度学习, 网络, deep learning]
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

The hidden unit managed to drop information that is irrelevant

## Usage

> SMT: statistical machine learning

- machine translation
- Scoring Phrase Pairs with RNN Encoder–Decoder

## casual notes

- the difference between rnn encoder decoder and [Zou's work](https://aclanthology.org/D13-1141.pdf) is that, rnn ignores the order of the words.

## experiment

- evaluate on WMT'14
- dataset includes: Europarl, news commentary, UN, two crawled corpora of 90M and 780M words.
- All the word counts refer to French words after tokenization
- one should focus on the most relevant subset of the data for a given task.
- selected a subset of 418M words out of more than 2G words for language modeling and a subset of 348M out of 850M words for training the RNN Encoder–Decoder
- test set newstest2012 and 2013 for data selection and weight tuning with MERT, andnewstest2014 as our test set.
- The baseline phrase-based SMT system was built using Moses with default settings.
- All the weight parameters in the RNN Encoder– Decoder were initialized by sampling from an isotropic zero-mean (white) Gaussian distribution with its standard deviation fixed to 0.01, except for the recurrent weight parameters.
- We used Adadelta and stochastic gradient descent to train the RNN Encoder–Decoder with hyperparameters $\sigma = 10^{−6}$ and $\rho = 0.95$.
- All the weight parameters were initialized uniformly between −0.01 and 0.01, and the model was trained until the validation perplexity did not improve for 10 epochs.

## result

![result](images/gru/result.jpg)
