---
layout: post
title:  "语音识别中的LM rescoring"
date:   2019-04-21 10:00:00 +0800
categories: ASR
---
语音识别静态解码中，一般使用语言模型不会太大

从候选集上分为N-best rescoring和lattice rescoring. lattice 本质上也是一种N-best的更紧凑的数据结构表示，lattice rescoring相比N-best rescoring，可以在同样的时空复杂度下处理更多的候选序列。

从语言模型上分为back-off ngram(bong)和nnlm两种。

从而至少有四种rescoring方法。

1. N-best + bong
对于N-best中的每个结果，已知LM和AM的得分，使用bong重新计算该序列的LM得分
score = w1\*bong-lm-score + w2 \* lm-score + w3 \* am-score
使用该得分对结果进行重新排序
可以使用很大的bong

2. N-best + nnlm
和1中方法一样，只是换为使用nnlm计算lm得分
score = w1\*nn-lm-score + w2 \* lm-score + w3 \* am-score
使用该得分对结果进行重新排序
nnlm的ppl一般比较好，但是nnlm的计算开销大

3. Lattice + bong
使用lattice rescoring，分为

lattice rescoring
假设first-pass中的语言模型为G,用于rescoring的语言模型为R,静态构建好一个fst G'= G^-1 R
Lattice 为A
对A和G'执行compose操作即可得到rescoring后的新的lattice。G'中的G^-1用于移除原始语言模型得分，R用于新增语言模型得分。

A和G'的compose操作比较费时？所以用on-the-fly的方法？

问问的rescoring方法？



4. Lattice + nnlm
思路1，从nnlm生成ngram语言模型,从而转变为方法3.论文?google？
