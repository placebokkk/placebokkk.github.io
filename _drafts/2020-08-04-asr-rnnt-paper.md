---
layout: post
title:  "端到端ASR:RNN-T相关paper"
date:   2020-08-10 10:00:00 +0800
categories: asr
---

## 模型简述：

RNN-T的提出是想解决CTC的两个问题：
1. 输出序列不能比输出序列长
2. 输出之间是条件独立的。

因此RNN-T中将输出序列和输入序列的建模拆成了两个网络，这点和后面广为流行的AED是一致的，而两者区别在于如何建模Encoder和Decoder(原始Rnn-t叫Predict)之间的对齐
1. 在Rnn-T中，构建两个序列的对齐lattice（类似与HMM模型），通过前向后向算法计算梯度，要求两个序列是monotonic的。
2. 在AED中，通过attention机制对齐，不要求两个序列是monotonic的。

这也是RNN-T和AED模型的本质区别。

原始的RNN-Transducer中的网络结构使用的RNN(LSTM),也可以使用其他网络结构，如1dConv、2dConv和Self-Attention，
一般若在对其上使用的是RNN-T的方式，模型叫就做*-Transuduer。


## RNN-T的研究问题
1. 预训练问题, 尤其是语言模型的预训练问题。
2. 计算优化
3. 网络结构
4. 建模单元

Attention机制和monotonic alignment Lattice机制可以同时使用。在Rnn-T框架下加入一个attention.效果？

RNN-T和CTC是frame-sync decode
attention是label-sync decode

## 论文
### 2020

**SYNCHRONOUS TRANSFORMERS FOR END-；TO-END SPEECH RECOGNITION**
https://arxiv.org/pdf/1912.02958.pdf

**Attention-based Transducer for Online Speech Recognition**
和论文SYNCHRONOUS TRANSFORMERS几乎一样，但角度不同：
本文从RNN-T出发，想要优化RNN-T的计算和label不平衡问题，引入了chunk。
从Transfomer出发，为了处理chunk的对齐问题，引入了RNN-T的前向后向算法来计算目标函数。

#### 提出问题：
1. RNN-T的输出序列长度是T(T个blank)+U，其中T >> U，因此label大量都是blank，导致输出时倾向于输出blank。deletion错误多。
2. 计算复杂度问题 joint net的输出是T*U*(Y+1).batch只用了AED的四分之一长20000frames(也就是) 
3. joint net中没用到只用了encoder的当前output，而没用前后时间的output.

#### 解决方法：
模型结构：
* Encoder = pLSTMs(subsampling提高训练和解码速度) +  LSTMs + windowed multi-head self-attention(左右相等上下文长度)
* Decoder = 2 LSTMs, scheduled sampling
* Joint = Encoder切分成chunk，Decoder和chunk做multi-head attention

对问题1，2:
* 通过pLSTMs的subsampling和chunk化，Encoder的输出长度为T/uw，因此对于问题1和2。降低了blank和non-blank的比值，减小了grid矩阵大小。

对问题3:
* joint net计算对encoder的每个chunk内做attention，用一组上下文来计算单个对齐点的分布。


#### 实验结果：
流利说中国人说英文的数据。
500 units wordpiece by BPE

Encoder = 3 pLSTMS(1024) + 2 LSTMs(1024) + 4-heads self-attention
Decoder = 2 LSTMs(512)
Joint = 4-heads attention

对比不同r(Encoder中self attention的上下文长度)和w(Joint中chunk大小)影响. r=2,w=4效果最好。

内部500小时测试集上，
* 超过了CER attention transducer(15.98%) < baseline RNN-T(17.90%) < LAS(18.67%)
* deletion error降低了30%。

内部10000小时数据上，和production(Kaldi 17 TDNN-F,LFMMI)对比
* CER更低
* RTF差一些
* Latency更优(右侧上下文少)

chunk-wise的视角可以统一AED和RNN-T
1. 当chunk大小等于encoder长度时，等价于AED。
2. 当chunk大小等于1时，等价于原始的RNN-T


*A New Training Pipeline for an Improved Neural Transducer**


**EXPLORING PRE-TRAINING WITH ALIGNMENTS FOR RNN TRANSDUCER BASED END-TO-END SPEECH RECOGNITION**
* pretrain encoder
在encoder部分增加一个全联接+softmax， 用force alignment的结果做
用的word alignment（可能是微软内部的工具），然后等分word内的word piece的时间戳。

* pretrain encoder+decoder
对U*T矩阵，用alignment进行U*T矩阵大小的扩展。
三种方式：
1. 每个u对应的t序列都用alignent label
2. 按alignment的填充label
3. 去除掉blank，减少模型对blank的bias

效果3 > 1 > 2
感觉这篇论文

疑问：
说CTC大部分是blank，所以encoder学到大部分都是blank，效果不好，但是Rnn-T不也是大部分是blank吗？

结论：
encoder的CE pre-train最好。比CTC pre-train要好。


**HYBRID AUTOREGRESSIVE TRANSDUCER (HAT)**
如何去除Decoder中的语言模型得分部分，得到am的得分。
再用一个外部的lm。

通过将local score变为。一个伯努利分布+一个softmax，从而将blank从softmax中分离，
softmax可以用来计算序列的语言模型得分。

f1网络算伯努利和f2网络算softmax。
涉及到laplace估计？？这个推导？arxiv的submission？


**RNN-TRANSDUCER WITH STATELESS PREDICTION NETWORK**
1. 用1%数据训练得到baseline，用全量数据训练不同部分，对比提升效果。
* 只训PN+JN效果也会提升 （32->24）
* 只训enocoder(32->17)比只训PN+JN收益最大。
* 全部都训效果最好。(32->15 )

2. PN用一个前向网络（bi-gram） RNNT-SLP(StateLess Prediction)
grapheme上下降多，主要是连续字符错误 food foood fooood， 没给出具体WER。
论文认为：
`an important job of the prediction network is that it stops the model from outputting repetitive symbols many times`
在word-piece上下降少。

实际2000小时的挪威上更好(22.2 -> 19.9)，11000hr的西班牙上9.6->9.7 在30000hr的英文上下降明显。
 6.8->7.2

论文认为
`the total number of the parameters is more important than whether the recurrent layers are in the encoder or the prediction network.`
即模型总参数比pn中的recurrent层更重要，如果encoder增加2层，可以抵消掉pn中少的recurrent，然而该setting的实验结果是6.8->7.1
本文有点睁眼说瞎话的感觉。 

比不带语言模型解码的CTC要好 (13 vs 20.3)

**RNN-T FOR LATENCY CONTROLLED ASR WITH IMPROVED BEAM SEARCH**



**Research on Modeling Units of Transformer Transducer for Mandarin Speech
Recognition**

https://arxiv.org/pdf/2004.13522.pdf

京东论文，介绍了京东RNN-T的工作。

mix-bandwith将8K和16K混用

网络结构: 
Encoder=CNN + self-attention
Decoder=LSTM

**数据集**

Internal 
JD dataset1 8kHz ~5000h
JD dataset2 16kHz ~5000h
public
AISHELL-1 16kHz 178h
AISHELL-2 16kHz 1000h
THCHS-30 16kHz 30h
Primewords [24] 16kHz 100h
ST-CMDS [25] 16kHz 122h
Total: ~12000h

**建模单元**

Modeling Units  Unit Number Maximum Length
syllable initial/finalwith tone 227 119
syllable with tone 1256 40
Chinese character 7228 40



#### 结论
CER: syllable with tone < syllable initial/final with tone < character
CER没说清楚syllable是怎么算的？是把label转为syllable计算，还是把syllable转为word（需要一个网络或者其他模型）去计算？？
2019
**TRANSFORMER-TRANSDUCER: END-TO-END SPEECH RECOGNITION WITH SELF-ATTENTION**
Self-attention

2018

**EXPLORING ARCHITECTURES, DATA AND UNITS FOR STREAMING END-TO-END SPEECH RECOGNITIONWITH RNN-TRANSDUCER**





2. 性能改进


时间戳

建模单元问题：

中文

字
syllabel
syllabel+tone
phone
phone+tone
BPE on phone
BPE on syllabel
BPE on phone+tone
BPE on syllabel+tone


Acoustic Encoder部分用CTC预训练
Acoustic Encoder部分用CTC 做mult-task训练。
