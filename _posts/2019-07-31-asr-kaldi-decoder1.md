---
layout: post
title:  "Kaldi中的decoder(一）- 基础和viterbi解码"
date:   2019-07-31 11:11:59 +0800
categories: kaldi
---

*[Kaldi文档][kaldi-decoder-url]里的内容非常有价值，本文作为辅助理解的材料。 [Kaldi代码][kaldi-simple-decoder]实现也非常简单清晰。*

需要掌握的代码文件
* decoder/simple-decoder.cc
* decoder/simple-decoder.h
* gmmbin/gmm-decode-simple.cc
* itf/decodable-itf.h

Kaldi的decoder的实现在decoder/目录下，根据下面功能需要实现了不同的类型的解码器。
* 是否nbest(simple or lattice) 
* CPU还是GPU 
* 普通版还是加速版(faster)
* 离线批处理还是online。

各个不同的解码器对应的文件都以decoder结尾。
* simple-decoder
* faster-decoder
* lattice-decoder
* lattice-faster-decoder

## 解码器对象的设计

Kaldi中抽象出DecodableInterface的概念，将解码器算法和声学模型计算模块解耦合，解码器算法和声学模型类型无关，从而可以被不同类型声学模型复用。

### DecodableInterface接口
* LogLikelihood
* IsLastFrame
* NumFramesReady
* NumIndices

Decoder对象上只绑定了一个解码图Fst对象(HCLG).DecodableInterface对象在调用Decode时传入。

DecodableInterface用于计算声学模型的输出，对于Decoder相当于一个Data Provider，其主要接口为LogLikelihood，该接口返回t时刻，transition-id对应的声学模型得分。声学模型的输出是特征在上下文相关音素绑定后的状态（pdf-id）上的概率值，但在kaldi的hclg解码图FST里，边上输入是transition-id，这个数据结构包含了比pdf-id更多的信息。在LogLikelihood内部会把需要计算的transition-id转为pdf-id，然后计算pdf-id对应的得分。

Decoder

在kaldi中，DecodableInterface的输出一般是各transition-id的声学模型后验概率值，transition-id是HCLG的输入，Decoder要根据transition-id的得分选择一条（或者多条）最优路径。

Kaldi中没有为Decoder设计基类接口，但是每个Decoder在实现时都遵循了相同的接口约定。


## 解码可执行程序

根据使用的DecodableInterface，也就是声学模型(GMM / NNet)，最终的解码可执行程序在 gmmbin和nnet3bin目录下面
* 1-best的解码叫 \*-decode-\*
* lattice解码叫 \*-latgen-\*

解码可执行程序中主要做的事情是
* 读取FST解码图模型
* 非online模式读取特征feature，online模式则是配置特征提取模块FeaturePipeline
* 配置DecodableInterface(feature)或者DecodableInterface(featurepipeline)
* 配置Decoder(fst)
* 调用decoder.Decode(decodable)开始解码

可以阅读gmm-decode-simple学习怎么使用一个decoder.

## 解码器算法
解码器的核心算法在decoder/下面的文件里。本文简单介绍最简单的SimpleDecoder，其仅在gmm解码里使用(gmm-decode-simple)，nnet3并没有使用这个decoder. 

SimpleDecoder里实现的token-pass算法是viterbi算法的一种实现方法，这个是其他解码算法的基础，必须要完全理解，深入掌握，代码非常简洁，可以多看几遍。

### Token 

* token可以理解为在state上
* **prev_** 指向前一个token（从哪个token扩展来的），引用计数
* **cost_** 累计cost，t时刻，token A经过边 `arc(trans-id, word-id, graph_weight)` 扩展为token B。 B的cost等于 `A的cost + (-am_score) + graph_weight`。其中am_score为t时刻特征`Ft`的声学模型上对应的`pdf-id`(`trans-id`对应的`pdf-id`)的得分。
* **arc_** token构建时传入的是grpah上对应的StdArc类型的arc，而token内部实际存储的arc类型是LatticeArc，hclg用的StdArc类型的arc里weight只能存一个值，而LatticeArc里weight把acoustic和graph cost分开存储，因此在回溯时，可以分别输出acoustic得分和lm得分.
* **TokenDelete()** 利用用引用计数的方式对token进行释放管理。
* 重载了 **<** 符号，比较两个token的cost大小, 若 a.cost > b.cost, 则a < b.

### SimpleDecoder类
* prev_toks_ 保存前一帧（t-1时刻）的所有tokens
* cur_toks_ 保存当前帧（t时刻）扩展出的tokens
* prev_toks_和cur_toks_是从stateID到token的map类型对象，类型为unordered_map<StateId, Token*>， 因为只需要1-best路径，每个状态上最多只有1个token
* 对于t-1之前时刻的tokens，不专门维护，他们只在到达终止时刻T时，cur里的token做backtrace时用到。
* beam_ 剪枝阈值
* num_frames_decoded_ 当前完成解码(token pass)的帧数


### Decode()

算法非常简单

**ProcessEmitting**

每一时刻t，对prev_toks_(t-1时刻)里每个状态上的token（SimpleDecoder只找最优路径，每个state只要保留一个token即可，这就是viterbi算法。若想保留全局的Nbest路径，每个状态上需要记录N个cost最小的t时刻token），根据该状态的arc进行转移。若arc的输入不是epsilon，则创建一个新的token，放进cur_toks_(t时刻)里。若新的token对应的state上在t时刻已经有了token，则比较两者的cost，保留cost小的.

因为解码图包含大量state，如果每个时刻都保留所有扩展出的tokens，想象一下我们的HCLG有那么多state，每个state上0到T时刻的token都要保存，带来的计算和存储都非常大。所以需要剪枝删除一些token，一个简单的方法为：在每一个时刻t，记录t时刻扩展出的新token中的最小的cost，在扩展t时刻token时，只有保留cost和该最小cost差值小于某个阈值(beam)的新token。这种根据阈值进行剪枝的搜索方法叫beam search，其可以减少计算和存储开销，但可能丢失全局cost最小的那条路径。


**ProcessNonemitting**

对于输入是epsilon的边，跳转是不消耗时间的，因此需要单独处理。 在ProcessEmitting完成后，若cur_toks_中的token所在状态存在输入为eps边，则将这些token继续扩展并放入cur_toks_里。(所以自跳转上的边的输入一定不是epsilon，不然这一步无法结束。)代码中，fst里保留0作为epsilon的index，所以代码中根据arc.ilabel是否为0判断输入是否为epsilon。

在ProcessEmitting中，每一个时间点，每个token只能沿着输入非epsilon的arc扩展一步，而对于ProcessNonemitting，如果某个state后面有连续多个epsilon的arc，是需要向后连续扩展token的，因此在代码实现上用了一个队列来进行**深度优先搜索**。

ProcessNonemitting中的剪枝和ProcessEmitting有些区别，方法为算出t-1时刻（ProcessEmitting中是t时刻)的token中的最小cost，在扩展时t时刻token时，只有保留cost和该最小cost差值小于某个阈值(beam)的新token。

**PruneToks**

在t时刻全部扩展完成后，对所有的新token再扫描一遍，只保留那些cost和最小cost差值小于某个阈值(beam)的token。将prune后不包含token的state从map中去除。

**GetBestPath**

当到达T时刻，只要找出cur_toks_里cost最小的state(如果要求final，则必须使用final状态中的token)，然后根据其backtrace指针不断往前回溯找到所有的token，把这些token记录的arc上的output输出即可。GetBestPath里实际是用token里的arc信息构建了一个Lattice，不过这个Lattice其实是一条直线。每个状态都只有一个边。由于有输入是epsilon的边，所以这条路径的边个数可能大于语音的帧数。

以上就是全部算法。谢谢阅读。


[kaldi-decoder-url]: http://kaldi-asr.org/doc/decoders.html
[kaldi-simple-decoder]: https://github.com/kaldi-asr/kaldi/src/decoder/simple-decoder.cc