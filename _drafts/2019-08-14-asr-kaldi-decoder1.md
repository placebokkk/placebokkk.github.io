---
layout: post
title:  "理解kaldi中的decoder(一）- 基础和viterbi解码"
date:   2019-07-30 11:11:59 +0800
categories: kaldi
---

*参考[Kaldi文档][kaldi-decoder-url]里的内容非常有价值，本文作为辅助理解的材料。 [Kaldi代码][kaldi-simple-decoder]实现也非常简介清晰。*

Kaldi的decoder的实现在decoder/目录下，根据下面功能需要实现了不同的类型的解码器。
* 是否nbest(simple or lattice) 
* CPU还是GPU 
* 普通版还是加速版(faster)
* 离线批处理还是online。

各个不同的解码器对应的文件都以decoder结尾。
* simple-decoder
* lattice-decoder
* faster-decoder
* lattice-faster-decoder

## 解码器对象的设计

Kaldi中抽象出DecodableInterface的概念，将解码器算法和声学模型计算模块解耦合，解码器算法和声学模型类型无关，从而可以被不同类型声学模型复用。

### DecodableInterface接口
* LogLikelihood
* IsLastFrame
* NumFramesReady
* NumIndices

Decoder对象绑定了一个DecodableInterface对象和一个解码图Fst对象(HCLG).

Decoder接受特征序列数据(T*D，T个时刻，每个时刻D维特征)，t时刻的特征序列数据经过DecodableInterface的输出值会作为t时刻Decoder的输入信息。

在kaldi中，DecodableInterface的输出一般是各transition-id的在声学模型上的后验概率值，transition-id是HCLG的输入，Decoder要根据transition-id的得分选择一条（或者多条）最优路径。

Kaldi中没有为Decoder设计基类接口，但是每个Decoder在实现时都遵循了相同的接口约定。


根据使用的DecodableInterface，也就是声学模型(GMM / NNet)，最终的解码可执行程序在 gmmbin和nnet3bin目录下面
* 1-best的解码叫 \*-decode-\*
* lattice解码叫 \*-latgen-\*

解码可执行程序中主要做的事情是
* 配置特征提取模块FeaturePipeline
* 加载FST
* 配置DecodableInterface
* 配置Decoder
* 调用Decoder开始解码

## 解码器算法
解码器的核心算法在decode/下面的文件里。本文简单介绍最简单的SimpleDecoder，其仅在gmm解码里使用(gmm-decode-simple)，nnet3并没有使用这个decoder.

SimpleDecoder里实现的token-pass算法是viterbi算法的一种实现方法，这个是其他解码算法的基础，必须要完全理解，深入掌握，代码非常简洁，可以多看几遍。

### Token 

* token可以理解为在state上，
* prev_指向前一个token（从哪个token扩展来的），引用计数
* cost_累计cost，t时刻，token A经过边 `arc(pdf-id, word-id, graph_weight)` 扩展为token B，计算t时刻特征`Ft`在声学模型上输出是`pdf-id`的得分得到`am_score`, B的cost等于 `A的cost + (-am_score) + graph_weight`
* arc_, token构建时传入的是grpah上对应的arc，而token内部实际存的的arc是LatticeArc，hclg用的StdArc类型的arc里weight只能存一个值，而LatticeArc里weight把acoustic和graph cost* 分开存储，因此在回溯时，可以分别输出acoustic得分和lm得分.
* TokenDelete() 利用用引用计数的方式对token进行释放管理。
* 重载了 < 符号，比较两个token的cost大小, 若 a.cost > b.cost, 则a < b.

### SimpleDecoder类
* prev_toks_ 前一帧时刻的所有tokens，
* cur_toks_ 当前帧扩展出的tokens，
* prev_toks_和cur_toks_都是一个从stateID到token的map，类型为unordered_map<StateId, Token*>
* 对于t-1之前时刻的tokens，不专门维护，他们只在到达终止时刻T时，cur里的token做backtrace时用到。
* beam_
* num_frames_decoded_


Decode()算法非常简单：
1. ProcessEmitting: 每一时刻t，对prev_toks_(t-1时刻)里每个状态里的token（SimpleDecoder只找最优路径，每个state只要保留一个token即可，这就是viterbi算法。每个状态上可能有多个t时刻的token，记录从不同的边到达该状态，NbestDecoder），根据该状态的arc转移，如arc的输入不是epsilon，则创建新的token，放进cur_toks_(t时刻)里。若新的token对应的state上在t时刻已经有了token，则比较两者的cost，选择cost小的.
2. ProcessNonemitting: 而对于输入是epsilon的边，虽然会跳转，但是这是不消耗时间的，因此需要单独处理. 在ProcessEmitting完成后，若cur_toks_中的token所在状态有输入为eps边，则将这些token继续扩展并放入cur_toks_里。
（所以自跳转上的边的输入一定不是epsilon，不然这一步无法结束了。）代码中，fst里保留0作为epsilon的index，所以代码里根据arc.ilabel是否为0判断输入是否为epsilon。
3. PruneToks：因为有很多state，每个state上也都有很多跳转边，如果穷觉所有的扩展tokens，想象一下我们的HCLG上那么多state，每个state上都有0-T时刻的token，这些token太多了，计算和存储都非常大。所以我们要剪枝，最简单的方法就是每一个时刻t，记录当前扩展出的cost最小的token，在扩展时只有新token的cost和最小cost差值小于某个阈值(beam)才保留。（这一步在ProcessEmitting和ProcessNonemitting中完成）
由于最小cost也在不断变化，所以在t时刻全部扩展完成后，对所有的新token再扫描一遍，只保留那些cost和最小cost差值小于某个阈值(beam)的token。（这一步在PruneToks中完成）
4. GetBestPath:当到的T时刻是，只要找出各个cur_toks_里cost最小的state(如果要求final，则必须使用final状态中的token)，然后根据其backtrace指针不断往前回溯找到所有的token，把这些token记录的arc上的output输出即可。GetBestPath里实际是用token里的arc信息构建了一个Lattice，不过这个Lattice其是一条直线。每个状态都只有一个边。

以上就是全部算法。谢谢阅读。


[kaldi-decoder-url]: http://kaldi-asr.org/doc/decoders.html
[kaldi-simple-decoder]: https://github.com/kaldi-asr/kaldi/src/decoder/simple-decoder.cc