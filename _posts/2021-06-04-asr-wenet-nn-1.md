---
layout: post
title:  "Wenet网络设计与实现"
date:   2021-06-04 10:00:00 +0800
categories: wenet
---

**本文目前为草稿，仅完成50%**

本文分为三部分
* 端到端语音识别基础
* pytorch的实现介绍
* 进阶内容:mask和cache

## 端到端语音识别基础

传统语音识别通过通过HMM来约束输出和输入的时序同步性，并对音素，词典，语言模型分层次建模。这一框架下，声学模型往往在三音素的hmm状态级别建模，同时声学模型与语言模型分开建模，这和最终的任务目标并不完全一致。
另外，这个框架的模型训练会涉及到上下文相关音素，音素聚类，HMM-GMM训练，帧强制对齐等过程，比较繁琐。

HMM-DNN模型的训练过程，仅用于说明训练过程的复杂，具体内容可以不看。

```
1.对于每个句子扩展为单音素序列，用前向后向EM训练，得到单因素的hmm-单高斯model1。
2.用model1对句子做对齐，单高斯进行2倍分裂，更新模型，迭代这个对齐/分裂的过程n次，得到单因素的hmm-gmm模型model2.
3.用model2对句子做对齐，将音素根据上下文扩展为三音素，使用单高斯学习每个音素的决策树，，最后每个叶子结点对应一个单高斯. 得到一个三音素-hmm-单高斯模型model3
4.类似于第2步，用model3不停迭代分裂高斯，得到三音素hmm-gmm的model4
5. model4对句子做对齐，对齐数据用于帧级别NN训练.
...
```

注意:
一般在传统HMM框架下，会先利用HMM-GMM模型，通过对齐的方式，得到帧级别的对应标注，再通过帧级别损失函数来优化神经网络模型，
但是这并不是必须要，HMM框架也可以不使用帧对齐，后面提到的CTC其实只是HMM框架下一种特殊的topo结构。（End-to-end speech recognition using lattice-free MMI）

而近几年在基于神经网络的端到端建模方式则更佳简洁
* 直接以目标单元作为建模对象，比如中文的字，英文的字符或者BPE. 
* 通过特殊的模型（目标函数），处理输入输出存在多种对齐可能的问题。

语音识别端到端常用的建模方式有两种: CTC目标函数和Attention-Encoder-Decoder模型。在这两种框架下，可以选用各种不同的类型神经网络。比如DNN，RNN，CNN，Self-Attention。

### CTC目标函数

传统语音识别通过HMM来约束输出和输入的对齐方式（时间上保持单调），CTC是一种特殊的HMM约束。


CTC本质上穷举所有合法的输出和输入对齐方式，所谓合法，即对齐后的输出序列能够按CTC规则规约得到的原标注序列，则为合法对齐。

使用CTC目标函数会引入一个blank的输出单元，CTC规约规则为：
* 连续的相同字符进行合并
* 移除blank字符

一个例子：

某段语音数据，输入帧数为7帧（此处仅用于举例），原始的标注序列为“出门问问”。则下面两种对齐，通过CTC规则规约，

```
出-门问问-问  -> 出门问问
出-门--问问 -> 出门问
```

第一个对齐序列"出-门问问-问"是合法对齐学列，第二个对齐序列"出-门--问问"不是合法对齐序列。

除了`出-门问问-问`还有很多其他合法序列，比如
```
出出门问问-问
出出出门问-问
出-门-问-问
出---门问问
...
```

CTC目标函数的思想就是，既然不知道哪个对齐关系是正确的，那就最大化所有合法CTC对齐的概率之和。所以对于这个样本，目标就是最大化如下概率。

```
P(出门问问|X) = P(出-门问问-问|X) + P(出出门问问-问|X)
              + P(出出出门问-问|X)+ ... + P(出---门问问|X)
```

求这个目标函数梯度的一种方式是穷举所有的有效CTC对齐，分别求梯度相加。但是这种方法复杂度太高。由于CTC本身结构，存在一种更高效的动态规划算法，可以极大的提升速度。具体可参考论文和实现：

具体可参考论文，pytorch中也直接实现了这个ctc_loss的backwark方法，实际使用非常简单，ctc_loss即可。


解码时，模型对每一个输入帧都给出输出，这种解码方法称为Frame同步解码。若某些帧输出为blank或者和前一帧是重复的字符，则可以合并。
由于穷举序列中blank占的个数最多。最后模型倾向于输出尽量少的非blank字符，因此解码序列中往往每个非blank字符只输出一次，这个叫做CTC的尖峰效应。


### Attention-based Encoder Decoder

Attention-based Encoder Decoder简称AED，也叫Seq2Seq框架，在ASR领域里，该框架也叫做LAS（Listen, Attend and Spell）。

这个模型的encoder对输入序列进行信息提取，decoder则是一个在目标序列上的自回归模型（输入之前的单元，预测下一个单元），同时在自回归计算时，通过attention方式去获取encoder的输出编码信息，从而能够利用到输入序列的信息。

这种建模方式，可以不必显示建模输出和输入之间的对齐关系，而是利用attention机制交给网络去学习出隐含的对齐。相比如CTC，AED允许输入输出单元之间存在时序上的交换，因此特别适用于机器翻译这种任务。但是对于语音识别或者语音合成这些存在时序单调性的任务，这种无约束反而会带来一些问题。

**AED的解码**

解码时，不需要对每一个输入帧都进行输出，而是根据整个输入序列信息和已输出信息进行下一次输出，直到输出一个特殊结束字符。
这种解码方法称为Label同步解码。

**多说一句**

CTC没有显示构建文本之间的关系，RNN-t模型是一种显示建模了文本关系的帧同步解码的模型。

标准的AED中，decoder和encoder之间cross-attention需要看到encoder的完整序列，所以无法进行流式识别。可利用GMM-attention/Mocha/MMA等单调递进的局部Attention方法进行改进。

### 联合建模

研究者发现，联合使用CTC loss和AED可以有效的加速训练收敛，同时得到更好的识别结果。目前这个方法已经成为端到端学习的标准方案。

在解码时，同时使用CTC和AED的输出，可以提高识别率，但是由于AED本身是非流式的解码，在Wenet中，则没采用联合解码的方式，而是采用了先使用CTC解码，再用AED对CTC的Nbest结果进行Rescoring，这样即结合了两种模型的效果，又可以应用于流式场景。


### 神经网络类型

常用的神经网络类型包括DNN，CNN，RNN，Self-attention等，这些方法进行组合，衍生除了各种模型，Wenet中，对于encoder网络部分，选用了Transformer和Conformer两种类型。Wenet中，对于decoder网络部分，选用了Transformer两种类型。

Transformer中每层单元使用self-attention，res，relu，ff层。

Conformer中每层单元使用conv，self-attention，res，relu，ff层。

**降采样/降帧率**

输入序列越长，即帧的个数越多，网络计算量就越大。而在语音识别中，一定时间范围内的语音信号是接近的，多个连续帧对应的是同一个发音，另外，端到端语音识别使用建模单元一般是一个时间延续较长的单元（粗粒度），比如建模单元是一个中文汉字，假如一个汉字用时0.2s，0.2s对应20帧，那如果将20帧的信息进行合并，比如合并为5帧，则可以线性的减少后续encoder网络的前向计算、CTC loss和AED计算cross attention时的开销。

可以用不同的神经网络来进行降采样，Wenet中使用的是2D-CNN。

### 流式语音识别
模型能否进行流式识别，取决于对右侧的依赖的，但RNN，在基于CNN的模型中，对于右侧的依，然
chunk-based attention




## Wenet中的神经网络设计与实现

前文介绍了端到端神经网络的基本知识，本文介绍Wenet中的设计与实现。

Wenet的代码借鉴了Espnet等开源实现，比较简洁，但是为了实现基于chunk的流式解码，以及处理batch内不等长序列，引入的一些实现技巧，比如cache和mask，使得一些代码细节初次阅读时不直观，
可在第一步学习代码时略过相关内容。

核心模型的代码位于wenet/transformer/

### 模型入口 asr_model.py

**wenet/transformer/asr_model.py**

#### 模型定义
使用pytorch Module构建神经网络时，在init中定义用到的子模块，在forword中定义数据如何使用各模块进行前向计算，即网络的拓扑。

```
class ASRModel(torch.nn.Module)
def __init__():
    self.encoder = encoder 
    self.decoder = decoder
    self.ctc = ctc
    self.criterion_att = LabelSmoothingLoss(...）# AED的loss
```

ASRModel的init中定义了encoder, decoder, ctc, criterion_att几个基本模块。
* encoder是shared Encoder
* decoder是attention-based decoder网络
* ctc是ctc decoder网络（很简单，仅仅是前向网络和softmax）和ctc loss
* criterion_att是attention-based decoder的自回归似然loss，实际是一个LabelSmoothing的loss。

![train-arch](/assets/images/wenet/train-arch.png)


这些模块又有自己的子模块，可以通过print打印出完整的模型结构。
```
model = ASRModel(...)
print(model)
```


#### 创建模型
```
def init_asr_model(config):
```

该方法根据传入的config，创建一个ASRModel实例。 config内容由训练模型时使用的yaml文件提供。这个创建仅仅是构建了一个初始模型，其参数是随机的，可以通过load state加载已经训练好的参数。

#### 前向计算
pytorch框架下，只需定义模型的前向计算forword。对于每个Module，可以通过forward学习其具体实现。

#### 其他接口
ASRModel除了定义模型结构和实现前向计算用于训练外，还有两个功能：
* 提供多种python的解码接口
* 二是提供了runtime中需要使用的接口。

初学者可以先关注训练时使用的forward()函数。


python解码接口
```
recognize() # attention decoder
attention_rescoring() # CTC + attention rescoring
ctc_prefix_beam_search() # CTC prefix beamsearch
ctc_greedy_search() # CTC greedy search
```

用于Runtime的接口, 这些接口均有@torch.jit.export注解，可以在C++中调用
```

subsampling_rate()
right_context()
sos_symbol()
eos_symbol()
forward_encoder_chunk() #
forward_attention_decoder()
ctc_activation()
```


其中比较重要的是:
* forward_attention_decoder() Attention decoder的序列forward计算，非自回归模式。
* ctc_activation() CTC decoder forward计算
* forward_encoder_chunk() 基于chunk的Encoder forward计算



### Encoder网络
**wenet/transformer/encoder.py**

Wenet的encoder支持Transformer和Conformer两种网络结构，实现时使用了模版方法的设计模式进代码复用。BaseEncoder中定义了如下统一的前向过程，由TransformerEncoder，ConformerEncoder继承BaseEncoder后分别定义各自的self.encoders的结构。

```
def forward(...):
    xs, pos_emb, masks = self.embed(xs, masks)
    chunk_masks = add_optional_chunk_mask(xs, ..)
    for layer in self.encoders:
        xs, chunk_masks, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
    if self.normalize_before:
        xs = self.after_norm(xs)
```

可以看到Encoder分为两大部分
* self.embed是Subsampling层
* self.encoders是一组相同结构网络（Encoder Blocks）的堆叠

除了forward，Encoder还实现了两个方法，此处不展开介绍。

* forward_chunk_by_chunk，用于python解码时，模拟流式解码时，基于chunk的前向计算。
* forward_chunk, 用于runtime解码时，基于chunk的前向计算。



下面先介绍Subsampling部分，再介绍Encoder Block

#### Subsampling网络


输入的序列数据越长，即帧的个数越多，网络计算量就越大。而在语音识别中，一定时间范围内的语音信号是接近的，多个连续帧对应的是同一个发音，另外，端到端语音识别使用建模单元一般是一个时间延续较长的单元（粗力度），比如建模单元是一个中文汉字，假如一个汉字用时0.2s，0.2s对应20帧，那如果将20帧的信息进行合并，比如合并为5帧，则可以线性的减少后续encoder网络前向计算、CTC loss和AED计算cross attention时的开销。这个过程叫降采样或者叫降帧率。Wenet中采用2D-CNN来实现降帧率。


Subsampling网络实现参考**wenet/transformer/subsampling.py**,这里选择其中降帧率降低为1/4的网络来说明。

```
class Conv2dSubsampling4(BaseSubsampling):
    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim)
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) / 2 * stride  * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) / 2 * 2 * 1 + (3 - 1) / 2 * 2 * 2
        self.right_context = 6
```

语音任务里有两种使用CNN的方式，一种是2D-Conv，一种是1D-Conv：
* 2D-Conv: 输入数据看作是深度(通道数）为1，高度为F（Fbank特征维度，idim），宽度为T（帧数）的一张图.
* 1D-Conv: 输入数据看作是深度(通道数）为F（Fbank特征维度)，高度为1，宽度为T（帧数）的一张图.

Kaldi中著名的TDNN就是是1D-Conv，而此处的subsampling则使用2D-Conv。

`Conv2dSubsampling4`通过两个`stride=2`的2d-CNN，把图像的宽和高都降为1/4. 因为图像的宽即是帧数，所以帧数变为1/4. 
```
torch.nn.Conv2d(1, odim, kernel_size=3, stride=2)
torch.nn.Conv2d(odim, odim, kernel_size=3, stride=2)
```


self.right_context = 6
卷积使用到了右侧的信息，流式解码时，需要知道右侧上下依赖长度？？？细节有点晕。

具体的实现过程。
```
def forward(...):
    x = x.unsqueeze(1)  # (b, c=1, t, f)
    x = self.conv(x)
    b, c, t, f = x.size()
    x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
    x, pos_emb = self.pos_enc(x, offset)
    return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2]

```

* x = x.unsqueeze(1)  # (b, c=1, t, f) 增加channel维，以符合2dConv需要的数据格式。
* conv(x)中进行两次卷积，此时t维度约等于原来的1/4，因为没加padding，实际上是从长度T变为长度((T-1)/2-1)/2）。注意经过卷积后深度不再是1。
* view(b, t, c * f) 将深度和高度合并平铺到同一维，然后通过self.out(）对每帧Affine变换, 从而每帧是odim维特征。
* pos_enc(x, offset) 经过subsampling之后，帧数变少了，此时再计算Positional Eembedding。

在纯self-attention层构建的网络里，为了保证序列的顺序不可变性而引入了PE，从而交换序列中的两帧，输出会不同。但是由于subsampling的存在，序列本身已经失去了交换不变性，所以其实PE可以省去。

x_mask是原始帧率下的记录batch各序列长度的mask，在计算attention以及ctc loss时均要使用，现在帧数降低了，x_mask也要跟着变化。

返回独立的pos_emb，是因为在rel attention中，需要获取relative pos_emb的信息。在标准attention中该返回值不会被用到。



【图】
torch.nn.Conv2d(1, odim, 3, 2), 使用3*3的kernel，进行stride=2的卷积，通道=odim，经过该层后，变为长度为(T-1)/2（帧数被将采样2倍），高度为(D-1)/2，通道深度为odim的图像。


【图】
再经过一次torch.nn.Conv2d(1, odim, 3, 2)的计算，变成高度为((D-1)/2-1)/2, 长度为((T-1)/2-1)/2的图像



#### Encoder Block
对于Encoder, Wenet提供了Transformer和Conformer两种结构，Conformer在Transformer里引入了卷积层。
强烈建议阅读这篇文章 [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) , 了解Transformer的结构和实现。


Transformer的self.encoders由一组TransformerEncoderLayer组成
```
            self.encoders = torch.nn.ModuleList([
            TransformerEncoderLayer(
                output_size,
                MultiHeadedAttention(attention_heads, output_size,
                                     attention_dropout_rate),
                PositionwiseFeedForward(output_size, linear_units,
                                        dropout_rate), dropout_rate,
                normalize_before, concat_after) for _ in range(num_blocks)
        ])
```

Conformer的self.encoders由一组ConformerEncoderLayer组成
```
            self.encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                RelPositionMultiHeadedAttention(*encoder_selfattn_layer_args),
                PositionwiseFeedForward(*positionwise_layer_args),
                PositionwiseFeedForward(*positionwise_layer_args)
                if macaron_style else None,
                ConvolutionModule(*convolution_layer_args)
                if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ) for _ in range(num_blocks)
        ])

```

ConformerEncoderLayer涉及的主要模块有：
* RelPositionMultiHeadedAttention
* PositionwiseFeedForward
* ConvolutionModule

如果不考虑cache，normalize_before=True，feed_forward_macaron=True，则ConformerEncoderLayer的forward可以简化为
```
def forward(...):

        residual = x
        x = self.norm_ff_macaron(x)
        x = self.feed_forward_macaron(x)
        x = residual + 0.5 * self.dropout(x)
        
        residual = x
        x = self.norm_mha(x)
        x_att = self.self_attn(x, x, x, pos_emb, mask)
        x = residual + self.dropout(x_att)
        
        residual = x
        x = self.norm_conv(x)
        x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
        x = x + self.dropout(x)
        
        residual = x
        x = self.norm_ff(x)
        x = self.feed_forward(x)
        x = residual + 0.5 * self.dropout(x)
        
        x = self.norm_final(x)
```

参考conformer block的图

![conformer](/assets/images/wenet/conformer.png)

Layernorm + RelPositionMultiHeadedAttention + Dropout + Res
Layernorm + ConvolutionModule + Dropout + Res
Layernorm + PositionwiseFeedForward + Dropout + Res


每层前有layernorm，后有dropout，再搭配res。


**Conformer Block - RelPositionMultiHeadedAttention**

*wenet/transformer/attention.py*

attention.py中提供了两种attention的实现，MultiHeadedAttention可用于encoder和decoder的self-attention层，
也可以用作decoder和encoder之间的inter-attention。

原始的Conformer论文中提到的self-attention是Relative Position Multi Headed Attention，这是transformer-xl中提出的一种改进attention，和标准attention的区别在于，其中显示利用了相对位置信息，具体实现可参考文章。 [Conformer ASR中的Relative Positional Embedding](https://zhuanlan.zhihu.com/p/344604604)

注意，wenet中实现的Relative Position Multi Headed Attention实际上是有问题的（和论文不同）, 因为采用正确的实现并没有什么提升，就没有合并更新代码。



**Conformer Block - PositionwiseFeedForward**

*wenet/transformer/positionwise_feed_forward.py*

PositionwiseFeedForward，对每个帧时刻的输入去做Affine计算，即通过一个[H1,H2]的的前向矩阵，把[B,T,H1]变为[B，T，H2]。


**Conformer Block - ConvolutionModule**

wenet/transformer/convolution.py

ConvolutionModule结构如下

Wenet中采用了使用了因果卷积，即不看右侧上下文, 这样无论模型含有多少卷积层，对右侧的上下文都无依赖。
Causal Convolution。wenet/transformer/convolution.py

原始的对称卷积，如果不左右padding，则做完卷积后长度会减小。

![casual-conv](/assets/images/wenet/casual-conv.png)

因此标准的卷积，为了保证卷积后序列长度一致，需要在左右各pad长度为(kernel_size - 1) // 2的 0.

```
  if causal:
    padding = 0
    self.lorder = kernel_size - 1
  else:
    # kernel_size should be an odd number for none causal convolution
    assert (kernel_size - 1) % 2 == 0
    padding = (kernel_size - 1) // 2
    self.lorder = 0
```

因果卷积的实现其实很简单，只在左侧pad长度为kernel_size - 1的0，即可实现。
```
if self.lorder > 0:
  if cache is None:
    x = nn.functional.pad(x, (self.lorder, 0), 'constant', 0.0)
```


# Decoder网络

对于Decoder, Wenet提供了自回归Transformer和双向自回归Transformer结构。 所为自回归，既上一时刻的网络输出要作为当前时刻的网络输入。
在ASR整个任务中，输入是当前产生的文本，输出接下来要产生的文本，因此这个模型建模了语言模型的信息。

这种网络在解码时，只能依次产生输出，而不能一次产生整个输出序列。



# 网络的整体结构

```
ASRModel(
  (encoder): ConformerEncoder(
    (global_cmvn): GlobalCMVN()
    (embed): Conv2dSubsampling4(
      (conv): Sequential(
        (0): Conv2d(1, 256, kernel_size=(3, 3), stride=(2, 2))
        (1): ReLU()
        (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2))
        (3): ReLU()
      )
      (out): Sequential(
        (0): Linear(in_features=4864, out_features=256, bias=True)
      )
      (pos_enc): RelPositionalEncoding(
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
    (after_norm): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
    (encoders): ModuleList(
      (0): ConformerEncoderLayer(
        (self_attn): RelPositionMultiHeadedAttention(
          (linear_q): Linear(in_features=256, out_features=256, bias=True)
          (linear_k): Linear(in_features=256, out_features=256, bias=True)
          (linear_v): Linear(in_features=256, out_features=256, bias=True)
          (linear_out): Linear(in_features=256, out_features=256, bias=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (linear_pos): Linear(in_features=256, out_features=256, bias=False)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=256, out_features=2048, bias=True)
          (activation): Swish()
          (dropout): Dropout(p=0.1, inplace=False)
          (w_2): Linear(in_features=2048, out_features=256, bias=True)
        )
        (feed_forward_macaron): PositionwiseFeedForward(
          (w_1): Linear(in_features=256, out_features=2048, bias=True)
          (activation): Swish()
          (dropout): Dropout(p=0.1, inplace=False)
          (w_2): Linear(in_features=2048, out_features=256, bias=True)
        )
        (conv_module): ConvolutionModule(
          (pointwise_conv1): Conv1d(256, 512, kernel_size=(1,), stride=(1,))
          (depthwise_conv): Conv1d(256, 256, kernel_size=(15,), stride=(1,), groups=256)
          (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (pointwise_conv2): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
          (activation): Swish()
        )
        (norm_ff): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        (norm_mha): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        (norm_ff_macaron): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        (norm_conv): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        (norm_final): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (concat_linear): Linear(in_features=512, out_features=256, bias=True)
      )
      ...
    )
  )
 (decoder): TransformerDecoder(
    (embed): Sequential(
      (0): Embedding(4233, 256)
      (1): PositionalEncoding(
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
    (after_norm): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
    (output_layer): Linear(in_features=256, out_features=4233, bias=True)
    (decoders): ModuleList(
      (0): DecoderLayer(
        (self_attn): MultiHeadedAttention(
          (linear_q): Linear(in_features=256, out_features=256, bias=True)
          (linear_k): Linear(in_features=256, out_features=256, bias=True)
          (linear_v): Linear(in_features=256, out_features=256, bias=True)
          (linear_out): Linear(in_features=256, out_features=256, bias=True)
          (dropout): Dropout(p=0.0, inplace=False)
        )
        (src_attn): MultiHeadedAttention(
          (linear_q): Linear(in_features=256, out_features=256, bias=True)
          (linear_k): Linear(in_features=256, out_features=256, bias=True)
          (linear_v): Linear(in_features=256, out_features=256, bias=True)
          (linear_out): Linear(in_features=256, out_features=256, bias=True)
          (dropout): Dropout(p=0.0, inplace=False)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=256, out_features=2048, bias=True)
          (activation): ReLU()
          (dropout): Dropout(p=0.1, inplace=False)
          (w_2): Linear(in_features=2048, out_features=256, bias=True)
        )
        (norm1): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        (norm3): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (concat_linear1): Linear(in_features=512, out_features=256, bias=True)
        (concat_linear2): Linear(in_features=512, out_features=256, bias=True)
      )
      ...
    )
  )
  (ctc): CTC(
    (ctc_lo): Linear(in_features=256, out_features=4233, bias=True)
    (ctc_loss): CTCLoss()
  )
  (criterion_att): LabelSmoothingLoss(
    (criterion): KLDivLoss()
  )
)
```


