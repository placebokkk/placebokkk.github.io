---
layout: post
title:  "Kaldi-nnet3-上下文和egs"
date:   2020-03-15 11:11:59 +0800
categories: kaldi-nnet3
---
{: class="table-of-content"}
* TOC
{:toc}

# 上下文

## model-context

t时刻的输出label依赖的输入特征的上下文宽度叫做model-context，包括左右的model-context

* model-left-context(也叫net-left-context， 有时在也称为left-context，需要根据语境判断)
* model-right-context(也叫net-right-context，有时在也称为right-context，需要根据语境判断)

## extra-context

kaldi中还有一个概念是extra-left-context和extra-right-context，这个是用于recurrent网络的recurrent计算，
需要多少context计算得到recurrent的输入。 对于Bi-RNN，还有右侧的recurrent输入，所以需要右侧的上下文。

## 网络上下文的计算

定义网络
```
## tdnn.network.xconfig
  input dim=40 name=input
  relu-batchnorm-layer name=tdnn dim=64 input=Append(-2,0,2)
  fast-lstm-layer name=lstm cell-dim=32 decay-time=20 delay=-3
  output-layer name=output input=lstm dim=10 max-change=1.5

```

查看其左右上下文大小
```
## steps/nnet3/xconfig_to_configs.py --xconfig-file tdnn.network.xconfig --config-dir ./
## nnet3-info ref.raw
left-context: 2
right-context: 2
num-parameters: 20586
modulus: 1
```

各层参数，其中+1是affine层的bias项。相加等于20586
```
tdnn affine (40 * 3 +1)* 64 
lstm affine ((64+32)+1) * 32 * 4 
lstm nonlin 32* 3
softmax affine (32+1)*10 
```

看一个更复杂的网络的上下文结果，比如我们使用wsj的chain中的tdnn模型。
```
    input dim=100 name=ivector
    input dim=40 name=input

    idct-layer name=idct input=input dim=40 cepstral-lifter=22 affine-transform-file=$dir/configs/idct.mat
    delta-layer name=delta input=idct
    no-op-component name=input2 input=Append(delta, Scale(1.0, ReplaceIndex(ivector, t, 0)))

    # the first splicing is moved before the lda layer, so no splicing here
    relu-batchnorm-layer name=tdnn1 $tdnn_opts dim=1024 input=input2
    tdnnf-layer name=tdnnf2 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
    tdnnf-layer name=tdnnf3 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
    tdnnf-layer name=tdnnf4 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
    tdnnf-layer name=tdnnf5 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=0
    tdnnf-layer name=tdnnf6 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
    tdnnf-layer name=tdnnf7 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
    tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
    tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
    tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
    tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
    tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
    tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
    linear-component name=prefinal-l dim=192 $linear_opts

    prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1024 small-dim=192
    output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

    prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1024 small-dim=192
    output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
```

查看其左右上下文大小
```
## steps/nnet3/xconfig_to_configs.py --xconfig-file tdnn-lstm.xconfig --config-dir ./
## nnet3-info wsj/configs/ref.raw
left-context: 29
right-context: 29
num-parameters: 8642592
```
其中tdnnf的time-stride表示左右上下文，累加起来1\*3 + 3\*8 = 27.那为何网络左右上下文是29呢，因为delta层也会引入左右上下文依赖。delta-delta时左右上下文都为2.
因此left-context=29，right-context=29.

```
delta(t) = f(t+1) - f(t-1) 
delta-delta(t)  = f(t-2) - 2f(t) + f(t+2)  
```




# EGS
```
/steps/nnet3/chain/get_egs.sh
/steps/nnet3/get_egs.sh
```

nnet3需要提前生成好训练样本，生成的训练样本叫做egs(examples),会根据模型的上下文生成对应egs，一旦模型上下文变了，
egs也需要重新生成。

每个example是一个长度固定chunk，即一组连续的label和其根据上下文需要的输入特征。



下面的是一个模型上下文为左侧16，右侧12对应的egs,01.wav这个音频被切分成了很多chunk，-87表示第87个chunk.
```
01.wav-87 
<Nnet3Eg> 
<NumIo> 2 
    <NnetIo> input 
        <I1V> 36 
        <I1> 0 -16 0 <I1> 0 -15 0 <I1> 0 -14 0 <I1> 0 -13 0 <I1> 0 -12 0 <I1> 0 -11 0 <I1> 0 -10 0 <I1> 0 -9 0 <I1> 0 -8 0 <I1> 0 -7 0 <I1> 0 -6 0 <I1> 0 -5 0 <I1> 0 -4 0 <I1> 0 -3 0 <I1> 0 -2 0 <I1> 0 -1 0 <I1> 0 0 0 <I1> 0 1 0 <I1> 0 2 0 <I1> 0 3 0 <I1> 0 4 0 <I1> 0 5 0 <I1> 0 6 0 <I1> 0 7 0 <I1> 0 8 0 <I1> 0 9 0 <I1> 0 10 0 <I1> 0 11 0 <I1> 0 12 0 <I1> 0 13 0 <I1> 0 14 0 <I1> 0 15 0 <I1> 0 16 0 <I1> 0 17 0 <I1> 0 18 0 <I1> 0 19 0
        [
        97.30364 -3.910992 -30.80304 -3.48616 -66.96941 -29.00227 -88.99083 21.78121 -62.15462 -4.151295 -35.50327 -4.110187 -18.79502 70.5752 -1.083632 38.90316 -15.14383 -4.67984 3.844811 0.7083166 -2.214117 2.550018 1.870166 1.769193 -1.508338 0.2067747 -6.741038 5.639647 -10.63978 10.62695 -7.235626 16.03898 -12.22836 3.21749 -4.498186 1.417891 2.665665 -1.110982 -5.823317 1.687254
        99.10075 2.414833 -38.25296 -1.217627 -71.97943 -41.72618 -68.26828 18.00008 -59.13413 -6.729376 -18.06305 -11.4845 -19.0238 69.58968 4.329955 42.28728 2.413088 -11.88971 -5.693924 5.658363 -0.2441978 4.228565 0.1156468 2.205399 -3.323665 -0.6477451 1.101045 5.354781 -3.874245 4.511787 -4.556805 9.047355 -13.33412 7.345819 -1.971818 -0.5560644 -3.297477 -3.503572 -4.566537 2.10817
        99.55003 9.926746 -53.82625 11.6374 -71.47052 -26.45749 -52.59174 5.153442 -57.12048 -26.77036 0.3643494 -47.99191 -0.248088 63.67664 -3.669809 49.73235 1.612537 -23.90616 -8.469775 -3.113039 -0.4495978 5.011887 0.7205315 1.575324 -1.529328 -0.7170305 3.179681 3.93045 6.694186 -0.7881775 -9.213675 13.28471 -13.14982 5.640185 8.76902 -1.971353 1.758175 -1.027197 -6.577384 2.859537
        ...
        ]
    </NnetIo>
    <NnetIo> output 
        <I1V> 8 
        <I1> 0 0 0 <I1> 0 1 0 <I1> 0 2 0 <I1> 0 3 0 <I1> 0 4 0 <I1> 0 5 0 <I1> 0 6 0 <I1> 0 7 0 
        rows=8 dim=2040 [ 228 1 ] dim=2040 [ 264 1 ] dim=2040 [ 264 1 ] dim=2040 [ 264 1 ] dim=2040 [ 1631 1 ] dim=2040 [ 1631 1 ] dim=2040 [ 1631 1 ] dim=2040 [ 1631 1 ]
    </NnetIo> 
</Nnet3Eg>
```

其中各标签含义如下

* <NumIo> 后表示<NnetIo>的个数，egs里有input和ouput两个<NnetIo>，所以是2
* <NnetIo> 后是该NnetIo的类型，代码里根据改值对后面进行不同的格式解析。
* <I1V>后面跟着帧数，输入是36帧，label是8帧。 8帧，t到t+7，所以需要的输入上下文是t-16到t+19
* <I1> 后是每帧的Index (minibatch_index, time_index, conv_index)
* input中给出所有输入帧的特征组成的矩阵
* output中给出是8帧label的具体值[228,264,264,2641631,1631,1631,1631]. 1表示概率为1(也可以用小于1的比如0.5？？label smoothing??). 2040是output pdf总数？？？？