---
layout: post
title:  "Kaldi中的特征提取(二）- 特征变换"
date:   2019-08-05 11:12:59 +0800
categories: kaldi
---

# 特征变换

kaldi的特征变换实现位于src/transform/

* *CMVN* 对特征进行归一化，均值M做减法的归一化，方差V做归一化
* *差分模型（deltas)* 是识别中经常使用的方法，即将不同帧的特征之间（比如当前帧特征减去前一帧特征，以及更高阶的差分）差值也作为特征.
kaldi存储的特征文件保留原始特征，而不包含CMVN变换和差分的特征，在每次需要（训练，识别或者对其）时再提取这些特征。

脚本steps/train_mono.sh（使用deltas特征训练mono-phone）和steps/train_deltas.sh（使用deltas特征训练tri-phone）中的特征变换
{% highlight bash %}
feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |"
{% endhighlight %}
其中apply-cmvn做CMVN变换，add-deltas增加差分维度。

除CMVN+deltas外，CMVN+LDA变换也是常用的变换

steps/align_si.sh，steps/decode.sh等脚本支持使用不同的特征变换

{% highlight bash %}
if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
    cp $srcdir/final.mat $srcdir/full.mat $dir
   ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac
{% endhighlight %}

### CMVN
CMVN对特征每一维做均值/方差归一化，每一维特征之间是独立的。

$$ F_k^{cmvn} = (F_k - \mu_k) / \delta_k $$

**CMVN的模型存储结构**

对于K维特征，kaldi存储 2* K+1的矩阵，其中存储的是统计量而不是计算好的均值和方差。
* 第一行存储K个均值统计量M和数据个数N。使用时，用M[k]/N作为归一化均值，
* 第二行存储K个方差统计量V和一个0（用不到）。

通过cmvn可以了解kaldi的项目整体设计
* 顶层脚本 
  * 训练语料计算cmvn统计量 steps/compute_cmvn_stats.sh
  * 需要应用特征变换的脚本 steps/align_si.sh，steps/decode.sh
* C++实现的工具
  * 训练语料计算cmvn统计量 /src/featbin/compute-cmvn-stats.cc
  * 应用特征变换  /src/featbin/apply-cmvn.cc
* C++核心实现 /src/transform/cmvn.cc


### LDA

### 
[kaldi-lattice-url]: http://kaldi-asr.org/doc/lattices.html
[povey-lattice-paper]: https://www.danielpovey.com/files/2012_icassp_lattices.pdf