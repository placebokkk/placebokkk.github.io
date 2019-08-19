---
layout: post
title:  "理解kaldi中的训练过程(一）- 三音素HMM-GMM模型"
date:   2019-07-29 11:11:59 +0800
categories: kaldi
---
{: class="table-of-content"}
* TOC
{:toc}


先用上一个模型得到对齐数据。

所有的三因素悬链steps/train_deltas.sh这个脚本
```
Usage: steps/train_deltas.sh <num-leaves> <tot-gauss> <data-dir> <lang-dir> <alignment-dir> <exp-dir>
e.g.: steps/train_deltas.sh 2000 10000 data/train_si84_half data/lang exp/mono_ali exp/tri1
main options (for others, see top of script file)
--cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs.
--config <config-file>                           # config containing options
--stage <stage>                                  # stage to do partial re-run from.
   exit 1;
```
决策树