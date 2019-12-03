---
layout: post
title:  "Kaldi中的训练过程(二）- 三音素HMM-GMM模型"
date:   2019-07-29 11:11:59 +0800
categories: kaldi
---
{: class="table-of-content"}
* TOC
{:toc}


先用上一个模型得到对齐数据。
```
steps/align_si.sh --cmd "$train_cmd" --nj 10 \
  data/train data/lang exp/mono exp/mono_ali || exit 1;
```
```
steps/align_si.sh --cmd queue.pl --nj 10 data/train data/lang exp/tri1 exp/tri1_ali
steps/align_si.sh: feature type is delta
steps/align_si.sh: aligning data in data/train using model from exp/tri1, putting alignments in exp/tri1_ali
steps/diagnostic/analyze_alignments.sh --cmd queue.pl data/lang exp/tri1_ali
Use of uninitialized value $joball in numeric lt (<) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 357.
Use of uninitialized value $jobend in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
Use of uninitialized value $num_to_process in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
steps/diagnostic/analyze_alignments.sh: see stats in exp/tri1_ali/log/analyze_alignments.log
steps/align_si.sh: done aligning data.
```

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
 <num-leaves>=2000 ,pdf-id个数
 <tot-gauss>=10000， 总的gauss分量个数。一个GMM(pdf)包含多个gauss，按这个配置平均5个。
 

问题集的格式
exp/tri1/questions.int


需要重新构建解码graph。
LG不用重新编译。
```
tree-info exp/tri1/tree
tree-info exp/tri1/tree
fstcomposecontext --context-size=3 --central-position=1 --read-disambig-syms=data/lang_test/phones/disambig.int --write-disambig-syms=data/lang_test/tmp/disambig_ilabels_3_1.int data/lang_test/tmp/ilabels_3_1.32132
fstisstochastic data/lang_test/tmp/CLG_3_1.fst
0 -0.0360721
[info]: CLG not stochastic.
make-h-transducer --disambig-syms-out=exp/tri1/graph/disambig_tid.int --transition-scale=1.0 data/lang_test/tmp/ilabels_3_1 exp/tri1/tree exp/tri1/final.mdl
fstrmepslocal
fstrmsymbols exp/tri1/graph/disambig_tid.int
fstminimizeencoded
fsttablecompose exp/tri1/graph/Ha.fst data/lang_test/tmp/CLG_3_1.fst
fstdeterminizestar --use-log=true
fstisstochastic exp/tri1/graph/HCLGa.fst
0.000486141 -0.1013
HCLGa is not stochastic
add-self-loops --self-loop-scale=0.1 --reorder=true exp/tri1/final.mdl
steps/decode.sh --cmd queue.pl --config conf/decode.config --nj 10 exp/tri1/graph data/test exp/tri1/decode_test
```



决策树

gmm-info exp/mono/final.mdl
number of phones 337
number of pdfs 125
number of transition-ids 2082
number of transition-states 1021
feature dimension 48
number of gaussians 1001

gmm-info exp/tri1/final.mdl
number of phones 337
number of pdfs 1920
number of transition-ids 28998
number of transition-states 14479
feature dimension 48
number of gaussians 20052

gmm-info exp/tri2/final.mdl
number of phones 337
number of pdfs 1976
number of transition-ids 29364
number of transition-states 14662
feature dimension 48
number of gaussians 20038


LDA一般要用吗？本质在干啥？
steps/train_lda_mllt.sh

acc-lda
est-lda


decode时如果发现exp目录下有final.mat，则使用lda特征
```
case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |";;
```
src/featbin/transform-feats.cc



SAT
说话人自适应训练，为每个说话人训练一个特征变换矩阵。
如果测试集中是同样的说话人，则使用该矩阵。
若测试集中的说话人训练集中没有，则用对角阵。

我们没有说话人信息，每个utt2spk里每个wav对应一个独立说话人，所以每个变换矩阵都是对角阵。

cd/exp/tria4/

copy-matrix ark:trans.1 ark,t:trans.1.txt


TDNN训练

CTM格式
ma001.wav 1 0.260 0.140 是
ma001.wav 1 0.400 0.190 属
ma001.wav 1 0.590 0.110 马
ma001.wav 1 0.700 0.210 的
