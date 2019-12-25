---
layout: post
title:  "Kaldi中的训练过程(一）- 单音素HMM-GMM模型"
date:   2019-07-29 11:11:59 +0800
categories: kaldi
---
{: class="table-of-content"}
* TOC
{:toc}

# 环境配置

{% highlight bash %}
. ./path.sh || exit 1
. ./cmd.sh || exit 1
ln -fs $KALDI_ROOT/egs/wsj/s5/utils/
ln -fs $KALDI_ROOT/egs/wsj/s5/steps/
{% endhighlight %}

./path.sh中配置了Kaldi的根目录以及相关的工具目录
{% highlight bash %}
# Defining Kaldi root directory
# Setting paths to useful tools
export KALDI_ROOT=/export/maryland/binbinzhang/kaldi
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
{% endhighlight %}

cmd.sh 里配置脚本的执行方式，run.pl使用单机模式，queue.pl使用sun的grid.
{% highlight bash %}
export train_cmd=run.pl
export decode_cmd=run.pl
{% endhighlight %}


# 准备词典

准备词典相关内容

{% highlight bash %}
utils/prepare_lang.sh data/local/dict "!SIL" data/local/lang data/lang
{% endhighlight %}

!SIL是oov_word的符号。

data/local/dict中需要准备的文件
```
extra_questions.txt
lexicon.txt
nonsilence_phones.txt
optional_silence.txt
silence_phones.txt
```

生成的中间文件在data/local/lang中

最终文件在data/lang中，包含
```
L.fst   Lexicon的FST
L_disambig.fst  带消歧符号#n的Lexicon的FST
oov.int oov对应的id
oov.txt oov的word
phones/ 一些文件？？
phones.txt  Lexicon的输入符号表，音素集合
topo    HMM的topo结构
words.txt   Lexicon的输出符号表，词集合
```

默认对phone做_B/_I/_E/_S扩展后缀。如果不想使用，要加选项`--position-dependent-phones false`

```
==> data/lang/phones.txt <==
<eps> 0
SIL 1
SIL_B 2
SIL_E 3
SIL_I 4
SIL_S 5
NG_B 6
NG_E 7
NG_I 8
NG_S 9
```

Topo文件
```
==> data/lang/topo <==
<Topology>
<TopologyEntry>
<ForPhones>
6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139
</ForPhones>
<State> 0 <PdfClass> 0 <Transition> 0 0.75 <Transition> 1 0.25 </State>
<State> 1 <PdfClass> 1 <Transition> 1 0.75 <Transition> 2 0.25 </State>
<State> 2 <PdfClass> 2 <Transition> 2 0.75 <Transition> 3 0.25 </State>
<State> 3 </State>
</TopologyEntry>
<TopologyEntry>
<ForPhones>
1 2 3 4 5
</ForPhones>
<State> 0 <PdfClass> 0 <Transition> 0 0.25 <Transition> 1 0.25 <Transition> 2 0.25 <Transition> 3 0.25 </State>
<State> 1 <PdfClass> 1 <Transition> 1 0.25 <Transition> 2 0.25 <Transition> 3 0.25 <Transition> 4 0.25 </State>
<State> 2 <PdfClass> 2 <Transition> 1 0.25 <Transition> 2 0.25 <Transition> 3 0.25 <Transition> 4 0.25 </State>
<State> 3 <PdfClass> 3 <Transition> 1 0.25 <Transition> 2 0.25 <Transition> 3 0.25 <Transition> 4 0.25 </State>
<State> 4 <PdfClass> 4 <Transition> 4 0.75 <Transition> 5 0.25 </State>
<State> 5 </State>
</TopologyEntry>
</Topology>
```

对于1到5五个SIL音素，使用五状态的HMM。
对于其他音素，使用三状态的HMM.
`<eps>`是保留符号，idx是0.

# 音频标注数据准备
标注数据准备
data下面准备
dev 评估集
test 测试集
train 训练集

data/train至少准备如下四个文件，dev和test也是一样的文件格式
spk2utt text  utt2spk  wav.scp

# 特征提取


```
mfccdir=mfcc
for x in train dev test; do
        steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj 10 data/$x exp/make_mfcc/$x $mfccdir || exit 1;
        steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
        utils/fix_data_dir.sh data/$x || exit 1;
done
```



--nj 10制定job的格式，指定为10，则kaldi里会把数据先切分为10份，放到
data/train/split10/1，data/train/split10/2，... data/train/split10/10

steps/make_mfcc_pitch.sh执行后
* 日志放在exp/make_mfcc/$x中
* 提取的特征数据在mfcc/raw_mfcc_pitch_*.n.ark中
每个音频对应的特征列表在data/train/feats.scp

```
wav001 /home/chaoyang/am/t0/mfcc/raw_mfcc_pitch_train.1.ark:91
wav002 /home/chaoyang/am/t0/mfcc/raw_mfcc_pitch_train.1.ark:2700
```

`:`前是文件，`:`后是文件中的位置。

ark是kaldi的特殊二进制存储格式

steps/compute_cmvn_stats.sh执行后
* 日志放在exp/make_mfcc/$x中
* 提取的特征数据在mfcc/cmvn_x.ark中
* 每个音频对应的特征列表在data/train/cmvn.scp

```
wav001 /home/chaoyang/am/t0/mfcc/cmvn_train.ark:91
wav002 /home/chaoyang/am/t0/mfcc/cmvn_train.ark:470
wav003 /home/chaoyang/am/t0/mfcc/cmvn_train.ark:849
```


`utils/fix_data_dir.sh`用于清除不完整信息的数据
v
```
#This script makes sure that only the segments present in 
#all of "feats.scp", "wav.scp" [if present], segments 
#[if present] text, and utt2spk are present in any of them.
```


# 训练单音素模型
训练HMM-GMM参数时可以用EM方法(soft-align)或者viterbi align(hard align)方法更新。
Kaldi里只用viterbi align的方法。


使用差分特征
```
feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |"
```
初始化HMM-GMM
```
gmm-init-mono $shared_phones_opt "--train-feats=$feats subset-feats --n=10 ark:- ark:-|" $lang/topo $feat_dim \
$dir/0.mdl $dir/tree || exit 1;
 ```

对每个句子按找word序列->音素序列->hmm状态序列展开，得到句子对应的hmm-state级别的fst。
```
compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int $dir/tree $dir/0.mdl  $lang/L.fst \
"ark:sym2int.pl --map-oov $oov_sym -f 2- $lang/words.txt < $sdata/JOB/text|" \
"ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1; 
```
但是第一次训练模型时，模型随机初始化，使用flat的对其方式初始化alignment，然后根据对其信息统计每个gmm的观察统计量。
```
align-equal-compiled "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" ark,t:-  \| \
gmm-acc-stats-ali --binary=true $dir/0.mdl "$feats" ark:- \
$dir/0.JOB.acc || exit 1;
```
根据统计量更新GMM参数
```
gmm-est --min-gaussian-occupancy=3  --mix-up=$numgauss --power=$power \
    $dir/0.mdl "gmm-sum-accs - $dir/0.*.acc|" $dir/1.mdl 2> $dir/log/update.0.log || exit 1;
```

之后利用前一轮的更新的模型，使用viterbi解码（）得到数据在hmm-state上的对齐信息(gmm-align-compiled内部使用FasterDecoder进行解码).然后计算统计量，更新GMM参数。
```
x=1
while [ $x -lt $num_iters ]; do
    gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$[$beam*4] --careful=$careful "$mdl" \ "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" "ark t:|gzip -c >$dir/ali.JOB.gz" \
    || exit 1;
    gmm-acc-stats-ali  $dir/$x.mdl "$feats" "ark:gunzip -c $dir/ali.JOB.gz|" \
      $dir/$x.JOB.acc || exit 1;
    gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss --power=$power $dir/$x.mdl \
    "gmm-sum-accs - $dir/$x.*.acc|" $dir/$[$x+1].mdl || exit 1;
done
```

```
steps/train_mono.sh --cmd run.pl --nj 10 data/train data/lang exp/mono
steps/train_mono.sh: Initializing monophone system.
steps/train_mono.sh: Compiling training graphs
steps/train_mono.sh: Aligning data equally (pass 0)
steps/train_mono.sh: Pass 1
steps/train_mono.sh: Aligning data
steps/train_mono.sh: Pass 2
...
steps/train_mono.sh: Aligning data
steps/train_mono.sh: Pass 39
steps/diagnostic/analyze_alignments.sh --cmd run.pl data/lang exp/mono
steps/diagnostic/analyze_alignments.sh: see stats in exp/mono/log/analyze_alignments.log
81000 warnings in exp/mono/log/align.*.*.log
120 warnings in exp/mono/log/update.*.log
2596 warnings in exp/mono/log/acc.*.*.log
exp/mono: nj=10 align prob=-91.95 over 21.14h [retry=1.9%, fail=0.1%] states=125 gauss=1001
steps/train_mono.sh: Done training monophone system in exp/mono
```


exp/mono下的输出文件
```
0.mdl 初始模型参数
40.mdl  当前轮模型参数，每次更新完，删除上一轮的模型
final.mdl   最终的模型参数
40.occs 当前统计量
final.occs 最终的统计量
ali.[1-10].gz    第1-10份数据的对齐信息
fsts.[1-10].gz   第1-10份数据的fst
log/ 日志信息
phones.txt  音素表
tree    决策树，单音素的决策树放啥？
cmvn_opts  cmvn的配置
num_jobs    job个数
```

# 编译语言模型
此处省略语言模型的训练，假设有个训练好的语言模型data/local/lm/lm.3g.arpa.gz，使用utils/format_lm.sh将其转为fst.
```
## LM training
...
## G compilation, check LG composition
utils/format_lm.sh data/lang data/local/lm/lm.3g.arpa.gz\
    data/local/dict/lexicon.txt data/lang_test || exit 1;
```

生成的语言模型G在data/lang_test/G.fst

输出日志信息如下
```
Converting 'data/local/lm/lm.3g.arpa.gz' to FST
arpa2fst --disambig-symbol=#0 --read-symbol-table=data/lang_test/words.txt - data/lang_test/G.fst
LOG (arpa2fst[5.4.253~1-34df]:Read():arpa-file-parser.cc:94) Reading \data\ section.
LOG (arpa2fst[5.4.253~1-34df]:Read():arpa-file-parser.cc:149) Reading \1-grams: section.
WARNING (arpa2fst[5.4.253~1-34df]:Read():arpa-file-parser.cc:219) line 7 [-4.3244762    <unk>   0] skipped: word '<unk>' not in symbol table
WARNING (arpa2fst[5.4.253~1-34df]:Read():arpa-file-parser.cc:219) line 25 [-3.8398454   马年    -0.35380507] skipped: word '马年' not in symbol table
LOG (arpa2fst[5.4.253~1-34df]:Read():arpa-file-parser.cc:149) Reading \2-grams: section.
LOG (arpa2fst[5.4.253~1-34df]:Read():arpa-file-parser.cc:149) Reading \3-grams: section.
WARNING (arpa2fst[5.4.253~1-34df]:Read():arpa-file-parser.cc:259) Of 11400 parse warnings, 30 were reported. Run program with --max_warnings=-1 to see all warnings
LOG (arpa2fst[5.4.253~1-34df]:RemoveRedundantStates():arpa-lm-compiler.cc:359) Reduced num-states from 20502 to 8951
fstisstochastic data/lang_test/G.fst
3.19443 -0.445902
Succeeded in formatting LM: 'data/local/lm/lm.3g.arpa.gz'
```
其中--disambig-symbol=#0指定G fst里backoff边上的符号.

# 编译HCLG解码图
```
utils/mkgraph.sh data/lang_test exp/mono exp/mono/graph || exit 1;
```
需要data/lang_test中的L.fst, G.fst, exp/mono中的tree和final.mdl,
输出文件exp/mono/graph/
```
HCLG.fst    输入hmm-state（kaldi里输入是transition-id）输出words的G
disambig_tid.int
num_pdfs pdf的个数125个，和occs里个数一致

```

```
tree-info exp/mono/tree
tree-info exp/mono/tree
fsttablecompose data/lang_test/L_disambig.fst data/lang_test/G.fst
fstminimizeencoded
fstdeterminizestar --use-log=true
fstpushspecial
fstisstochastic data/lang_test/tmp/LG.fst
-0.0352461 -0.0360721
[info]: LG not stochastic.
fstcomposecontext --context-size=1 --central-position=0 --read-disambig-syms=data/lang_test/phones/disambig.int --write-disambig-syms=data/lang_test/tmp/disambig_ilabels_1_0.int data/lang_test/tmp/ilabels_1_0.10332
fstisstochastic data/lang_test/tmp/CLG_1_0.fst
-0.0352461 -0.0360721
[info]: CLG not stochastic.
make-h-transducer --disambig-syms-out=exp/mono/graph/disambig_tid.int --transition-scale=1.0 data/lang_test/tmp/ilabels_1_0 exp/mono/tree exp/mono/final.mdl
fstminimizeencoded
fsttablecompose exp/mono/graph/Ha.fst data/lang_test/tmp/CLG_1_0.fst
fstdeterminizestar --use-log=true
fstrmepslocal
fstrmsymbols exp/mono/graph/disambig_tid.int
fstisstochastic exp/mono/graph/HCLGa.fst
0.000234453 -0.0719051
HCLGa is not stochastic
add-self-loops --self-loop-scale=0.1 --reorder=true exp/mono/final.mdl
```

fstinfo exp/mono/graph/HCLG.fst

# 测试模型识别率
在dev集合上解码测试识别率。
```
steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 --stage 2\
  exp/mono/graph data/dev exp/mono/decode_dev
```
日志如下，提示需要local/score.sh来计算
```
decode.sh: feature type is delta
steps/diagnostic/analyze_lats.sh --cmd run.pl exp/mono/graph exp/mono/decode_dev
steps/diagnostic/analyze_lats.sh: see stats in exp/mono/decode_dev/log/analyze_alignments.log
Overall, lattice depth (10,50,90-percentile)=(1,9,96) and mean=35.8
steps/diagnostic/analyze_lats.sh: see stats in exp/mono/decode_dev/log/analyze_lattice_depth_stats.log
steps/decode.sh: Not scoring because local/score.sh does not exist or not executable.
```
准备local/score.sh,计算词错误率wer和字错误率cer
{% highlight bash %}
#!/bin/bash

set -e -o pipefail
set -x
steps/scoring/score_kaldi_wer.sh "$@"
steps/scoring/score_kaldi_cer.sh --stage 2 "$@"

echo "$0: Done"
{% endhighlight %}

再次执行steps/decode.sh
```
steps/scoring/score_kaldi_wer.sh --cmd run.pl data/dev exp/mono/graph exp/mono/decode_dev
steps/scoring/score_kaldi_wer.sh: scoring with word insertion penalty=0.0,0.5,1.0
+ steps/scoring/score_kaldi_cer.sh --stage 2 --cmd run.pl data/dev exp/mono/graph exp/mono/decode_dev
steps/scoring/score_kaldi_cer.sh --stage 2 --cmd run.pl data/dev exp/mono/graph exp/mono/decode_dev
steps/scoring/score_kaldi_cer.sh: scoring with word insertion penalty=0.0,0.5,1.0
+ echo 'local/score.sh: Done'
```

kaldi对使用不同的beam和weight计算识别率，exp/mono/decode_dev/scoring_kaldi/下面可以找到最优的wer和cer，以及一些用于分析错误的文件
```
head exp/mono/decode_dev/scoring_kaldi/best_cer
%WER 40.12 [ 2078 / 5179, 91 ins, 399 del, 1588 sub ] exp/mono/decode_dev/cer_13_0.0
head exp/mono/decode_dev/scoring_kaldi/best_wer
%WER 46.75 [ 1631 / 3489, 178 ins, 392 del, 1061 sub ] exp/mono/decode_dev/wer_13_0.0
```

exp/mono/decode_dev/下放置所有的配置的cer和wer，格式为`*_LMWT_$wip`，cer_13_0.0 表示LMweight=13，word insert penalty=0.0的CER

[kaldi-lattice-url]: http://kaldi-asr.org/doc/lattices.html
[povey-lattice-paper]: https://www.danielpovey.com/files/2012_icassp_lattices.pdf