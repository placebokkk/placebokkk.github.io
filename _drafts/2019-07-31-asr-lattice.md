---
layout: post
title:  "理解kaldi中的Lattice"
date:   2019-08-07 11:11:59 +0800
categories: kaldi
---

Lattice是什么？
给定模型P(W|A)和输入语音A，语音识别的解码目标是找到一个P(W|A)概率最大的文本W，当然，也可以找到最大的N个文本，我们叫N-best解码。
N-best就是N个文本序列，因为很多序列比较相近，可以


参考资料

[jrmeyer-url]: http://jrmeyer.github.io/asr/2016/12/15/Visualize-lattice-kaldi.html
[bo-blog-lattice-url]: http://codingandlearning.blogspot.com/2014/01/kaldi-lattices.html
[kaldi-lattice-url]: http://kaldi-asr.org/doc/lattices.html
[povey-lattice-paper]: https://www.danielpovey.com/files/2012_icassp_lattices.pdf