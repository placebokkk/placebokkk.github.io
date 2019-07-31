---
layout: post
title:  "理解kaldi中的decoder(二）- 加速解码"
date:   2019-07-30 11:11:59 +0800
categories: kaldi
---

理解kaldi中的decoder(一）中介绍了解码器的设计和kaldi中实现的viterbi解码算法和剪枝方法。本文介绍kaldi中实现的加速版本


max-active
HashList<StateId, Token*>

参考资料

[kaldi-lattice-url]: http://kaldi-asr.org/doc/lattices.html
[povey-lattice-paper]: https://www.danielpovey.com/files/2012_icassp_lattices.pdf