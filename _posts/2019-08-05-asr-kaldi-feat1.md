---
layout: post
title:  "Kaldi中的特征提取(一）- 特征类型"
date:   2019-08-05 11:11:59 +0800
categories: kaldi
---
# 特征提取

我们从mfcc特征入手了解kaldi的特征提取的设计实现。

make_mfcc_pitch.sh文件里提取mfcc和pitch特征的部分

{% highlight bash %}
mfcc_feats="ark:extract-segments scp,p:$scp $logdir/segments.JOB ark:- | compute-mfcc-feats $vtln_opts --verbose=2 --config=$mfcc_config ark:- ark:- |"
pitch_feats="ark,s,cs:extract-segments scp,p:$scp $logdir/segments.JOB ark:- | compute-kaldi-pitch-feats --verbose=2 --config=$pitch_config ark:- ark:- | process-kaldi-pitch-feats $postprocess_config_opt ark:- ark:- |"

$cmd JOB=1:$nj $logdir/make_mfcc_pitch_${name}.JOB.log \
  paste-feats --length-tolerance=$paste_length_tolerance "$mfcc_feats" "$pitch_feats" ark:- \| \
  copy-feats --compress=$compress $write_num_frames_opt ark:- \
    ark,scp:$mfcc_pitch_dir/raw_mfcc_pitch_$name.JOB.ark,$mfcc_pitch_dir/raw_mfcc_pitch_$name.JOB.scp \
    || exit 1;
{% endhighlight %}

* compute-mfcc-feats从音频提取的mfcc特征，
* compute-kaldi-pitch-feats从音频提取的pitch特征，

我们先只关注mfcc的提取，看一下compute-mfcc-feats的代码，其源码位于kaldi/src/featbin/compute-mfcc-feats.cc，其中的核心部分
{% highlight cpp %}
Mfcc mfcc(mfcc_opts);
SubVector<BaseFloat> waveform(wave_data.Data(), this_chan);
Matrix<BaseFloat> features;
mfcc.ComputeFeatures(waveform, wave_data.SampFreq(), vtln_warp_local, &features);
{% endhighlight %}

Mfcc类位于kaldi/src/feat/feature-mfcc.h
{% highlight cpp %}
class MfccOptions;
class MfccComputer;
typedef OfflineFeatureTpl<MfccComputer> Mfcc;
{% endhighlight %}



* MfccOptions是mfcc的配置
* MfccComputer是负责Mfcc特征计算的类。
* Kaldi封装了一个类OfflineFeatureTpl<F>，先进行采样率变化，分帧，加窗，dither，预加重等和特征提取算法无关的操作，然后调用对应的特征提取模块F提取特征。

OfflineFeatureTpl<F>的源码位于kaldi/src/feat/feature-common-inl.h

{% highlight cpp %}
template <class F>
void OfflineFeatureTpl<F>::ComputeFeatures(
     const VectorBase<BaseFloat> &wave,
    BaseFloat sample_freq,
    BaseFloat vtln_warp,
    Matrix<BaseFloat> *output) {
    ...
    if (sample_freq == new_sample_freq)
    Compute(wave, vtln_warp, output);
    else {
     DownsampleWaveForm(sample_freq, wave,
                         new_sample_freq, &downsampled_wave);
      Compute(downsampled_wave, vtln_warp, output);
    }
}

template <class F>
void OfflineFeatureTpl<F>::Compute(
    const VectorBase<BaseFloat> &wave,
    BaseFloat vtln_warp,
    Matrix<BaseFloat> *output) {
  ...
  bool use_raw_log_energy = computer_.NeedRawLogEnergy();
  for (int32 r = 0; r < rows_out; r++) {  // r is frame index.
    BaseFloat raw_log_energy = 0.0;
    ExtractWindow(0, wave, r, computer_.GetFrameOptions(),
                  feature_window_function_, &window,
                  (use_raw_log_energy ? &raw_log_energy : NULL));
    SubVector<BaseFloat> output_row(*output, r);
    computer_.Compute(raw_log_energy, vtln_warp, &window, &output_row);
  } 
}

{% endhighlight %}

* DownsampleWaveForm()处理音频采样率和特征配置的采样率不一致的问题，进行采样率转换
* ExtractWindow()位于kaldi/src/feat/feature-window.cc，完成分帧，加窗，去直流，dither，预加重等工作
* 而MfccComputer的Compute()中进行Mfcc相关的特征提取操作，位于 kaldi/src/feat/feature-mfcc.cc

类似的，在feat/下的feature-*文件中有对应的特征提取算法，如plp，fbank，mfcc。

注意pitch特征没有对用的feature-pitch.ccw文件，其计算函数ComputeKaldiPitch位于pitch-functions.cc中，因为Pitch的计算和mfcc,fbank等不同，它不是一个频域分析得到的特征，而是从波形上直接进行提取。

### 
[kaldi-lattice-url]: http://kaldi-asr.org/doc/lattices.html
[povey-lattice-paper]: https://www.danielpovey.com/files/2012_icassp_lattices.pdf