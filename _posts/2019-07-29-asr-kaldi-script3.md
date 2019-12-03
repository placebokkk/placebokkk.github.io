---
layout: post
title:  "Kaldi中的训练过程(三）- TDNN模型"
date:   2019-07-29 11:11:59 +0800
categories: kaldi
---
{: class="table-of-content"}
* TOC
{:toc}


run tdnn：

local/nnet3/run_tdnn.sh: creating neural net configs
tree-info exp/tri3a_ali/tree
steps/nnet3/xconfig_to_configs.py --xconfig-file exp/nnet3/tdnn_sp/configs/network.xconfig --config-dir exp/nnet3/tdnn_sp/configs/
nnet3-init exp/nnet3/tdnn_sp/configs//init.config exp/nnet3/tdnn_sp/configs//init.raw
LOG (nnet3-init[5.4.253~1-34df]:main():nnet3-init.cc:80) Initialized raw neural net and wrote it to exp/nnet3/tdnn_sp/configs//init.raw
nnet3-info exp/nnet3/tdnn_sp/configs//init.raw
nnet3-init exp/nnet3/tdnn_sp/configs//ref.config exp/nnet3/tdnn_sp/configs//ref.raw
LOG (nnet3-init[5.4.253~1-34df]:main():nnet3-init.cc:80) Initialized raw neural net and wrote it to exp/nnet3/tdnn_sp/configs//ref.raw
nnet3-info exp/nnet3/tdnn_sp/configs//ref.raw
nnet3-init exp/nnet3/tdnn_sp/configs//ref.config exp/nnet3/tdnn_sp/configs//ref.raw
LOG (nnet3-init[5.4.253~1-34df]:main():nnet3-init.cc:80) Initialized raw neural net and wrote it to exp/nnet3/tdnn_sp/configs//ref.raw
nnet3-info exp/nnet3/tdnn_sp/configs//ref.raw
2019-10-09 17:17:58,902 [steps/nnet3/train_dnn.py:35 - <module> - INFO ] Starting DNN trainer (train_dnn.py)
steps/nnet3/train_dnn.py --stage=-10 --cmd=queue.pl --feat.cmvn-opts=--norm-means=false --norm-vars=false --trainer.num-epochs 4 --trainer.optimization.num-jobs-initial 2 --trainer.optimization.num-jobs-final 6 --trainer.optimization.initial-effective-lrate 0.0015 --trainer.optimization.final-effective-lrate 0.00015 --egs.dir  --cleanup.remove-egs true --cleanup.preserve-model-interval 500 --use-gpu $rue --feat-dir=data/train_hires --ali-dir exp/tri3a_ali --lang data/lang --reporting.email= --dir=exp/nnet3/tdnn_sp
['steps/nnet3/train_dnn.py', '--stage=-10', '--cmd=queue.pl', '--feat.cmvn-opts=--norm-means=false --norm-vars=false', '--trainer.num-epochs', '4', '--trainer.optimization.num-jobs-initial', '2', '--tr$iner.optimization.num-jobs-final', '6', '--trainer.optimization.initial-effective-lrate', '0.0015', '--trainer.optimization.final-effective-lrate', '0.00015', '--egs.dir', '', '--cleanup.remove-egs', '$rue', '--cleanup.preserve-model-interval', '500', '--use-gpu', 'true', '--feat-dir=data/train_hires', '--ali-dir', 'exp/tri3a_ali', '--lang', 'data/lang', '--reporting.email=', '--dir=exp/nnet3/tdnn_sp$]
2019-10-09 17:17:58,914 [steps/nnet3/train_dnn.py:176 - train - INFO ] Arguments for the experiment


2019-10-09 17:17:58,914 [steps/nnet3/train_dnn.py:176 - train - INFO ] Arguments for the experiment
{'ali_dir': 'exp/tri3a_ali',
 'backstitch_training_interval': 1,
 'backstitch_training_scale': 0.0,
 'cleanup': True,
 'cmvn_opts': '--norm-means=false --norm-vars=false',
 'combine_sum_to_one_penalty': 0.0,
 'command': 'queue.pl',
 'compute_per_dim_accuracy': False,
 'dir': 'exp/nnet3/tdnn_sp',
 'do_final_combination': True,
 'dropout_schedule': None,
 'egs_command': None,
 'egs_dir': None,
 'egs_opts': None,
 'egs_stage': 0,
 'email': None,
 'exit_stage': None,
 'feat_dir': 'data/train_hires',
 'final_effective_lrate': 0.00015,
 'frames_per_eg': 8,
 'initial_effective_lrate': 0.0015,
 'input_model': None,
 'lang': 'data/lang',
 'max_lda_jobs': 10,
 'max_models_combine': 20,
 'max_objective_evaluations': 30,
 'max_param_change': 2.0,
 'minibatch_size': '512',
 'momentum': 0.0,
 'num_epochs': 4.0,
 'num_jobs_compute_prior': 10,
 'num_jobs_final': 6,
 'num_jobs_initial': 2,
 'online_ivector_dir': None,
 'preserve_model_interval': 500,
 'presoftmax_prior_scale_power': -0.25,
 'prior_subset_size': 20000,
 'proportional_shrink': 0.0,
 'rand_prune': 4.0,
 'remove_egs': True,
 'reporting_interval': 0.1,
 'samples_per_iter': 400000,
 'shuffle_buffer_size': 5000,
 'srand': 0,
 'stage': -10,
 'train_opts': [],
 'use_gpu': 'yes'}

 2019-10-09 17:18:01,991 [steps/nnet3/train_dnn.py:226 - train - INFO ] Initializing a basic network for estimating preconditioning matrix
Use of uninitialized value $joball in numeric lt (<) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 357.
Use of uninitialized value $jobend in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
Use of uninitialized value $num_to_process in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
2019-10-09 17:18:16,196 [steps/nnet3/train_dnn.py:235 - train - INFO ] Generating egs
steps/nnet3/get_egs.sh --cmd queue.pl --cmvn-opts --norm-means=false --norm-vars=false --online-ivector-dir  --left-context 16 --right-context 12 --left-context-initial -1 --right-context-final -1 --stage 0 --samples-per-iter 400000 --frames-per-eg 8 --srand 0 data/train_hires exp/tri3a_ali exp/nnet3/tdnn_sp/egs
steps/nnet3/get_egs.sh: creating egs.  To ensure they are not deleted later you can do:  touch exp/nnet3/tdnn_sp/egs/.nodelete
steps/nnet3/get_egs.sh: feature type is raw
steps/nnet3/get_egs.sh: working out number of frames of training data
utils/data/get_utt2dur.sh: segments file does not exist so getting durations from wave files
cat: write error: Broken pipe
utils/data/get_utt2dur.sh: could not get utterance lengths from sphere-file headers, using wav-to-duration
utils/data/get_utt2dur.sh: computed data/train_hires/utt2dur
feat-to-len 'scp:head -n 10 data/train_hires/feats.scp|' ark,t:-
steps/nnet3/get_egs.sh: working out feature dim
steps/nnet3/get_egs.sh: creating 3 archives, each with 319888 egs, with
steps/nnet3/get_egs.sh:   8 labels per example, and (left,right) context = (16,12)
steps/nnet3/get_egs.sh: copying data alignments
copy-int-vector ark:- ark,scp:exp/nnet3/tdnn_sp/egs/ali.ark,exp/nnet3/tdnn_sp/egs/ali.scp
LOG (copy-int-vector[5.4.253~1-34df]:main():copy-int-vector.cc:83) Copied 27932 vectors of int32.
steps/nnet3/get_egs.sh: Getting validation and training subset examples.
steps/nnet3/get_egs.sh: ... extracting validation and training-subset alignments.
Use of uninitialized value $joball in numeric lt (<) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 357.
Use of uninitialized value $joball in numeric lt (<) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 357.
Use of uninitialized value $jobend in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
Use of uninitialized value $num_to_process in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
Use of uninitialized value $jobend in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
Use of uninitialized value $num_to_process in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
... Getting subsets of validation examples for diagnostics and combination.
Use of uninitialized value $joball in numeric lt (<) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 357.
Use of uninitialized value $joball in numeric lt (<) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 357.
Use of uninitialized value $joball in numeric lt (<) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 357.
Use of uninitialized value $joball in numeric lt (<) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 357.
Use of uninitialized value $jobend in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
Use of uninitialized value $num_to_process in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
Use of uninitialized value $jobend in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
Use of uninitialized value $num_to_process in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
Use of uninitialized value $jobend in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
Use of uninitialized value $num_to_process in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
Use of uninitialized value $jobend in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
Use of uninitialized value $num_to_process in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
steps/nnet3/get_egs.sh: Generating training examples on disk
steps/nnet3/get_egs.sh: recombining and shuffling order of archives on disk

steps/nnet3/get_egs.sh: removing temporary archives
steps/nnet3/get_egs.sh: removing temporary alignments
steps/nnet3/get_egs.sh: Finished preparing training examples
2019-10-09 17:32:08,485 [steps/nnet3/train_dnn.py:273 - train - INFO ] Computing the preconditioning matrix for input features
Use of uninitialized value $joball in numeric lt (<) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 357.
Use of uninitialized value $jobend in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
Use of uninitialized value $num_to_process in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
Use of uninitialized value $joball in numeric lt (<) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 357.
Use of uninitialized value $jobend in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
Use of uninitialized value $num_to_process in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
2019-10-09 17:33:16,798 [steps/nnet3/train_dnn.py:284 - train - INFO ] Computing initial vector for FixedScaleComponent before softmax, using priors^-0.25 and rescaling to average 1
Use of uninitialized value $joball in numeric lt (<) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 357.
Use of uninitialized value $jobend in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
Use of uninitialized value $num_to_process in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
2019-10-09 17:33:50,491 [steps/nnet3/train_dnn.py:291 - train - INFO ] Preparing the initial acoustic model.
Use of uninitialized value $joball in numeric lt (<) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 357.
Use of uninitialized value $jobend in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
Use of uninitialized value $num_to_process in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
Use of uninitialized value $joball in numeric lt (<) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 357.
Use of uninitialized value $jobend in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
Use of uninitialized value $num_to_process in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
2019-10-09 17:34:18,304 [steps/nnet3/train_dnn.py:317 - train - INFO ] Training will run for 4.0 epochs = 38 iterations
2019-10-09 17:34:18,305 [steps/nnet3/train_dnn.py:351 - train - INFO ] Iter: 0/37    Epoch: 0.00/4.0 (0.0% complete)    lr: 0.003000
Use of uninitialized value $joball in numeric lt (<) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 357.
Use of uninitialized value $joball in numeric lt (<) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 357.
Use of uninitialized value $joball in numeric lt (<) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 357.
Use of uninitialized value $joball in numeric lt (<) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 357.
Use of uninitialized value $jobend in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
Use of uninitialized value $num_to_process in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
Use of uninitialized value $jobend in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
Use of uninitialized value $num_to_process in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
Use of uninitialized value $jobend in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
Use of uninitialized value $num_to_process in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
Use of uninitialized value $jobend in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
Use of uninitialized value $num_to_process in subtraction (-) at /export/share-nfs-1/chaoyang/am/t0/utils/queue.pl line 501.
queue.pl: job failed with status 143, log is in exp/nnet3/tdnn_sp/log/train.0.1.log
2019-10-09 17:34:32,742 [steps/libs/common.py:235 - background_command_waiter - ERROR ] Command exited with status 1: queue.pl --gpu 1 exp/nnet3/tdnn_sp/log/train.0.1.log                     nnet3-train --use-gpu=yes  --write-cache=exp/nnet3/tdnn_sp/cache.1                       --print-interval=10                     --momentum=0.0                     --max-param-change=1.41421356237                     --backstitch-training-scale=0.0                     --l2-regularize-factor=0.5                     --backstitch-training-interval=1                     --srand=0                       "nnet3-copy --learning-rate=0.003 --scale=1.0 exp/nnet3/tdnn_sp/0.mdl - |" "ark,bg:nnet3-copy-egs --frame=1              ark:exp/nnet3/tdnn_sp/egs/egs.1.ark ark:- |             nnet3-shuffle-egs --buffer-size=5000             --srand=0 ark:- ark:- |              nnet3-merge-egs --minibatch-size=256 ark:- ark:- |"                     exp/nnet3/tdnn_sp/1.1.raw
queue.pl: job failed with status 143, log is in exp/nnet3/tdnn_sp/log/train.0.2.log




steps/nnet3/train_dnn.py --stage=$train_stage \
    --cmd="$cuda_cmd" \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --egs.dir "$common_egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval 500 \
    --use-gpu true \
    --feat-dir=data/${train_set}_hires \
    --ali-dir $ali_dir \
    --lang data/lang \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;

stage 0开始训练， 
-4 生成egs，
-3 compute_presoftmax_prior_scale
-2
-1

--egs.dir 指定egs的位置，若为空，生成egs，位置在
--cleanup.remove-egs 设为true则执行完删除生成的egs

怎么指定egs

==> exp/mono/decode_test/scoring_kaldi/best_cer <==
%WER 40.12 [ 2078 / 5179, 91 ins, 399 del, 1588 sub ] exp/mono/decode_test/cer_13_0.0

==> exp/tri1/decode_test/scoring_kaldi/best_cer <==
%WER 23.31 [ 1207 / 5179, 119 ins, 109 del, 979 sub ] exp/tri1/decode_test/cer_13_0.0

==> exp/tri2/decode_test/scoring_kaldi/best_cer <==
%WER 23.05 [ 1194 / 5179, 94 ins, 165 del, 935 sub ] exp/tri2/decode_test/cer_16_0.5

==> exp/tri3a/decode_test/scoring_kaldi/best_cer <==
%WER 21.16 [ 1096 / 5179, 79 ins, 168 del, 849 sub ] exp/tri3a/decode_test/cer_14_0.0

==> exp/tri4a/decode_test/scoring_kaldi/best_cer <==
%WER 21.30 [ 1103 / 5179, 91 ins, 168 del, 844 sub ] exp/tri4a/decode_test/cer_15_0.0

==> exp/nnet3/tdnn_sp/decode_test/scoring_kaldi/best_cer <==
%WER 15.74 [ 815 / 5179, 28 ins, 287 del, 500 sub ] exp/nnet3/tdnn_sp/decode_test/cer_7_0.5

chain和tdnn差不多
==> exp/chain/tdnn_1a_all_sp/decode_test/scoring_kaldi/best_cer
%WER 15.87 [ 822 / 5179, 37 ins, 119 del, 666 sub ] exp/chain/tdnn_1a_all_sp/decode_test/cer_7_0.0