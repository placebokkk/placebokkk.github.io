---
layout: post
title:  "Kaldi中的训练过程(四）- Chain模型"
date:   2019-11-04 11:11:59 +0800
categories: kaldi
---
{: class="table-of-content"}
* TOC
{:toc}

local/chain/run_tdnn.sh
utils/copy_data_dir.sh: copied data from data/train to data/train_hires
utils/validate_data_dir.sh: Successfully validated data-directory data/train_hires
steps/make_mfcc_pitch.sh --mfcc-config conf/mfcc_hires.conf --pitch-config conf/pitch.conf --nj 15 data/train_hires exp/make_mfcc/ mfcc_hires
utils/validate_data_dir.sh: Successfully validated data-directory data/train_hires
steps/make_mfcc_pitch.sh: [info]: no segments file exists: assuming wav.scp indexed by utterance.
steps/make_mfcc_pitch.sh: Succeeded creating MFCC and pitch features for train_hires
utils/copy_data_dir.sh: copied data from data/dev to data/dev_hires
utils/validate_data_dir.sh: Successfully validated data-directory data/dev_hires
steps/make_mfcc_pitch.sh --mfcc-config conf/mfcc_hires.conf --pitch-config conf/pitch.conf --nj 15 data/dev_hires exp/make_mfcc/ mfcc_hires
steps/make_mfcc_pitch.sh: moving data/dev_hires/feats.scp to data/dev_hires/.backup
utils/validate_data_dir.sh: Successfully validated data-directory data/dev_hires
steps/make_mfcc_pitch.sh: [info]: no segments file exists: assuming wav.scp indexed by utterance.
steps/make_mfcc_pitch.sh: Succeeded creating MFCC and pitch features for dev_hires
utils/copy_data_dir.sh: copied data from data/test to data/test_hires
utils/validate_data_dir.sh: Successfully validated data-directory data/test_hires
steps/make_mfcc_pitch.sh --mfcc-config conf/mfcc_hires.conf --pitch-config conf/pitch.conf --nj 15 data/test_hires exp/make_mfcc/ mfcc_hires
steps/make_mfcc_pitch.sh: moving data/test_hires/feats.scp to data/test_hires/.backup
utils/validate_data_dir.sh: Successfully validated data-directory data/test_hires
steps/make_mfcc_pitch.sh: [info]: no segments file exists: assuming wav.scp indexed by utterance.
steps/make_mfcc_pitch.sh: Succeeded creating MFCC and pitch features for test_hires
local/chain/run_tdnn.sh: computing a subset of data to train the diagonal UBM.
utils/data/subset_data_dir.sh: reducing #utt from 28000 to 7000
steps/compute_cmvn_stats.sh exp/chain/diag_ubm_all/train_subset
Succeeded creating CMVN stats for train_subset
local/chain/run_tdnn.sh: computing a PCA transform from the hires data.
feat-to-dim scp:exp/chain/diag_ubm_all/train_subset/feats.scp -
steps/online/nnet2/get_pca_transform.sh --cmd queue.pl --splice-opts --left-context=3 --right-context=3 --max-utts 10000 --subsample 2 --dim 43 exp/chain/diag_ubm_all/train_subset exp/chain/pca_transform_all
Done estimating PCA transform in exp/chain/pca_transform_all
local/chain/run_tdnn.sh: training the diagonal UBM.
steps/online/nnet2/train_diag_ubm.sh --cmd queue.pl --nj 15 --num-frames 700000 --num-threads 8 exp/chain/diag_ubm_all/train_subset 512 exp/chain/pca_transform_all exp/chain/diag_ubm_all
steps/online/nnet2/train_diag_ubm.sh: expecting file conf/online_cmvn.conf to exist

local/chain/run_tdnn.sh --stage 6
local/chain/run_tdnn.sh: computing a subset of data to train the diagonal UBM.
utils/data/subset_data_dir.sh: reducing #utt from 28000 to 7000
steps/compute_cmvn_stats.sh exp/chain/diag_ubm_all/train_subset
Succeeded creating CMVN stats for train_subset
local/chain/run_tdnn.sh: computing a PCA transform from the hires data.
feat-to-dim scp:exp/chain/diag_ubm_all/train_subset/feats.scp -
steps/online/nnet2/get_pca_transform.sh --cmd queue.pl --splice-opts --left-context=3 --right-context=3 --max-utts 10000 --subsample 2 --dim 43 exp/chain/diag_ubm_all/train_subset exp/chain/pca_transfor
m_all
Done estimating PCA transform in exp/chain/pca_transform_all
local/chain/run_tdnn.sh: training the diagonal UBM.
steps/online/nnet2/train_diag_ubm.sh --cmd queue.pl --nj 15 --num-frames 700000 --num-threads 8 exp/chain/diag_ubm_all/train_subset 512 exp/chain/pca_transform_all exp/chain/diag_ubm_all
steps/online/nnet2/train_diag_ubm.sh: Directory exp/chain/diag_ubm_all already exists. Backing up diagonal UBM in exp/chain/diag_ubm_all/backup.jV0
steps/online/nnet2/train_diag_ubm.sh: initializing model from E-M in memory,
steps/online/nnet2/train_diag_ubm.sh: starting from 256 Gaussians, reaching 512;
steps/online/nnet2/train_diag_ubm.sh: for 20 iterations, using at most 700000 frames of data
Getting Gaussian-selection info
steps/online/nnet2/train_diag_ubm.sh: will train for 4 iterations, in parallel over
steps/online/nnet2/train_diag_ubm.sh: 15 machines, parallelized with 'queue.pl'
steps/online/nnet2/train_diag_ubm.sh: Training pass 0
steps/online/nnet2/train_diag_ubm.sh: Training pass 1
steps/online/nnet2/train_diag_ubm.sh: Training pass 2
steps/online/nnet2/train_diag_ubm.sh: Training pass 3
local/chain/run_tdnn.sh: training the iVector extractor
steps/online/nnet2/train_ivector_extractor.sh --cmd queue.pl --nj 15 data/train_hires exp/chain/diag_ubm_all exp/chain/extractor_all
steps/online/nnet2/train_ivector_extractor.sh: doing Gaussian selection and posterior computation
Accumulating stats (pass 0)
Summing accs (pass 0)
Updating model (pass 0)
Accumulating stats (pass 1)
Summing accs (pass 1)
Updating model (pass 1)
Accumulating stats (pass 2)
Summing accs (pass 2)
Updating model (pass 2)
Accumulating stats (pass 3)
Summing accs (pass 3)
Summing accs (pass 4)
Updating model (pass 4)
Accumulating stats (pass 5)
Summing accs (pass 5)
Updating model (pass 5)
Accumulating stats (pass 6)
Summing accs (pass 6)
Updating model (pass 6)
Accumulating stats (pass 7)
Summing accs (pass 7)
Updating model (pass 7)
Accumulating stats (pass 8)
Summing accs (pass 8)
Updating model (pass 8)
Accumulating stats (pass 9)
Summing accs (pass 9)
Updating model (pass 9)
steps/online/nnet2/copy_data_dir.sh: this script is deprecated, please use utils/data/modify_speaker_info.sh.
steps/online/nnet2/copy_data_dir.sh: mapping cmvn.scp, but you may want to recompute it if it's needed,
 as it would probably change.
steps/online/nnet2/copy_data_dir.sh: copied data from data/train_hires to data/train_hires_max2, with --utts-per-spk-max 2
utils/validate_data_dir.sh: Successfully validated data-directory data/train_hires_max2
steps/online/nnet2/extract_ivectors_online.sh --cmd queue.pl --nj 15 data/train_hires_max2 exp/chain/extractor_all exp/chain/ivectors_train_all
steps/online/nnet2/extract_ivectors_online.sh: extracting iVectors
steps/online/nnet2/extract_ivectors_online.sh: combining iVectors across jobs
steps/online/nnet2/extract_ivectors_online.sh: done extracting (online) iVectors to exp/chain/ivectors_train_all using the extractor in exp/chain/extractor_all.
steps/online/nnet2/copy_data_dir.sh: this script is deprecated, please use utils/data/modify_speaker_info.sh.
steps/online/nnet2/copy_data_dir.sh: mapping cmvn.scp, but you may want to recompute it if it's needed,
 as it would probably change.
steps/online/nnet2/copy_data_dir.sh: copied data from data/dev_hires to data/dev_hires_max2, with --utts-per-spk-max 2
utils/validate_data_dir.sh: Successfully validated data-directory data/dev_hires_max2
steps/online/nnet2/extract_ivectors_online.sh --cmd queue.pl --nj 15 data/dev_hires_max2 exp/chain/extractor_all exp/chain/ivectors_dev_all
steps/online/nnet2/extract_ivectors_online.sh: extracting iVectors
steps/online/nnet2/extract_ivectors_online.sh: combining iVectors across jobs
steps/online/nnet2/extract_ivectors_online.sh: done extracting (online) iVectors to exp/chain/ivectors_dev_all using the extractor in exp/chain/extractor_all.
steps/online/nnet2/copy_data_dir.sh: this script is deprecated, please use utils/data/modify_speaker_info.sh.
steps/online/nnet2/copy_data_dir.sh: mapping cmvn.scp, but you may want to recompute it if it's needed,
 as it would probably change.
steps/online/nnet2/copy_data_dir.sh: copied data from data/test_hires to data/test_hires_max2, with --utts-per-spk-max 2
utils/validate_data_dir.sh: Successfully validated data-directory data/test_hires_max2
steps/online/nnet2/extract_ivectors_online.sh --cmd queue.pl --nj 15 data/test_hires_max2 exp/chain/extractor_all exp/chain/ivectors_test_all
steps/online/nnet2/extract_ivectors_online.sh: extracting iVectors
steps/online/nnet2/extract_ivectors_online.sh: combining iVectors across jobs
steps/online/nnet2/extract_ivectors_online.sh: done extracting (online) iVectors to exp/chain/ivectors_test_all using the extractor in exp/chain/extractor_all.
steps/align_fmllr_lats.sh --nj 10 --cmd queue.pl data/train data/lang exp/tri3a exp/tri4_sp_lats
steps/align_fmllr_lats.sh: feature type is lda
steps/align_fmllr_lats.sh: compiling training graphs
steps/align_fmllr_lats.sh: aligning data in data/train using exp/tri3a/final.mdl and speaker-independent features.
steps/align_fmllr_lats.sh: computing fMLLR transforms
steps/align_fmllr_lats.sh: generating lattices containing alternate pronunciations.
steps/align_fmllr_lats.sh: done generating lattices from training transcripts.
398 warnings in exp/tri4_sp_lats/log/align_pass1.*.log
33 warnings in exp/tri4_sp_lats/log/generate_lattices.*.log
27611 warnings in exp/tri4_sp_lats/log/fmllr.*.log
steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 --context-opts --context-width=2 --central-position=1 --cmd queue.pl 5000 data/train data/lang_chain exp/tri3a_ali exp/chain/tri4_cd_tree_sp
train_sat.sh: no such file data/train/feats.scp



local/chain/run_tdnn.sh --stage 7
steps/align_fmllr_lats.sh --nj 10 --cmd queue.pl data/train data/lang exp/tri3a exp/tri3a_lats
steps/align_fmllr_lats.sh: feature type is lda
steps/align_fmllr_lats.sh: compiling training graphs
steps/align_fmllr_lats.sh: aligning data in data/train using exp/tri3a/final.mdl and speaker-independent features.
steps/align_fmllr_lats.sh: computing fMLLR transforms
steps/align_fmllr_lats.sh: generating lattices containing alternate pronunciations.
steps/align_fmllr_lats.sh: done generating lattices from training transcripts.
33 warnings in exp/tri3a_lats/log/generate_lattices.*.log
27611 warnings in exp/tri3a_lats/log/fmllr.*.log
398 warnings in exp/tri3a_lats/log/align_pass1.*.log
steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 --context-opts --context-width=2 --central-position=1 --cmd queue.pl 5000 data/train data/lang_chain exp/tri3a_ali exp/chain/tri4_cd_tree_sp
steps/nnet3/chain/build_tree.sh: feature type is lda
steps/nnet3/chain/build_tree.sh: Using transforms from exp/tri3a_ali
steps/nnet3/chain/build_tree.sh: Initializing monophone model (for alignment conversion, in case topology changed)
steps/nnet3/chain/build_tree.sh: Accumulating tree stats
steps/nnet3/chain/build_tree.sh: Getting questions for tree clustering.
steps/nnet3/chain/build_tree.sh: Building the tree
steps/nnet3/chain/build_tree.sh: Initializing the model
WARNING (gmm-init-model[5.5.532~1-7f0c9]:InitAmGmm():gmm-init-model.cc:55) Tree has pdf-id 39 with no stats; corresponding phone list: 330 331 332 333
This is a bad warning.
steps/nnet3/chain/build_tree.sh: Converting alignments from exp/tri3a_ali to use current tree
steps/nnet3/chain/build_tree.sh: Done building tree
local/chain/run_tdnn.sh: creating neural net configs using the xconfig parser
feat-to-dim scp:data/train_hires/feats.scp -
tree-info exp/chain/tri4_cd_tree_sp/tree



local/chain/run_tdnn.sh --stage 10
local/chain/run_tdnn.sh: creating neural net configs using the xconfig parser
feat-to-dim scp:data/train_hires/feats.scp -
tree-info exp/chain/tri4_cd_tree_sp/tree
steps/nnet3/xconfig_to_configs.py --xconfig-file exp/chain/tdnn_1a_all_sp/configs/network.xconfig --config-dir exp/chain/tdnn_1a_all_sp/configs/
nnet3-init exp/chain/tdnn_1a_all_sp/configs//init.config exp/chain/tdnn_1a_all_sp/configs//init.raw
LOG (nnet3-init[5.4.253~1-34df]:main():nnet3-init.cc:80) Initialized raw neural net and wrote it to exp/chain/tdnn_1a_all_sp/configs//init.raw
nnet3-info exp/chain/tdnn_1a_all_sp/configs//init.raw
nnet3-init exp/chain/tdnn_1a_all_sp/configs//ref.config exp/chain/tdnn_1a_all_sp/configs//ref.raw
LOG (nnet3-init[5.4.253~1-34df]:main():nnet3-init.cc:80) Initialized raw neural net and wrote it to exp/chain/tdnn_1a_all_sp/configs//ref.raw
nnet3-info exp/chain/tdnn_1a_all_sp/configs//ref.raw
nnet3-init exp/chain/tdnn_1a_all_sp/configs//ref.config exp/chain/tdnn_1a_all_sp/configs//ref.raw
LOG (nnet3-init[5.4.253~1-34df]:main():nnet3-init.cc:80) Initialized raw neural net and wrote it to exp/chain/tdnn_1a_all_sp/configs//ref.raw
nnet3-info exp/chain/tdnn_1a_all_sp/configs//ref.raw
2019-11-04 16:39:26,796 [steps/nnet3/chain/train.py:33 - <module> - INFO ] Starting chain model trainer (train.py)
steps/nnet3/chain/train.py --stage -10 --cmd queue.pl --feat.online-ivector-dir exp/chain/ivectors_train_all --feat.cmvn-opts --norm-means=false --norm-vars=false --chain.xent-regularize 0.1 --chain.lea
ky-hmm-coefficient 0.1 --chain.l2-regularize 0.00005 --chain.apply-deriv-weights false --chain.lm-opts=--num-extra-lm-states=2000 --egs.dir  --egs.stage -10 --egs.opts --frames-overlap-per-eg 0 --egs.ch
unk-width 150,110,90 --trainer.dropout-schedule 0,0@0.20,0.3@0.50,0 --trainer.num-chunk-per-minibatch 128 --trainer.frames-per-iter 1500000 --trainer.num-epochs 4 --trainer.optimization.num-jobs-initial
 2 --trainer.optimization.num-jobs-final 4 --trainer.optimization.initial-effective-lrate 0.001 --trainer.optimization.final-effective-lrate 0.0001 --trainer.max-param-change 2.0 --cleanup.remove-egs tr
ue --feat-dir data/train_hires --tree-dir exp/chain/tri4_cd_tree_sp --lat-dir exp/tri3a_lats --dir exp/chain/tdnn_1a_all_sp
['steps/nnet3/chain/train.py', '--stage', '-10', '--cmd', 'queue.pl', '--feat.online-ivector-dir', 'exp/chain/ivectors_train_all', '--feat.cmvn-opts', '--norm-means=false --norm-vars=false', '--chain.xe
nt-regularize', '0.1', '--chain.leaky-hmm-coefficient', '0.1', '--chain.l2-regularize', '0.00005', '--chain.apply-deriv-weights', 'false', '--chain.lm-opts=--num-extra-lm-states=2000', '--egs.dir', '',
'--egs.stage', '-10', '--egs.opts', '--frames-overlap-per-eg 0', '--egs.chunk-width', '150,110,90', '--trainer.dropout-schedule', '0,0@0.20,0.3@0.50,0', '--trainer.num-chunk-per-minibatch', '128', '--tr
ainer.frames-per-iter', '1500000', '--trainer.num-epochs', '4', '--trainer.optimization.num-jobs-initial', '2', '--trainer.optimization.num-jobs-final', '4', '--trainer.optimization.initial-effective-lr
ate', '0.001', '--trainer.optimization.final-effective-lrate', '0.0001', '--trainer.max-param-change', '2.0', '--cleanup.remove-egs', 'true', '--feat-dir', 'data/train_hires', '--tree-dir', 'exp/chain/t
ri4_cd_tree_sp', '--lat-dir', 'exp/tri3a_lats', '--dir', 'exp/chain/tdnn_1a_all_sp']
2019-11-04 16:39:26,812 [steps/nnet3/chain/train.py:274 - train - INFO ] Arguments for the experiment
{'alignment_subsampling_factor': 3,
 'apply_deriv_weights': False,
 'backstitch_training_interval': 1,
 'backstitch_training_scale': 0.0,
 'chunk_left_context': 0,
 'chunk_left_context_initial': -1,
 'chunk_right_context': 0,
 'chunk_right_context_final': -1,
 'chunk_width': '150,110,90',
 'cleanup': True,
 'cmvn_opts': '--norm-means=false --norm-vars=false',
 'combine_sum_to_one_penalty': 0.0,
 'command': 'queue.pl',
 'compute_per_dim_accuracy': False,
 'deriv_truncate_margin': None,
 'dir': 'exp/chain/tdnn_1a_all_sp',
 'do_final_combination': True,
 'dropout_schedule': '0,0@0.20,0.3@0.50,0',
 'dynamic_augment': False,
 'egs_command': None,
 'egs_dir': None,
 'egs_opts': '--frames-overlap-per-eg 0',
 'egs_stage': -10,
 'email': None,
 'exit_stage': None,
 'feat_dir': 'data/train_hires',
 'final_effective_lrate': 0.0001,
 'frame_subsampling_factor': 3,
 'frames_per_iter': 1500000,
 'initial_effective_lrate': 0.001,
 'input_model': None,
 'l2_regularize': 5e-05,
 'lat_dir': 'exp/tri3a_lats',
 'leaky_hmm_coefficient': 0.1,
 'left_deriv_truncate': None,
 'left_tolerance': 5,
 'lm_opts': '--num-extra-lm-states=2000',
 'max_lda_jobs': 10,
 'max_models_combine': 20,
 'max_objective_evaluations': 30,
 'max_param_change': 2.0,
 'momentum': 0.0,
 'num_chunk_per_minibatch': '128',
 'num_epochs': 4.0,
 'num_jobs_final': 4,
 'num_jobs_initial': 2,
 'online_ivector_dir': 'exp/chain/ivectors_train_all',
 'preserve_model_interval': 100,
 'presoftmax_prior_scale_power': -0.25,
 'proportional_shrink': 0.0,
 'rand_prune': 4.0,
 'remove_egs': True,
 'reporting_interval': 0.1,
 'right_tolerance': 5,
 'samples_per_iter': 400000,
 'shrink_saturation_threshold': 0.4,
 'shrink_value': 1.0,
 'shuffle_buffer_size': 5000,
 'srand': 0,
 'stage': -10,
 'train_opts': [],
 'tree_dir': 'exp/chain/tri4_cd_tree_sp',
 'use_gpu': 'yes',
 'xent_regularize': 0.1}
2019-11-04 16:39:39,569 [steps/nnet3/chain/train.py:328 - train - INFO ] Creating phone language-model
2019-11-04 16:40:10,301 [steps/nnet3/chain/train.py:333 - train - INFO ] Creating denominator FST
copy-transition-model exp/chain/tri4_cd_tree_sp/final.mdl exp/chain/tdnn_1a_all_sp/0.trans_mdl
LOG (copy-transition-model[5.4.253~1-34df]:main():copy-transition-model.cc:62) Copied transition model.
2019-11-04 16:40:20,423 [steps/nnet3/chain/train.py:340 - train - INFO ] Initializing a basic network for estimating preconditioning matrix
��^B[2019-11-04 16:40:33,320 [steps/nnet3/chain/train.py:362 - train - INFO ] Generating egs
steps/nnet3/chain/get_egs.sh --frames-overlap-per-eg 0 --cmd queue.pl --cmvn-opts --norm-means=false --norm-vars=false --online-ivector-dir exp/chain/ivectors_train_all --left-context 22 --right-context 22 --left-context-initial -1 --right-context-final -1 --left-tolerance 5 --right-tolerance 5 --frame-subsampling-factor 3 --alignment-subsampling-factor 3 --stage -10 --frames-per-iter 1500000 --frames-per-eg 150,110,90 --srand 0 data/train_hires exp/chain/tdnn_1a_all_sp exp/tri3a_lats exp/chain/tdnn_1a_all_sp/egs
steps/nnet3/chain/get_egs.sh: creating egs.  To ensure they are not deleted later you can do:  touch exp/chain/tdnn_1a_all_sp/egs/.nodelete
steps/nnet3/chain/get_egs.sh: feature type is raw
Using speaker CMVN
tree-info exp/chain/tdnn_1a_all_sp/tree
feat-to-dim scp:exp/chain/ivectors_train_all/ivector_online.scp -
steps/nnet3/chain/get_egs.sh: working out number of frames of training data
steps/nnet3/chain/get_egs.sh: working out feature dim
steps/nnet3/chain/get_egs.sh: creating 6 archives, each with 14217 egs, with
steps/nnet3/chain/get_egs.sh:   150,110,90 labels per example, and (left,right) context = (22,22)
steps/nnet3/chain/get_egs.sh: Getting validation and training subset examples in background.
steps/nnet3/chain/get_egs.sh: Generating training examples on disk


steps/nnet3/chain/get_egs.sh: recombining and shuffling order of archives on disk
... Getting subsets of validation examples for diagnostics and combination.
steps/nnet3/chain/get_egs.sh: removing temporary archives
steps/nnet3/chain/get_egs.sh: removing temporary alignments, lattices and transforms
steps/nnet3/chain/get_egs.sh: Finished preparing training examples
2019-11-04 16:44:59,299 [steps/nnet3/chain/train.py:411 - train - INFO ] Copying the properties from exp/chain/tdnn_1a_all_sp/egs to exp/chain/tdnn_1a_all_sp
2019-11-04 16:44:59,304 [steps/nnet3/chain/train.py:425 - train - INFO ] Computing the preconditioning matrix for input features
2019-11-04 16:45:48,366 [steps/nnet3/chain/train.py:434 - train - INFO ] Preparing the initial acoustic model.


2019-11-04 16:46:24,366 [steps/nnet3/chain/train.py:468 - train - INFO ] Training will run for 4.0 epochs = 24 iterations
2019-11-04 16:46:24,366 [steps/nnet3/chain/train.py:510 - train - INFO ] Iter: 0/23    Epoch: 0.00/4.0 (0.0% complete)    lr: 0.002000
queue.pl: job failed with status 143, log is in exp/chain/tdnn_1a_all_sp/log/train.0.1.log
2019-11-04 16:46:33,597 [steps/libs/common.py:235 - background_command_waiter - ERROR ] Command exited with status 1: queue.pl --gpu 1 exp/chain/tdnn_1a_all_sp/log/train.0.1.log                     nnet
3-chain-train --use-gpu=yes                      --apply-deriv-weights=False                     --l2-regularize=5e-05 --leaky-hmm-coefficient=0.1                      --write-cache=exp/chain/tdnn_1a_al
l_sp/cache.1  --xent-regularize=0.1                                          --print-interval=10 --momentum=0.0                     --max-param-change=1.41421356237                     --backstitch-trai
ning-scale=0.0                     --backstitch-training-interval=1                     --l2-regularize-factor=0.5                      --srand=0                     "nnet3-am-copy --raw=true --learning
-rate=0.002 --scale=1.0 exp/chain/tdnn_1a_all_sp/0.mdl - |nnet3-copy --edits='set-dropout-proportion name=* proportion=0.0' - - |" exp/chain/tdnn_1a_all_sp/den.fst                     "ark,bg:nnet3-chai
n-copy-egs                          --frame-shift=1                         ark:exp/chain/tdnn_1a_all_sp/egs/cegs.1.ark ark:- |                         nnet3-chain-shuffle-egs --buffer-size=5000
                 --srand=0 ark:- ark:- | nnet3-chain-merge-egs                         --minibatch-size=64 ark:- ark:- |"                     exp/chain/tdnn_1a_all_sp/1.1.raw
(kaldi) chaoyang@mobvoi-rhea-02:/export/share-nfs-1/chaoyang/am/t0$ [queue.pl: job failed with status 143, log is in exp/chain/tdnn_1a_all_sp/log/train.0.2.log



2019-11-14 11:27:30,833 [steps/nnet3/chain/train.py:485 - train - INFO ] Training will run for 4.0 epochs = 24 iterations
2019-11-14 11:27:30,833 [steps/nnet3/chain/train.py:529 - train - INFO ] Iter: 0/23   Jobs: 2   Epoch: 0.00/4.0 (0.0% complete)   lr: 0.002000
2019-11-14 11:28:37,133 [steps/nnet3/chain/train.py:529 - train - INFO ] Iter: 1/23   Jobs: 2   Epoch: 0.11/4.0 (2.8% complete)   lr: 0.001876
2019-11-14 11:29:35,876 [steps/nnet3/chain/train.py:529 - train - INFO ] Iter: 2/23   Jobs: 2   Epoch: 0.22/4.0 (5.6% complete)   lr: 0.001760
2019-11-14 11:30:34,592 [steps/nnet3/chain/train.py:529 - train - INFO ] Iter: 3/23   Jobs: 2   Epoch: 0.33/4.0 (8.3% complete)   lr: 0.001651
2019-11-14 11:31:39,208 [steps/nnet3/chain/train.py:529 - train - INFO ] Iter: 4/23   Jobs: 2   Epoch: 0.44/4.0 (11.1% complete)   lr: 0.001549
2019-11-14 11:32:34,795 [steps/nnet3/chain/train.py:529 - train - INFO ] Iter: 5/23   Jobs: 2   Epoch: 0.56/4.0 (13.9% complete)   lr: 0.001453
2019-11-14 11:33:57,685 [steps/nnet3/chain/train.py:529 - train - INFO ] Iter: 6/23   Jobs: 3   Epoch: 0.67/4.0 (16.7% complete)   lr: 0.002044
2019-11-14 11:35:08,287 [steps/nnet3/chain/train.py:529 - train - INFO ] Iter: 7/23   Jobs: 3   Epoch: 0.83/4.0 (20.8% complete)   lr: 0.001857
2019-11-14 11:36:10,068 [steps/nnet3/chain/train.py:529 - train - INFO ] Iter: 8/23   Jobs: 3   Epoch: 1.00/4.0 (25.0% complete)   lr: 0.001687
2019-11-14 11:37:21,658 [steps/nnet3/chain/train.py:529 - train - INFO ] Iter: 9/23   Jobs: 3   Epoch: 1.17/4.0 (29.2% complete)   lr: 0.001533
2019-11-14 11:38:38,331 [steps/nnet3/chain/train.py:529 - train - INFO ] Iter: 10/23   Jobs: 3   Epoch: 1.33/4.0 (33.3% complete)   lr: 0.001392
2019-11-14 11:39:51,948 [steps/nnet3/chain/train.py:529 - train - INFO ] Iter: 11/23   Jobs: 3   Epoch: 1.50/4.0 (37.5% complete)   lr: 0.001265
2019-11-14 11:41:06,449 [steps/nnet3/chain/train.py:529 - train - INFO ] Iter: 12/23   Jobs: 3   Epoch: 1.67/4.0 (41.7% complete)   lr: 0.001149
2019-11-14 11:42:05,144 [steps/nnet3/chain/train.py:529 - train - INFO ] Iter: 13/23   Jobs: 3   Epoch: 1.83/4.0 (45.8% complete)   lr: 0.001044
2019-11-14 11:43:21,764 [steps/nnet3/chain/train.py:529 - train - INFO ] Iter: 14/23   Jobs: 3   Epoch: 2.00/4.0 (50.0% complete)   lr: 0.000949
2019-11-14 11:44:35,383 [steps/nnet3/chain/train.py:529 - train - INFO ] Iter: 15/23   Jobs: 3   Epoch: 2.17/4.0 (54.2% complete)   lr: 0.000862
2019-11-14 11:45:52,021 [steps/nnet3/chain/train.py:529 - train - INFO ] Iter: 16/23   Jobs: 3   Epoch: 2.33/4.0 (58.3% complete)   lr: 0.000783
2019-11-14 11:46:51,555 [steps/nnet3/chain/train.py:529 - train - INFO ] Iter: 17/23   Jobs: 3   Epoch: 2.50/4.0 (62.5% complete)   lr: 0.000711
2019-11-14 11:48:08,205 [steps/nnet3/chain/train.py:529 - train - INFO ] Iter: 18/23   Jobs: 4   Epoch: 2.67/4.0 (66.7% complete)   lr: 0.000862
2019-11-14 11:49:21,854 [steps/nnet3/chain/train.py:529 - train - INFO ] Iter: 19/23   Jobs: 4   Epoch: 2.89/4.0 (72.2% complete)   lr: 0.000758
2019-11-14 11:50:35,465 [steps/nnet3/chain/train.py:529 - train - INFO ] Iter: 20/23   Jobs: 4   Epoch: 3.11/4.0 (77.8% complete)   lr: 0.000667
2019-11-14 11:51:55,236 [steps/nnet3/chain/train.py:529 - train - INFO ] Iter: 21/23   Jobs: 4   Epoch: 3.33/4.0 (83.3% complete)   lr: 0.000587
2019-11-14 11:53:05,992 [steps/nnet3/chain/train.py:529 - train - INFO ] Iter: 22/23   Jobs: 4   Epoch: 3.56/4.0 (88.9% complete)   lr: 0.000517
2019-11-14 11:54:28,671 [steps/nnet3/chain/train.py:529 - train - INFO ] Iter: 23/23   Jobs: 4   Epoch: 3.78/4.0 (94.4% complete)   lr: 0.000400
2019-11-14 11:56:06,535 [steps/nnet3/chain/train.py:585 - train - INFO ] Doing final combination to produce final.mdl
2019-11-14 11:56:06,536 [steps/libs/nnet3/train/chain_objf/acoustic_model.py:571 - combine_models - INFO ] Combining set([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]) models.
2019-11-14 11:56:32,948 [steps/nnet3/chain/train.py:614 - train - INFO ] Cleaning up the experiment directory exp/chain/tdnn_1a_all_sp
steps/nnet2/remove_egs.sh: Finished deleting examples in exp/chain/tdnn_1a_all_sp/egs
exp/chain/tdnn_1a_all_sp: num-iters=24 nj=2..4 num-params=18.1M dim=43+100->2072 combine=-0.044->-0.044 (over 1) xent:train/valid[15,23]=(-0.864,-0.653/-1.00,-0.853) logprob:train/valid[15,23]=(-0.054,-0.040/-0.071,-0.066)
tree-info exp/chain/tdnn_1a_all_sp/tree
tree-info exp/chain/tdnn_1a_all_sp/tree
fstcomposecontext --context-size=2 --central-position=1 --read-disambig-syms=data/lang_test/phones/disambig.int --write-disambig-syms=data/lang_test/tmp/disambig_ilabels_2_1.int data/lang_test/tmp/ilabels_2_1.26358 data/lang_test/tmp/LG.fst
fstisstochastic data/lang_test/tmp/CLG_2_1.fst
-0.0352461 -0.0360721
[info]: CLG not stochastic.
make-h-transducer --disambig-syms-out=exp/chain/tdnn_1a_all_sp/graph/disambig_tid.int --transition-scale=1.0 data/lang_test/tmp/ilabels_2_1 exp/chain/tdnn_1a_all_sp/tree exp/chain/tdnn_1a_all_sp/final.mdl
fstrmsymbols exp/chain/tdnn_1a_all_sp/graph/disambig_tid.int
fstminimizeencoded
fstrmepslocal
fstdeterminizestar --use-log=true
fsttablecompose exp/chain/tdnn_1a_all_sp/graph/Ha.fst data/lang_test/tmp/CLG_2_1.fst
fstisstochastic exp/chain/tdnn_1a_all_sp/graph/HCLGa.fst
0.013628 -0.137429
HCLGa is not stochastic
add-self-loops --self-loop-scale=1.0 --reorder=true exp/chain/tdnn_1a_all_sp/final.mdl exp/chain/tdnn_1a_all_sp/graph/HCLGa.fst
fstisstochastic exp/chain/tdnn_1a_all_sp/graph/HCLG.fst
0.00679079 -0.0722656
[info]: final HCLG is not stochastic.
steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --nj 540 --cmd queue.pl --online-ivector-dir exp/chain/ivectors_dev_all exp/chain/tdnn_1a_all_sp/graph data/dev_hires exp/chain/tdnn_1a_all_sp/decode_dev
steps/nnet3/decode.sh: feature type is raw
queue.pl: 112 / 540 failed, log is in exp/chain/tdnn_1a_all_sp/decode_dev/log/decode.*.log


steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --nj 540 --cmd queue.pl --online-ivector-dir exp/chain/ivectors_test_all exp/chain/tdnn_1a_all_sp/graph data/test_hires exp/chain/tdnn_1a_all_sp/
decode_test
steps/nnet3/decode.sh: feature type is raw
steps/diagnostic/analyze_lats.sh --cmd queue.pl --iter final exp/chain/tdnn_1a_all_sp/graph exp/chain/tdnn_1a_all_sp/decode_test
Overall, lattice depth (10,50,90-percentile)=(1,2,64) and mean=24.3
steps/diagnostic/analyze_lats.sh: see stats in exp/chain/tdnn_1a_all_sp/decode_test/log/analyze_lattice_depth_stats.log
score best paths
+ steps/score_kaldi.sh --cmd queue.pl data/test_hires exp/chain/tdnn_1a_all_sp/graph exp/chain/tdnn_1a_all_sp/decode_test
steps/score_kaldi.sh --cmd queue.pl data/test_hires exp/chain/tdnn_1a_all_sp/graph exp/chain/tdnn_1a_all_sp/decode_test
steps/score_kaldi.sh: scoring with word insertion penalty=0.0,0.5,1.0
+ steps/scoring/score_kaldi_cer.sh --stage 2 --cmd queue.pl data/test_hires exp/chain/tdnn_1a_all_sp/graph exp/chain/tdnn_1a_all_sp/decode_test
steps/scoring/score_kaldi_cer.sh --stage 2 --cmd queue.pl data/test_hires exp/chain/tdnn_1a_all_sp/graph exp/chain/tdnn_1a_all_sp/decode_test
steps/scoring/score_kaldi_cer.sh: scoring with word insertion penalty=0.0,0.5,1.0
+ echo 'local/score.sh: Done'
local/score.sh: Done
score confidence and timing with sclite
Decoding done.
local/chain/run_tdnn.sh succeeded