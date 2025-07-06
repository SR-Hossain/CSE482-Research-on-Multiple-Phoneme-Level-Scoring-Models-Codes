#!/bin/bash
. ./path.sh

data=data/s5
mfccdir=mfcc

steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 4 --cmd run.pl \
    $data exp/make_mfcc $mfccdir || exit 1

steps/compute_cmvn_stats.sh $data exp/make_mfcc $mfccdir || exit 1
utils/fix_data_dir.sh $data
