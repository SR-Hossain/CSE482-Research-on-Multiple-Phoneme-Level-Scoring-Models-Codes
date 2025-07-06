#!/bin/bash
. ./path.sh

# Align S5 data using trained DNN model
steps/nnet2/align.sh \
    --nj 4 --cmd run.pl \
    data/s5 data/lang exp/nnet2_tanh exp/s5_ali || exit 1
