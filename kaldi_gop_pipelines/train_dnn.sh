#!/bin/bash
. ./path.sh

# Assumes you already trained a monophone + triphone system
# Now train DNN model on top of alignments (e.g., from BASR18)

steps/nnet2/train_tanh.sh \
  --num-epochs 5 --num-jobs-nnet 2 \
  --hidden-layer-dim 512 \
  data/train data/lang exp/tri3_ali exp/nnet2_tanh || exit 1
