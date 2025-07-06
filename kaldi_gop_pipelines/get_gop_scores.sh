#!/bin/bash
. ./path.sh

# Extract posterior scores and calculate GOP
local/get_gop.sh data/s5 data/lang exp/nnet2_tanh exp/s5_ali exp/s5_gop || exit 1

# Export per-phone log-likelihoods to ark or numpy
python3 local/convert_gop_to_numpy.py exp/s5_gop/gop.ark exp/s5_gop/
