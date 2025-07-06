#!/bin/bash
. ./path.sh

# Define your data dir
data_root=/data/speechocean762
dest_dir=data/s5

# Generate Kaldi data directories
python3 local/prepare_data.py $data_root $dest_dir || exit 1

# Validate
utils/validate_data_dir.sh --no-feats $dest_dir || exit 1
