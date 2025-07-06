import kaldiio
import numpy as np

ark_path = 'exp/s5_gop/gop.ark'
output_dir = 'exp/s5_gop/'

gop_feats = []
utt_ids = []

for utt_id, gop_vec in kaldiio.load_ark(ark_path):
    gop_feats.append(gop_vec)
    utt_ids.append(utt_id)

np.save(f'{output_dir}/gop_feat.npy', gop_feats)
np.save(f'{output_dir}/utt_ids.npy', utt_ids)
