> in gopt directory here, mkdir a folder named "seq_data_librispeech"
> Then copy the data gotten from Kaldi there in below format

```bash
seq_data_librispeech/
├── te_feat.npy
├── te_label_phn.npy
├── te_label_utt.npy
├── te_label_word.npy
├── tr_feat.npy
├── tr_label_phn.npy
├── tr_label_utt.npy
└── tr_label_word.npy
```

> Then run the notebook with gpu access