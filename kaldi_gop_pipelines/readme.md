# Kaldi GOP Pipeline (GOP-NN Scoring with GOPT)

This repository contains all scripts and steps used to compute phoneme-level Goodness of Pronunciation (GOP) scores using Kaldi and train a neural post-processing model (GOPT).

---
Use BASR18, and 50-60 hours of dataset from librispeech.

| Component   | Recommended                                     |
| ----------- | ----------------------------------------------- |
| **CPU**     | 8-core (e.g., Ryzen 7 5700X or Intel i7-12700K) |
| **RAM**     | 32 GB                                           |
| **Storage** | 500 GB SSD (1 TB if storing raw + processed)    |
| **GPU**     | NVIDIA RTX 3060 / 4060 (12 GB VRAM)             |
| **OS**      | Ubuntu 20.04 or 22.04 LTS                       |
| **CUDA**    | ≥ 11.3 (for GOPT PyTorch GPU support)           |



## 🔧 1. Install Kaldi

> 📍 Tested with Kaldi commit `bfb038d`

```bash
git clone https://github.com/kaldi-asr/kaldi.git
cd kaldi/tools
extras/check_dependencies.sh
make -j$(nproc)

cd ../src
./configure --shared
make depend -j$(nproc)
make -j$(nproc)
````

---

## 2. Directory Setup

Clone this repository into your Kaldi `egs` folder:

```bash
cd kaldi/egs/
git clone <this_repo_url> kaldi_gop_pipeline
cd kaldi_gop_pipeline
```

Directory layout:

```
kaldi_gop_pipeline/
├── data_prep.sh
├── feature_extract.sh
├── train_dnn.sh
├── align.sh
├── get_gop_scores.sh
├── local/
│   ├── prepare_data.py
│   ├── get_gop.sh
├── conf/
│   └── mfcc.conf
└── data/, exp/, mfcc/, utils/, steps/, path.sh
```

---

## 3. Download BASR18 dataset in a directory and set the data location to it

## 4. Run egs/librispeech/s5/run.sh to compute MFCC features and CMVN stats.

## 5. Train Acoustic Model (if not already trained)

> If you already have a DNN model (e.g., from BASR18), skip this step.

```bash
bash 3_train_dnn.sh
```

This script assumes mono and tri-phone models are already trained.

---

## 6. Align Utterances

```bash
bash 4_align.sh
```

This aligns your test data (S5 subset) using the trained model.

---

## 7. Compute GOP Scores using egs/s5/gop_speechocean762/run.sh

This generates `.ark` GOP files and converts them to `.npy`:

```
exp/s5_gop/
├── gop.ark
├── gop_feat.npy
└── utt_ids.npy
```

---

## 🤖 8. Post-process with GOPT (Optional Neural Classifier)

If you want to use GOPT:

1. Prepare `.npy` files:

   * `tr_feat.npy`, `tr_label_phn.npy`, ...
   * `te_feat.npy`, `te_label_phn.npy`, ...

2. Run training:

```bash
python train_gopt.py --exp_dir results/kaldi_gopt
```

3. Predictions will be saved as:

```
├── phn_pred.npy
├── word_pred.npy
├── utt_pred.npy
```

---

## 🧾 Notes

* All parameters (epochs, layers, features) can be adjusted in `train_gopt.py`.

---

## 📚 References

* Kaldi GOP: [https://github.com/kaldi-asr/kaldi/tree/master/egs/gop](https://github.com/kaldi-asr/kaldi/tree/master/egs/gop)
* GOPT: Adapted from \[Li et al., INTERSPEECH 2019]

