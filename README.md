# LSGAN for SVHN

This repository contains Python 3 routines (tensorflow) that train two Least Squares Generative Adversarial Networks - LSGAN, for Google Street View House Number - SVHN database. The LSGAN architectures herein used are those described in [1]. One routine that generate samples from one of the models is also avaliable.

# File description

The code is written in **PyTorch** (it runs on the GPU automatically if one is available, otherwise on the CPU).

- `lsgan_v1.py` / `lsgan_v2.py` train the first/second architecture of [1] (v1 is unconditional; v2 is conditional, so the generated digit can be chosen).
- `lsgan_generate.py` generates samples from the model trained by `lsgan_v2.py` (train it first).
- `lsgan_common.py` holds shared helpers (data loading, TF-slim-like layers, plotting).
- `tests/` has a short CPU smoke test using synthetic data.
- `legacy/` keeps the original TensorFlow 1.x scripts (`LSGAN_v1.py`, `LSGAN_v2.py`, `LSGAN_generate.py`), unchanged.

# How to run

```
pip install -r requirements.txt
python lsgan_v1.py          # or: python lsgan_v2.py
python lsgan_generate.py    # after lsgan_v2.py
python -m pytest tests      # quick sanity check, no dataset needed
```

Make sure 'test_32x32.mat' [2] is in the same folder. Only 20% of this file was used for training due to computational limitations. It's encouraged to use full 'train_32x32.mat'.

Models are saved with `torch.save` to `LSGAN_v1.pt` / `LSGAN_v2.pt` (generator, discriminator, optimizers and last step) and loaded with `torch.load`; if the file exists, training resumes from it.

# Results

This repository contains the best results shown in the video: https://youtu.be/-3Lr06syV2U

# References

[1] Xudong Mao, Qing Liy, Haoran Xiez, Raymond Y. K. Laux, Zhen Wang, and Stephen Paul Smolley. Least squares generative adversarial
    networks. arXiv:1611.04076v3 [cs.CV], 2017.

[2] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y. Ng. The Street View House Numbers (SVHN) Dataset,           http://ufldl.stanford.edu/housenumbers/
