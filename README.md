# LSGAN for SVHN

This repository contains Python 3 routines that train two Least Squares Generative Adversarial Networks - LSGAN, for Google Street View House Number - SVHN database. The LSGAN architectures herein used are those described in [1]. One routine that generate samples from one of the models is also avaliable.

# File description

LSGAN_v1.py /LSGAN_v2.py are used to train first/second architecture of [1].
LSGAN_generate.py is used to generate samples from model LSGAN_v2.py after its training (so, use above mentioned routine before this).

Make sure 'train_32x32.mat' [2] is in the same folder. 20% of this file was used due to computational limitations. It's encouraged to use full 'train_32x32.mat'.

# Results

This repository contains the best results shown in the video: https://youtu.be/-3Lr06syV2U

# References

[1] Xudong Mao, Qing Liy, Haoran Xiez, Raymond Y. K. Laux, Zhen Wang, and Stephen Paul Smolley. Least squares generative adversarial
    networks. arXiv:1611.04076v3 [cs.CV], 2017.

[2] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y. Ng. The Street View House Numbers (SVHN) Dataset,           http://ufldl.stanford.edu/housenumbers/
