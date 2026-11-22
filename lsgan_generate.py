"""
Generates samples based on a trained model (use lsgan_v2.py to train, before using this
file) based on second architecture of paper [1], for SVHN.

[1] Xudong Mao, Qing Liy, Haoran Xiez, Raymond Y. K. Laux, Zhen Wang, and Stephen Paul
    Smolley. Least squares generative adversarial networks. arXiv:1611.04076v3 [cs.CV], 2017.

author: Bernardo Foresti (please cite)
"""
import numpy as np
import torch

from lsgan_common import create, get_device, one_hot_labels, sample_z, save_imgs
from lsgan_v2 import N_INPUTS_G, Generator


class create_y(object):
    """
    Creates one-hot vectors representing the targeted labels to be created by the
    Generator Net (G) model.
    """
    def __init__(self, mb_size):
        self.mb = mb_size
        self.classes = np.array(range(10))
        self.n = self.classes.shape[0]

    def all_class(self):
        """Given a mb_size, creates int(mb_size/n) one-hot label tags for each class."""
        small_mb = int(self.mb / self.n)
        y = np.repeat(self.classes, small_mb).reshape(-1, 1)
        return one_hot_labels(y, self.n)


def generate(ckpt_path='LSGAN_v2.pt', n_samples=None, tag='mixed', device=None):
    """Samples from the trained generator and saves a 5 x 5 frame. Returns (G_new, y_in)."""
    device = device or get_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    G = Generator(ckpt.get('n_classes', 10)).to(device)
    G.load_state_dict(ckpt['G'])
    # Kept in train mode on purpose: the original model always normalized with batch
    # statistics (moving averages were never updated), so generation does the same.

    y_in = create_y(n_samples).all_class()
    mb_size = y_in.shape[0]
    z_in = sample_z(mb_size, N_INPUTS_G, device)
    with torch.no_grad():
        G_new = G(z_in, torch.from_numpy(y_in.astype(np.float32)).to(device)).cpu().numpy()
    G_new_shuffle = np.random.permutation(G_new)
    save_imgs(G_new_shuffle, 5, "LSGAN Final, number:" + str(tag), 'LSGAN_v2_tag' + str(tag) + '.png')

    # Uncomment to save generated samples
    # np.save('X_model', G_new)
    # np.save('y_model', y_in)
    return G_new, y_in


def main(mat_file='test_32x32.mat'):
    X_tr_full, _ = create(mat_file)
    ratio_tr = int(0.2 * X_tr_full.shape[0])
    del X_tr_full
    generate(n_samples=ratio_tr)


if __name__ == '__main__':
    main()
