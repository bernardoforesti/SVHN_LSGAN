"""
PyTorch implementation of LSGAN as second architecture of paper [1], but for SVHN.
This architecture, after training, allows generation of specified class.

[1] Xudong Mao, Qing Liy, Haoran Xiez, Raymond Y. K. Laux, Zhen Wang, and Stephen Paul
    Smolley. Least squares generative adversarial networks. arXiv:1611.04076v3 [cs.CV], 2017.

author: Bernardo Foresti (please cite)
"""
import os

import numpy as np
import torch
import torch.nn as nn

from lsgan_common import (Conv, ConvTranspose, FullyConnected, create, fetch_data, one_hot_labels,
                          get_device, leaky_relu, sample_z, save_imgs)

N_INPUTS_G = 64   # Shape of random input


class Generator(nn.Module):
    def __init__(self, n_classes=10, n_inputs=N_INPUTS_G):
        super().__init__()
        self.hidden1 = FullyConnected(n_inputs + n_classes, 8 * 8 * 128, batch_norm=True)
        self.hidden2 = ConvTranspose(128, 128, 5, 2, batch_norm=True)
        self.output = ConvTranspose(128, 3, 5, 2)

    def forward(self, z, y):
        h = torch.relu(self.hidden1(torch.cat([z, y], dim=1)))
        h = h.reshape(-1, 128, 8, 8)
        h = torch.relu(self.hidden2(h))
        return torch.tanh(self.output(h))


class Discriminator(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.hidden1 = Conv(3, 256, 5, 2)
        self.hidden2 = Conv(256 + n_classes, 320, 5, 2, batch_norm=True)
        self.hidden3 = FullyConnected((320 + n_classes) * 8 * 8, 1024, batch_norm=True)
        self.output = FullyConnected(1024 + n_classes, 1)

    @staticmethod
    def _tile(y, size):
        # (N, n_classes) -> (N, n_classes, size, size), as y * tf.ones(...)
        return y[:, :, None, None].expand(-1, -1, size, size)

    def forward(self, x, y):
        h = leaky_relu(self.hidden1(x))
        h = torch.cat([h, self._tile(y, 16)], dim=1)
        h = leaky_relu(self.hidden2(h))
        h = torch.cat([h, self._tile(y, 8)], dim=1)
        h = leaky_relu(self.hidden3(h.reshape(h.shape[0], -1)))
        return self.output(torch.cat([h, y], dim=1))


def train(X_tr, y_tr, n_steps=10000, mb_size=32, learning_rate_D=0.001, learning_rate_G=0.001,
          print_steps=10, ckpt_path='LSGAN_v2.pt', out_dir='.', device=None):
    """Trains the conditional LSGAN. X_tr is (m, 3, 32, 32) in [-1, 1]; y_tr is one-hot
    (m, n_classes). Returns (G, D)."""
    device = device or get_device()
    m = X_tr.shape[0]
    n_classes = y_tr.shape[1]
    n_batches = int(np.floor(m / mb_size))

    G, D = Generator(n_classes).to(device), Discriminator(n_classes).to(device)
    optimizer_D = torch.optim.Adam([p for p in D.parameters() if p.requires_grad],
                                   lr=learning_rate_D, betas=(0.5, 0.999))
    optimizer_G = torch.optim.Adam([p for p in G.parameters() if p.requires_grad],
                                   lr=learning_rate_G, betas=(0.5, 0.999))

    # Check if it's a resume run (inside if) or a new run
    resume_step = 0
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        G.load_state_dict(ckpt['G'])
        D.load_state_dict(ckpt['D'])
        optimizer_G.load_state_dict(ckpt['optimizer_G'])
        optimizer_D.load_state_dict(ckpt['optimizer_D'])
        resume_step = ckpt['step'] + 1   # last saved step + 1

    for step in range(resume_step, n_steps):
        print('step:', step)
        np.random.seed(step)
        img = np.random.permutation(X_tr)
        np.random.seed(step)
        y_img = np.random.permutation(y_tr)

        for batch_index in range(n_batches):
            print('batch:', batch_index, 'of:', n_batches)
            # Train Discriminator: -------------------------------------------------------
            x_in, y_in = fetch_data(img, y_img, batch_index, mb_size)
            x_in = torch.from_numpy(x_in).to(device)
            y_in = torch.from_numpy(y_in.astype(np.float32)).to(device)
            z_in = sample_z(mb_size, N_INPUTS_G, device)
            D1 = D(x_in, y_in)
            D2 = D(G(z_in, y_in).detach(), y_in)
            loss_D = 0.5 * (torch.mean((D1 - 1) ** 2) + torch.mean(D2 ** 2))
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
            # Train Generator: -----------------------------------------------------------
            z_in = sample_z(mb_size, N_INPUTS_G, device)
            D2 = D(G(z_in, y_in), y_in)
            loss_G = 0.5 * torch.mean((D2 - 1) ** 2)
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

        # Print loss each 'print_steps' steps
        if step % print_steps == 0:
            with torch.no_grad():
                G_new = G(z_in, y_in)
                accuracy = torch.mean(D(G_new, y_in)).item()
            save_imgs(G_new[:25].cpu().numpy(), 5, "LSGAN_v2 step: " + str(step),
                      os.path.join(out_dir, 'LSGAN_v2_step' + str(step) + '.png'))
            print("%d [D loss: %f, mean_D: %.2f%%] [G loss: %f]"
                  % (step, loss_D.item(), 100 * accuracy, loss_G.item()))
            torch.save({'G': G.state_dict(), 'D': D.state_dict(),
                        'optimizer_G': optimizer_G.state_dict(),
                        'optimizer_D': optimizer_D.state_dict(),
                        'n_classes': n_classes, 'step': step}, ckpt_path)
    return G, D


def main(mat_file='test_32x32.mat'):
    # Loads dataset and uses just a part of it
    X_tr_full, y_tr_full = create(mat_file)
    ratio_tr = int(0.2 * X_tr_full.shape[0])
    # y_tr as an array with shape=(m x n_classes)
    y_tr = one_hot_labels(y_tr_full[0:ratio_tr])
    train(X_tr_full[0:ratio_tr], y_tr)


if __name__ == '__main__':
    main()
