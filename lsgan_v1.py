"""
PyTorch implementation of LSGAN as first architecture of paper [1], but for SVHN.
This architecture, after training, does not allow generation of specified class.

[1] Xudong Mao, Qing Liy, Haoran Xiez, Raymond Y. K. Laux, Zhen Wang, and Stephen Paul
    Smolley. Least squares generative adversarial networks. arXiv:1611.04076v3 [cs.CV], 2017.

author: Bernardo Foresti (please cite)
"""
import os

import numpy as np
import torch
import torch.nn as nn

from lsgan_common import (Conv, ConvTranspose, FullyConnected, create, fetch_data,
                          get_device, leaky_relu, sample_z, save_imgs)

N_INPUTS_G = 64   # Shape of random input


class Generator(nn.Module):
    def __init__(self, n_inputs=N_INPUTS_G):
        super().__init__()
        self.input_G = FullyConnected(n_inputs, 2 * 2 * 256, batch_norm=True)
        self.conv1 = ConvTranspose(256, 256, 3, 2, batch_norm=True)
        self.conv2 = ConvTranspose(256, 256, 3, 1, batch_norm=True)
        self.conv3 = ConvTranspose(256, 256, 3, 2, batch_norm=True)
        self.conv4 = ConvTranspose(256, 256, 3, 1, batch_norm=True)
        self.conv5 = ConvTranspose(256, 128, 3, 2, batch_norm=True)
        self.conv6 = ConvTranspose(128, 64, 3, 2, batch_norm=True)
        self.conv7 = ConvTranspose(64, 3, 3, 1)

    def forward(self, z):
        h = self.input_G(z).reshape(-1, 256, 2, 2)
        for conv in (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6):
            h = torch.relu(conv(h))
        return torch.tanh(self.conv7(h))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv(3, 64, 5, 2)
        self.conv2 = Conv(64, 128, 5, 2, batch_norm=True)
        self.conv3 = Conv(128, 256, 5, 2, batch_norm=True)
        self.conv4 = Conv(256, 512, 5, 2, batch_norm=True)
        self.fc1 = FullyConnected(2 * 2 * 512, 1)

    def forward(self, x):
        h = x
        for conv in (self.conv1, self.conv2, self.conv3, self.conv4):
            h = leaky_relu(conv(h))
        return self.fc1(h.reshape(h.shape[0], -1))


def train(X_tr, n_steps=10000, mb_size=128, learning_rate_D=0.001, learning_rate_G=0.001,
          print_steps=10, ckpt_path='LSGAN_v1.pt', out_dir='.', device=None):
    """Trains the LSGAN. X_tr is a (m, 3, 32, 32) array in [-1, 1]. Returns (G, D)."""
    device = device or get_device()
    m = X_tr.shape[0]
    n_batches = int(np.floor(m / mb_size))

    G, D = Generator().to(device), Discriminator().to(device)
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

        for batch_index in range(n_batches):
            print('batch:', batch_index, 'of:', n_batches)
            # Train Discriminator: -------------------------------------------------------
            x_in, _ = fetch_data(img, img, batch_index, mb_size)
            x_in = torch.from_numpy(x_in).to(device)
            z_in = sample_z(mb_size, N_INPUTS_G, device)
            D1 = D(x_in)
            D2 = D(G(z_in).detach())
            loss_D = 0.5 * (torch.mean((D1 - 1) ** 2) + torch.mean(D2 ** 2))
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
            # Train Generator: -----------------------------------------------------------
            z_in = sample_z(mb_size, N_INPUTS_G, device)
            D2 = D(G(z_in))
            loss_G = 0.5 * torch.mean((D2 - 1) ** 2)
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

        # Print loss each 'print_steps' steps
        if step % print_steps == 0:
            with torch.no_grad():
                G_new = G(z_in)
                accuracy = torch.mean(D(G_new)).item()
            save_imgs(G_new[:25].cpu().numpy(), 5, "LSGAN_v1 step: " + str(step),
                      os.path.join(out_dir, 'LSGAN_v1_step' + str(step) + '.png'))
            print("%d [D loss: %f, mean_D: %.2f%%] [G loss: %f]"
                  % (step, loss_D.item(), 100 * accuracy, loss_G.item()))
            torch.save({'G': G.state_dict(), 'D': D.state_dict(),
                        'optimizer_G': optimizer_G.state_dict(),
                        'optimizer_D': optimizer_D.state_dict(),
                        'step': step}, ckpt_path)
    return G, D


def main(mat_file='test_32x32.mat'):
    # Loads dataset and uses just a part of it
    X_tr_full, _ = create(mat_file)
    ratio_tr = int(0.2 * X_tr_full.shape[0])
    train(X_tr_full[0:ratio_tr])


if __name__ == '__main__':
    main()
