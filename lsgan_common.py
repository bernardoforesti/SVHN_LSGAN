"""
Shared helpers for the PyTorch translation of the SVHN LSGAN routines.

Layers reproduce the behaviour of the TF-slim layers used in the original code
(see legacy/): 'SAME' padding (including its asymmetric split), Xavier-uniform
weights, zero biases, and batch norm with epsilon=1e-3 and no learned scale.

author: Bernardo Foresti (please cite)
"""
import math

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device():
    """Uses the GPU if there is one, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def leaky_relu(z, alpha=0.2):
    return torch.maximum(alpha * z, z)


def sample_z(m, n, device=None):
    """Creates a m x n random matrix with data between -1 and +1."""
    z = np.random.uniform(-1., 1., size=[m, n]).astype(np.float32)
    return torch.from_numpy(z).to(device)


def create(file_name):
    """Loads the SVHN .mat file. Returns X as (N, 3, 32, 32) in [-1, 1] and y."""
    dataset = spio.loadmat(file_name, squeeze_me=True)
    X_tr = np.transpose(dataset['X'], axes=(3, 2, 0, 1))  # NCHW (original: NHWC)
    y_tr = dataset['y']
    X_tr = X_tr / 127.5 - 1.
    return X_tr.astype(np.float32), y_tr


def fetch_data(data_X, data_y, batch_index, batch_size):
    """Creates batches for a given 'data'."""
    begin = batch_index * batch_size
    end = (batch_index + 1) * batch_size
    return data_X[begin:end], data_y[begin:end]


def save_imgs(samples, n_images, title, filename):
    """Saves a frame of n_images x n_images. 'samples' is a (N, 3, H, W) array in [-1, 1]."""
    samples = np.transpose(np.asarray(samples), (0, 2, 3, 1))
    fig = plt.figure(figsize=(n_images, n_images))
    fig.suptitle(title)
    gs = gridspec.GridSpec(n_images, n_images)
    gs.update(wspace=0.05, hspace=0.05)
    for i in range(n_images ** 2):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        img = (samples[i] + 1.) * 127.5
        plt.imshow(np.clip(img, 0, 255).astype(np.uint8))
    plt.savefig(filename)
    plt.close()
    return fig


def _same_pad(n_in, k, s):
    """TensorFlow 'SAME' padding for one dimension: returns (before, after)."""
    n_out = math.ceil(n_in / s)
    total = max((n_out - 1) * s + k - n_in, 0)
    return total // 2, total - total // 2


def _init(layer):
    nn.init.xavier_uniform_(layer.weight)  # slim default initializer
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class BatchNorm(nn.Module):
    """slim.batch_norm defaults: batch statistics (is_training=True), eps=1e-3,
    beta learned, gamma fixed at 1 (scale=False). Works for 2D (N,C) and 4D inputs."""

    def __init__(self, channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels, eps=1e-3)
        self.bn.weight.requires_grad_(False)

    def forward(self, x):
        if x.dim() == 2:
            return self.bn(x[:, :, None, None])[:, :, 0, 0]
        return self.bn(x)


class Conv(nn.Module):
    """slim.convolution (SAME padding). With batch norm there is no bias."""

    def __init__(self, c_in, c_out, k, stride, batch_norm=False):
        super().__init__()
        self.k, self.stride = k, stride
        self.conv = nn.Conv2d(c_in, c_out, k, stride, bias=not batch_norm)
        _init(self.conv)
        self.bn = BatchNorm(c_out) if batch_norm else None

    def forward(self, x):
        top, bottom = _same_pad(x.shape[2], self.k, self.stride)
        left, right = _same_pad(x.shape[3], self.k, self.stride)
        x = self.conv(F.pad(x, (left, right, top, bottom)))
        return self.bn(x) if self.bn is not None else x


class ConvTranspose(nn.Module):
    """slim.convolution2d_transpose (SAME padding, output = stride * input)."""

    def __init__(self, c_in, c_out, k, stride, batch_norm=False):
        super().__init__()
        self.k, self.stride = k, stride
        self.conv = nn.ConvTranspose2d(c_in, c_out, k, stride, bias=not batch_norm)
        _init(self.conv)
        self.bn = BatchNorm(c_out) if batch_norm else None

    def forward(self, x):
        h_out, w_out = x.shape[2] * self.stride, x.shape[3] * self.stride
        x = self.conv(x)
        # TF crops the 'full' transposed convolution output using SAME padding
        top = _same_pad(h_out, self.k, self.stride)[0]
        left = _same_pad(w_out, self.k, self.stride)[0]
        x = x[:, :, top:top + h_out, left:left + w_out]
        return self.bn(x) if self.bn is not None else x


class FullyConnected(nn.Module):
    """slim.fully_connected. With batch norm there is no bias."""

    def __init__(self, n_in, n_out, batch_norm=False):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out, bias=not batch_norm)
        _init(self.fc)
        self.bn = BatchNorm(n_out) if batch_norm else None

    def forward(self, x):
        x = self.fc(x)
        return self.bn(x) if self.bn is not None else x
