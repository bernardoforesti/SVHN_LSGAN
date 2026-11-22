"""Short CPU smoke tests with synthetic data (no real SVHN file needed)."""
import os
import sys

import numpy as np
import scipy.io as spio
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lsgan_common  # noqa: E402
import lsgan_generate  # noqa: E402
import lsgan_v1  # noqa: E402
import lsgan_v2  # noqa: E402

CPU = torch.device('cpu')


def synthetic_mat(path, n=64):
    rng = np.random.RandomState(0)
    X = rng.randint(0, 256, size=(32, 32, 3, n)).astype(np.uint8)
    y = rng.randint(1, 11, size=(n, 1)).astype(np.uint8)
    spio.savemat(path, {'X': X, 'y': y})


def test_load_mat(tmp_path):
    path = str(tmp_path / 'syn.mat')
    synthetic_mat(path)
    X, y = lsgan_common.create(path)
    assert X.shape == (64, 3, 32, 32) and y.shape == (64,)
    assert X.min() >= -1 and X.max() <= 1


def test_shapes():
    z = torch.zeros(4, 64)
    y = torch.eye(10)[:4]
    assert lsgan_v1.Generator()(z).shape == (4, 3, 32, 32)
    assert lsgan_v1.Discriminator()(torch.zeros(4, 3, 32, 32)).shape == (4, 1)
    assert lsgan_v2.Generator()(z, y).shape == (4, 3, 32, 32)
    assert lsgan_v2.Discriminator()(torch.zeros(4, 3, 32, 32), y).shape == (4, 1)


def test_v1_train_short(tmp_path):
    path = str(tmp_path / 'syn.mat')
    synthetic_mat(path)
    X, _ = lsgan_common.create(path)
    ckpt = str(tmp_path / 'v1.pt')
    G, D = lsgan_v1.train(X, n_steps=2, mb_size=32, print_steps=1, ckpt_path=ckpt,
                          out_dir=str(tmp_path), device=CPU)
    assert os.path.isfile(ckpt) and os.path.isfile(str(tmp_path / 'LSGAN_v1_step0.png'))
    with torch.no_grad():
        out = G(torch.zeros(4, 64))
    assert torch.isfinite(out).all()


def test_v2_train_resume_and_generate(tmp_path, monkeypatch):
    path = str(tmp_path / 'syn.mat')
    synthetic_mat(path)
    X, y = lsgan_common.create(path)
    y = lsgan_common.one_hot_labels(y)
    ckpt = str(tmp_path / 'v2.pt')
    lsgan_v2.train(X, y, n_steps=1, mb_size=32, print_steps=1, ckpt_path=ckpt,
                   out_dir=str(tmp_path), device=CPU)
    # resumes from the saved checkpoint (step 0 -> continues with step 1)
    lsgan_v2.train(X, y, n_steps=2, mb_size=32, print_steps=1, ckpt_path=ckpt,
                   out_dir=str(tmp_path), device=CPU)
    assert torch.load(ckpt)['step'] == 1

    monkeypatch.chdir(tmp_path)
    G_new, y_in = lsgan_generate.generate(ckpt_path=ckpt, n_samples=50, device=CPU)
    assert G_new.shape == (50, 3, 32, 32) and y_in.shape == (50, 10)
    assert os.path.isfile(str(tmp_path / 'LSGAN_v2_tagmixed.png'))


def test_label_mapping():
    # SVHN: digit 0 has label 10; column k of the one-hot must be digit k
    oh = lsgan_common.one_hot_labels(np.array([10, 1, 3, 9]))
    assert oh.shape == (4, 10)
    assert oh[0].argmax() == 0 and oh[1].argmax() == 1
    assert oh[2].argmax() == 3 and oh[3].argmax() == 9
    assert (oh.sum(axis=1) == 1).all()
    # always 10 columns, even if some digits are missing
    assert lsgan_common.one_hot_labels(np.array([[2], [2]])).shape == (2, 10)


def test_generate_labels_match_digits():
    y = lsgan_generate.create_y(20).all_class()
    assert y.shape == (20, 10)
    assert (y.argmax(axis=1) == np.repeat(np.arange(10), 2)).all()
