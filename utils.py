import torch
import numpy as np
from collections.abc import Iterable
import librosa


def get_signal(freq=None, mag=1.0, dur=1.0, sr=16_000, bias=0.0):
    if not isinstance(freq, Iterable):
        freq = [freq]
    if not isinstance(mag, Iterable):
        mag = [mag] * len(freq)

    assert len(freq) == len(mag)

    y = 0.0
    for f, m in zip(freq, mag):
        t = np.linspace(0, dur, int(dur * sr), endpoint=False)
        y += m * np.sin(2 * np.pi * f * t)
    return torch.FloatTensor(y) + bias


def get_signal_from_file(freq=None, root="./audios", sample_rate=16000):
    assert len(freq) == 3
    y, _ = librosa.load(f"{root}/{int(freq[0])}_{int(freq[1])}_{int(freq[2])}.wav", sr=sample_rate)
    y /= y.max()
    return torch.FloatTensor(y)


def get_step_signal(base_freq, sig_freq, sig_dur_ratio=0.1):
    base_sig = get_signal(freq=base_freq)
    sig = get_signal(freq=sig_freq)
    assert len(sig) % 2 == 0

    half_sig_dur = int(len(sig) // 2 * sig_dur_ratio)
    mask = torch.concat((
        torch.zeros(len(sig) // 2 - half_sig_dur),
        torch.ones(half_sig_dur * 2),
        torch.zeros(len(sig) // 2 - half_sig_dur),
    )).bool()
    return (sig * mask) + (base_sig * (~mask)), mask.numpy()


def get_feature(model, signal):
    model.eval()
    feats = model.wav2vec2.feature_extractor(signal[None, :])
    feats = feats.transpose(1, 2)
    _, feats = model.wav2vec2.feature_projection(feats)
    # batch_size, t_dim, f_dim = feats.shape
    return feats[0].detach().numpy()


def get_squashed_mask(mask, window=400, stride=320):
    assert len(mask) >= window >= stride

    squashed_mask = []
    for i in range(0, len(mask), 320):
        if len(mask[i:i+window]) == window:
            squashed_mask.append(mask[i:i+window].sum() / window)
    return np.array(squashed_mask)


def cosine_sim(feats_l, feats_r):
    dot_prod = (feats_l @ feats_r.T)
    size_l = np.sqrt(np.square(feats_l).sum(1))
    size_r = np.sqrt(np.square(feats_r).sum(1))
    return dot_prod / size_l[:, None] / size_r[None, :]


def linear_cka_sim(feats_l, feats_r):
    # Based on the code:
    # https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment

    def centering(K):
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        H = I - unit / n
        return np.dot(np.dot(H, K), H)

    def linear_HSIC(X, Y):
        L_X = np.dot(X, X.T)
        L_Y = np.dot(Y, Y.T)
        return np.sum(centering(L_X) * centering(L_Y))

    def linear_CKA(X, Y):
        hsic = linear_HSIC(X, Y)
        var1 = np.sqrt(linear_HSIC(X, X))
        var2 = np.sqrt(linear_HSIC(Y, Y))
        return hsic / (var1 * var2)

    return linear_CKA(feats_l, feats_r)
