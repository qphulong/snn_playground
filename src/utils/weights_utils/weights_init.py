import numpy as np


def gaussian_weight_matrix(n_pre, n_post, sigma, noise_std, w_sum, wmin, wmax):
    """Toroidal Gaussian initialisation."""
    i = np.arange(n_pre).reshape(-1, 1)
    j = np.arange(n_post).reshape(1, -1)
    dist = np.abs(i - j)
    dist = np.minimum(dist, n_pre - dist)
    w = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    w += np.random.normal(0, noise_std, size=w.shape)
    w = np.clip(w, 0, None)
    w = w / w.sum(axis=0, keepdims=True) * w_sum
    return np.clip(w, wmin, wmax)
