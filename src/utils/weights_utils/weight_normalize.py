import numpy as np


def l1_normalise_weights(w, spiked_neurons, limit, wmin, wmax):
    """Homeostatic column-sum normalisation for neurons that fired."""
    w = w.copy()
    for nrn in spiked_neurons:
        wsum = w[:, nrn].sum()
        if wsum > limit > 0:
            w[:, nrn] *= limit / wsum
    return np.clip(w, wmin, wmax)
