import numpy as np


def l1_normalise_weights(w, spiked_neurons, limit, wmin, wmax):
    """Homeostatic column-sum normalisation for neurons that fired."""
    w = w.copy()
    for nrn in spiked_neurons:
        wsum = w[:, nrn].sum()
        if wsum > limit > 0:
            w[:, nrn] *= limit / wsum
    return np.clip(w, wmin, wmax)


def l2_normalise_weights(w, spiked_neurons, limit, wmin, wmax):
    """Scale down each spiked hidden neuron's incoming weight vector if its L2 norm exceeds limit."""
    w = w.copy()
    for nrn in spiked_neurons:
        col  = w[:, nrn]
        norm = np.linalg.norm(col)
        if norm > limit > 0:
            w[:, nrn] = col * (limit / norm)
    return np.clip(w, wmin, wmax)
