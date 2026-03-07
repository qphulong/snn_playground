import numpy as np
from .audio_utils import auditory_frontend

def compute_spike_input_current(
    audio_path,
    sustained_per_band=5,
    onset_per_band=2,
    phase_per_band=2,
    scale=1,
    pct=95,
    sust_gain=1.3,
    onset_gain=2.0,
    phase_gain=1.0,
    sust_spread_min=0.6,
    sust_spread_max=1.4,
):
    """
    Convert an audio file into an input current matrix for a spiking neural network.

    This function takes auditory features produced by `spike_encoding()` and expands
    them into multiple neuron types per cochlear frequency band. Each neuron type
    represents different auditory response characteristics inspired by biological
    auditory nerve fibers.

    Pipeline
    --------
    1. Audio → auditory feature extraction via `spike_encoding()`
    2. Obtain three feature maps:
        - E   : sustained energy (IHC compressed output)
        - dE  : onset energy (positive temporal derivative)
        - phase : rectified gammatone signal
    3. Normalize each feature type using a high percentile (default 95th).
    4. Generate multiple neurons per frequency band:
        - sustained neurons (energy response)
        - onset neurons (transient response)
        - phase neurons (phase locking)
    5. Apply gain scaling and small Gaussian noise.
    6. Return a time-varying current matrix suitable for driving LIF neurons.

    Parameters
    ----------
    audio_path : str
        Path to the input audio (.wav) file.

    sustained_per_band : int, default=5
        Number of neurons per frequency band that encode sustained energy (E).

    onset_per_band : int, default=2
        Number of neurons per band that encode onset activity (dE).

    phase_per_band : int, default=2
        Number of neurons per band that encode phase-locking signals.

    scale : float, default=1
        Global gain multiplier applied to all input currents.

    pct : float, default=95
        Percentile used for robust normalization of each feature type.

    sust_gain : float, default=1.3
        Gain factor for sustained-energy neurons.

    onset_gain : float, default=2.0
        Gain factor for onset neurons.

    phase_gain : float, default=1.0
        Gain factor for phase-locking neurons.

    sust_spread_min : float, default=0.6
        Minimum multiplicative factor applied to sustained neurons within a band.

    sust_spread_max : float, default=1.4
        Maximum multiplicative factor applied to sustained neurons within a band.

    Returns
    -------
    I : np.ndarray, shape (N_in, T)
        Input current matrix for the spiking network.

    T : int
        Number of time steps in the signal.
    """

    feats = auditory_frontend(audio_path)

    E = feats["E"]
    dE = feats["dE"]
    phase = feats["phase"]

    n_channels, T = E.shape

    # percentile normalization
    E_95 = max(np.percentile(E, pct), 1e-6)
    dE_95 = max(np.percentile(dE, pct), 1e-6)
    phase_95 = max(np.percentile(phase, pct), 1e-6)

    g_sust = sust_gain / E_95
    g_onset = onset_gain / dE_95
    g_phase = phase_gain / phase_95

    neurons_per_band = sustained_per_band + onset_per_band + phase_per_band
    N_in = n_channels * neurons_per_band

    I = np.zeros((N_in, T), dtype=np.float32)

    idx = 0
    for ch in range(n_channels):

        spread = np.linspace(sust_spread_min, sust_spread_max, sustained_per_band)
        for mult in spread:
            I[idx] = g_sust * mult * scale * E[ch]
            idx += 1

        for _ in range(onset_per_band):
            I[idx] = g_onset * scale * dE[ch]
            idx += 1

        for _ in range(phase_per_band):
            I[idx] = g_phase * scale * phase[ch]
            idx += 1

    I += 0.01 * np.random.randn(*I.shape).astype(np.float32)

    return I, T