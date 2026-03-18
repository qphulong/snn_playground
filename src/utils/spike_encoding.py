import numpy as np
from .audio_utils import auditory_frontend

def compute_spike_input_current(
    audio_path,
    sustained_per_band=5,
    onset_per_band=2,
    phase_per_band=2,
    scale=1,
    sust_gain=1.0,
    onset_gain=2.0,
    phase_gain=1.0,
    sust_spread_min=0.6,
    sust_spread_max=1.4,
    audio_sample_rate=16000,
    simulation_sample_rate=1000,
):
    """
    Convert an audio file into a downsampled input current matrix for a spiking neural network.

    This function takes auditory features produced by `auditory_frontend()` and expands
    them into multiple neuron types per cochlear frequency band. Each neuron type
    represents different auditory response characteristics inspired by biological
    auditory nerve fibers.

    Pipeline
    --------
    1. Audio → auditory feature extraction via `auditory_frontend()`
    2. Obtain three feature maps:
        - E     : sustained energy (IHC compressed output)
        - dE    : onset energy (positive temporal derivative)
        - phase : rectified gammatone signal
    3. Generate multiple neurons per frequency band:
        - sustained neurons (energy response, spread across gain multipliers)
        - onset neurons (transient response)
        - phase neurons (phase locking)
    4. Apply gain scaling and small Gaussian noise.
    5. Downsample from `audio_sample_rate` to `simulation_sample_rate` by
       block-averaging across the decimation factor.
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

    sust_gain : float, default=1.3
        Gain factor for sustained-energy neurons.

    onset_gain : float, default=2.0
        Gain factor for onset neurons.

    phase_gain : float, default=1.0
        Gain factor for phase-locking neurons.

    sust_spread_min : float, default=0.6
        Minimum multiplicative factor applied across sustained neurons within a band.

    sust_spread_max : float, default=1.4
        Maximum multiplicative factor applied across sustained neurons within a band.

    audio_sample_rate : int, default=16000
        Sampling rate (Hz) of the raw audio and the auditory feature maps produced
        by `auditory_frontend()`.

    simulation_sample_rate : int, default=1000
        Target sampling rate (Hz) for the Brian2 simulation (i.e. 1 / defaultclock.dt).
        The current matrix is downsampled from `audio_sample_rate` to this rate by
        block-averaging. Must evenly divide `audio_sample_rate`.

    Returns
    -------
    I_sim : np.ndarray, shape (N_in, T_sim)
        Simulation-ready input current matrix, where
        T_sim = T // decimation_factor.

    T_sim : int
        Number of time steps after downsampling, corresponding to the
        total simulation duration in Brian2 timesteps.
    """

    feats = auditory_frontend(audio_path)

    E = feats["E"]
    dE = feats["dE"]
    phase = feats["phase"]

    n_channels, T = E.shape

    g_sust = sust_gain
    g_onset = onset_gain
    g_phase = phase_gain

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

    decimation_factor = audio_sample_rate // simulation_sample_rate
    I_sim = I.reshape(N_in, -1, decimation_factor).mean(axis=2)
    T_sim = I_sim.shape[1]

    return I_sim, T_sim