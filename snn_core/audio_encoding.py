import numpy as np
import librosa
from gammatone.filters import make_erb_filters, erb_filterbank, centre_freqs
from dataclasses import dataclass
from typing import Tuple


@dataclass
class AudioConfig:
    sr: int = 16000
    num_filters: int = 100
    f_min: float = 50.0
    sustained_per_band: int = 5
    onset_per_band: int = 2
    phase_per_band: int = 2
    SCALE: float = 1.2
    percentile: int = 95

    @property
    def neurons_per_band(self) -> int:
        return self.sustained_per_band + self.onset_per_band + self.phase_per_band

    @property
    def N_in(self) -> int:
        return self.num_filters * self.neurons_per_band


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    signal, _ = librosa.load(path, sr=sr)
    return signal


def create_gammatone_filterbank(sr: int, num_filters: int, f_min: float):
    """
    Returns (center_freqs, filter_coeffs) in LOW→HIGH order.
    """
    cf = centre_freqs(sr, num_filters, f_min)
    erb_filters = make_erb_filters(sr, cf)
    # Reorder LOW→HIGH (gammatone lib returns HIGH→LOW)
    cf = cf[::-1]
    return cf, erb_filters


def compute_input_current(audio_path: str, config: AudioConfig) -> Tuple[np.ndarray, int]:
    """
    Convert audio file to input current matrix ready for Brian2.

    Process:
        1. Load audio and apply gammatone filterbank
        2. IHC log compression (sustained neurons)
        3. Onset detection: half-wave rectified derivative
        4. Phase signal: half-wave rectification
        5. Global 95th-percentile normalization (preserves relative band energy)
        6. Distribute currents to neuron types with gain diversity for sustained
        7. Add small noise

    Returns:
        I : (N_in, T) float32 array of input currents
        T : number of time steps
    """
    signal = load_audio(audio_path, config.sr)

    cf = centre_freqs(config.sr, config.num_filters, config.f_min)
    erb_filters = make_erb_filters(config.sr, cf)
    filtered = erb_filterbank(signal, erb_filters)

    # Reorder LOW→HIGH
    filtered = filtered[::-1]
    n_channels, T = filtered.shape

    # IHC log compression
    E = np.log1p(4.0 * np.maximum(filtered, 0))

    # Onset detection (half-wave rectified temporal derivative)
    dE = np.diff(E, axis=1, prepend=E[:, :1])
    dE[dE < 0] = 0

    # Phase signal (half-wave rectification of raw filtered signal)
    phase = np.maximum(filtered, 0)

    # Global percentile normalization (preserves relative band energies)
    E_pct     = max(np.percentile(E,     config.percentile), 1e-6)
    dE_pct    = max(np.percentile(dE,    config.percentile), 1e-6)
    phase_pct = max(np.percentile(phase, config.percentile), 1e-6)

    g_sust  = 1.3 / E_pct
    g_onset = 2.0 / dE_pct
    g_phase = 1.0 / phase_pct

    N_in = n_channels * config.neurons_per_band
    I = np.zeros((N_in, T), dtype=np.float32)
    idx = 0

    for ch in range(n_channels):
        # Sustained: spread gains across [0.6, 1.4] to give diversity within band
        spread = np.linspace(0.6, 1.4, config.sustained_per_band)
        for mult in spread:
            I[idx] = g_sust * mult * config.SCALE * E[ch]
            idx += 1
        for _ in range(config.onset_per_band):
            I[idx] = g_onset * config.SCALE * dE[ch]
            idx += 1
        for _ in range(config.phase_per_band):
            I[idx] = g_phase * config.SCALE * phase[ch]
            idx += 1

    # Small noise to break ties
    rng = np.random.default_rng()
    I += (0.01 * rng.standard_normal(I.shape)).astype(np.float32)

    return I, T
