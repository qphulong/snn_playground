import numpy as np
import librosa
from gammatone.filters import centre_freqs, make_erb_filters, erb_filterbank

def load_mel_spectrogram(
    wav_path: str,
    n_mels: int = 256,
    fmax: int | None = 8000,
    target_frames_per_second: int = 1000,
    normalize: bool = True,
):
    audio, sr = librosa.load(wav_path, sr=None)

    hop_length = int(sr / target_frames_per_second)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        fmax=fmax,
        hop_length=hop_length
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    if normalize:
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    return mel_db, sr

def auditory_frontend(
    audio_path,
    sr=16000,
    num_filters=100,
    f_min=50,
    alpha=4.0,
    normalization="global",
    eps=1e-6,
):
    """
    Encode an audio waveform into auditory-inspired spike features.

    This function implements a biologically inspired auditory pipeline:
    waveform → gammatone filterbank → inner hair cell compression →
    onset detection → phase signal → normalization.

    Parameters
    ----------
    audio_path : str
        Path to the input audio file.

    sr : int, default=16000
        Target sampling rate for loading audio.

    num_filters : int, default=100
        Number of ERB-spaced gammatone filters (frequency channels).

    f_min : float, default=50
        Minimum center frequency (Hz) of the filterbank.

    alpha : float, default=4.0
        Compression strength for inner hair cell log compression:
        E = log(1 + alpha * max(signal, 0)).

    normalization : str, default="global"
        Normalization strategy for compressed energy.
        Options:
        - "global": divide by global max amplitude
        - "rms": per-channel RMS normalization
        - None: no normalization

    eps : float, default=1e-6
        Small constant to avoid division by zero.

    Returns
    -------
    dict
        Dictionary containing encoded auditory representations:

        - "E" : np.ndarray (n_channels, T)
            Log-compressed cochlear energy (IHC output)

        - "dE" : np.ndarray (n_channels, T)
            Onset detection signal (half-wave rectified temporal derivative)

        - "phase" : np.ndarray (n_channels, T)
            Half-wave rectified filterbank output representing phase locking

        - "cf" : np.ndarray (n_channels,)
            Center frequencies of filterbank channels (low → high)

        - "sr" : int
            Sampling rate of processed audio

    Notes
    -----
    Processing pipeline:

    1. Audio loading
    2. ERB-spaced gammatone filterbank
    3. Half-wave rectification
    4. Inner hair cell log compression
    5. Onset detection via positive temporal derivative
    6. Phase signal extraction
    7. Optional normalization

    All channel outputs are ordered from **low → high frequency**.
    """

    # ==============================
    # 1. Load audio
    # ==============================
    signal, sr = librosa.load(audio_path, sr=sr)

    # ==============================
    # 2. Gammatone filterbank
    # ==============================
    cf = centre_freqs(sr, num_filters, f_min)
    erb_filters = make_erb_filters(sr, cf)

    filtered_signals = erb_filterbank(signal, erb_filters)

    # reorder HIGH→LOW → LOW→HIGH
    cf = cf[::-1]
    filtered_signals = filtered_signals[::-1]

    signals = filtered_signals
    n_channels, T = signals.shape

    # ==============================
    # 3. Inner Hair Cell Compression
    # ==============================
    E = np.log1p(alpha * np.maximum(signals, 0))

    # ==============================
    # 4. Onset detection
    # ==============================
    dE = np.diff(E, axis=1, prepend=E[:, :1])
    dE[dE < 0] = 0

    # ==============================
    # 5. Phase signal
    # ==============================
    phase_signal = np.maximum(signals, 0)

    # ==============================
    # 6. Normalization
    # ==============================
    if normalization == "global":
        max_val = np.max(np.abs(E)) + eps
        E = E / max_val

    elif normalization == "rms":
        rms = np.sqrt(np.mean(E**2, axis=1, keepdims=True)) + eps
        E = E / rms

    elif normalization is None:
        pass

    else:
        raise ValueError("normalization must be {'global', 'rms', None}")

    return {
        "E": E,
        "dE": dE,
        "phase": phase_signal,
        "cf": cf,
        "sr": sr,
    }

