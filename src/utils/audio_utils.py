import numpy as np
import librosa
from gammatone.filters import centre_freqs, make_erb_filters, erb_filterbank


def _channel_contrast(X, channel_floor=0.8, contrast_curve=1.0,
                      noise_floor_frac=0.1, eps=1e-6):
    """Compress the across-channel activity distribution while preserving rank.

    Each channel's long-run activity (its time-mean) is remapped onto the range
    ``[channel_floor, 1]`` and the whole map is then normalised to unit mean.

    Parameters
    ----------
    X : np.ndarray (n_channels, T)
        A non-negative feature map (E, dE or phase).
    channel_floor : float in [0, 1]
        The weakest channel's target mean relative to the strongest one — the
        "activity distance" knob. ``1.0`` makes every channel equally active
        (no dominance); smaller values let loud channels dominate more, while
        order is always preserved. e.g. ``0.8`` -> weakest/strongest ≈ 0.8.
    contrast_curve : float
        Optional power-law shaping of the rank before the floor map. ``1.0`` is
        linear; ``<1`` lifts mid channels toward the top, ``>1`` pushes them down.
    noise_floor_frac : float
        Near-silent channels are lifted to the floor, but their amplification is
        capped at ``1 / noise_floor_frac`` to avoid blowing up noise.
    """
    m = X.mean(axis=1)                                 # (n_channels,)
    m_max = float(m.max())
    if m_max <= eps:
        return X
    r = m / (m_max + eps)                              # rank in [0, 1]
    if contrast_curve != 1.0:
        r = np.power(r, contrast_curve)
    g = channel_floor + (1.0 - channel_floor) * r     # target per-channel mean
    denom = np.maximum(m, m_max * noise_floor_frac)    # cap gain at 1/noise_floor_frac
    X = X * (g / (denom + eps))[:, None]
    X = X / (X.mean() + eps)
    return X


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
    alpha=1.0,
    normalization="global",
    gamma=0.5,
    eps=1e-6,
    onset_target_mean=0.2,
    phase_target_mean=0.2,
    channel_floor=0.8,
    contrast_curve=1.0,
    noise_floor_frac=0.1,
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

    alpha : float, default=1.0
        Compression strength for inner hair cell log compression:
        E = log(1 + alpha * max(signal, 0)).

    normalization : str, default="global"
        Normalization strategy for compressed energy.
        Options:
        - "global": divide by global max amplitude
        - "rms": per-channel RMS normalization
        - "perchannel": two-stage per-channel mean equalization + per-file unit mean
        - "contrast": remap each channel's mean onto [channel_floor, 1] (rank
          preserved) for E, dE and phase, each normalised to unit mean. Loud
          channels stay dominant by a bounded, tunable amount. In this mode the
          legacy onset/phase equalization and the ``gamma`` power-law are skipped
          (use ``channel_floor`` / ``contrast_curve`` instead).
        - None: no normalization

    channel_floor : float, default=0.8
        Only used when ``normalization="contrast"``. Weakest channel's target
        mean relative to the strongest — the across-channel "activity distance"
        knob. See ``_channel_contrast``.

    contrast_curve : float, default=1.0
        Only used when ``normalization="contrast"``. Power-law shaping of the
        channel rank before the floor map.

    noise_floor_frac : float, default=0.1
        Only used when ``normalization="contrast"``. Caps the amplification of
        near-silent channels at ``1 / noise_floor_frac``.

    gamma : float, default=0.3
        Power law exponent applied to E after normalization.
        Compresses dynamic range across channels while preserving relative dominance.
        Values < 1 boost weak channels without flattening strong ones (e.g. 0.3 turns
        a 100x ratio into ~4x). Set to 1.0 to disable.

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
    # 4. Onset detection (raw)
    # ==============================
    dE = np.diff(E, axis=1, prepend=E[:, :1])
    dE[dE < 0] = 0

    # ==============================
    # 5. Phase signal (raw)
    # ==============================
    phase_signal = np.maximum(signals, 0)

    # ==============================
    # 6. Normalization
    # ==============================
    if normalization == "contrast":
        # Remap each channel's mean onto [channel_floor, 1] (rank preserved) for
        # all three maps, each then normalised to unit mean. channel_floor is the
        # across-channel "activity distance" knob. The legacy onset/phase
        # equalization and the gamma power-law are intentionally skipped here —
        # channel_floor / contrast_curve play that role instead.
        E            = _channel_contrast(E,            channel_floor, contrast_curve, noise_floor_frac, eps)
        dE           = _channel_contrast(dE,           channel_floor, contrast_curve, noise_floor_frac, eps)
        phase_signal = _channel_contrast(phase_signal, channel_floor, contrast_curve, noise_floor_frac, eps)
        return {
            "E": E,
            "dE": dE,
            "phase": phase_signal,
            "cf": cf,
            "sr": sr,
        }

    # ── Legacy modes: per-channel equalization of onset / phase ──────────────
    # Per-channel mean equalization (mirrors E's perchannel normalization):
    # bring every channel's mean to the same level so no channel has a
    # systematic rate advantage, then normalize file-level mean to 1.
    dE_ch_mean = dE.mean(axis=1, keepdims=True)
    global_dE_mean = float(dE_ch_mean.mean())
    noise_floor_dE = global_dE_mean * 0.1
    dE = dE * (global_dE_mean / np.maximum(dE_ch_mean, noise_floor_dE))
    dE = dE / (dE.mean() + eps) * onset_target_mean

    # Per-channel mean equalization: same rationale as dE above.
    phase_ch_mean = phase_signal.mean(axis=1, keepdims=True)
    global_phase_mean = float(phase_ch_mean.mean())
    noise_floor_phase = global_phase_mean * 0.1
    phase_signal = phase_signal * (global_phase_mean / np.maximum(phase_ch_mean, noise_floor_phase))
    phase_signal = phase_signal / (phase_signal.mean() + eps) * phase_target_mean

    if normalization == "global":
        max_val = np.max(np.abs(E)) + eps
        E = E / max_val

    elif normalization == "rms": #TODO: consider remove this, likely useless in the future
        rms = np.sqrt(np.mean(E**2, axis=1, keepdims=True)) + eps
        E = E / rms

    elif normalization == "perchannel":
        # Stage 1 — per-channel: bring every channel's mean to the same level so
        # no channel has a systematic long-run rate advantage within a file.
        # noise_floor caps the gain at 10× for near-silent channels.
        ch_mean     = E.mean(axis=1, keepdims=True)        # (n_channels, 1)
        global_mean = float(ch_mean.mean())
        noise_floor = global_mean * 0.1                    # cap gain at 10×
        E = E * (global_mean / np.maximum(ch_mean, noise_floor))
        # Stage 2 — per-file: normalise the whole matrix to E.mean() == 1 so
        # total activity (and therefore spike count) is consistent across files.
        E = E / (E.mean() + eps)

    elif normalization is None:
        pass

    else:
        raise ValueError(
            "normalization must be {'global', 'rms', 'perchannel', 'contrast', None}"
        )

    if gamma != 1.0:
        E = np.power(E, gamma)

    return {
        "E": E,
        "dE": dE,
        "phase": phase_signal,
        "cf": cf,
        "sr": sr,
    }
