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
    alpha=1.0,
    norm_percentile=99.0,
    clip_val=1.0,
    per_channel=False,
    eps=1e-6,
):
    """
    Encode an audio waveform into auditory-inspired spike features.

    This function implements a biologically inspired auditory pipeline:
    waveform → gammatone filterbank → upstream percentile normalization →
    inner hair cell compression → onset detection → phase signal.

    Normalization is applied **once, upstream** to the (signed) filterbank output,
    before any nonlinearity. Because `log1p(alpha * x)` is not scale-invariant, the
    signal must be brought to a known scale *before* the log so compression is
    consistent across utterances. E, dE and phase are then all derived from the same
    normalized signal — their relative balance is therefore set only by the downstream
    gains, not by independent per-feature normalizations.

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
        E = log1p(alpha * |signal_norm|).

    norm_percentile : float, default=99.0
        Percentile of |filterbank output| used as the normalization scale. Robust to
        the loudest transients (top 1% at 99) compared to a plain max.

    clip_val : float, default=1.0
        After dividing by the percentile scale, the normalized signal is clipped to
        [-clip_val, clip_val]. This bounds the input to the log and saturates the
        loudest excursions.

    per_channel : bool, default=False
        If False (default), one global percentile scalar is computed over the whole
        (n_channels, T) magnitude array — this preserves cross-channel relative energy
        (formant/timbre structure useful for speaker discrimination). If True, the
        percentile is computed per channel, equalizing quiet and loud bands.

    eps : float, default=1e-6
        Small constant to avoid division by zero.

    Returns
    -------
    dict
        Dictionary containing encoded auditory representations:

        - "E" : np.ndarray (n_channels, T)
            Log-compressed cochlear energy (IHC output), full-wave rectified.

        - "dE" : np.ndarray (n_channels, T)
            Onset detection signal (half-wave rectified temporal derivative of E).

        - "phase" : np.ndarray (n_channels, T)
            Negative half-wave of the normalized filterbank output. Complementary in
            polarity to E's full-wave energy, so it is not redundant with E.

        - "cf" : np.ndarray (n_channels,)
            Center frequencies of filterbank channels (low → high)

        - "sr" : int
            Sampling rate of processed audio

    Notes
    -----
    Processing pipeline:

    1. Audio loading
    2. ERB-spaced gammatone filterbank
    3. Upstream percentile normalization + clip (on the signed signal)
    4. Inner hair cell log compression (full-wave): E = log1p(alpha * |sig_norm|)
    5. Onset detection via positive temporal derivative of E
    6. Phase signal: negative half-wave of sig_norm

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
    # 3. Upstream percentile normalization (on the signed signal, before any
    #    nonlinearity). One scale derived from |signals|, then clip. This keeps
    #    the log compression in a consistent regime across utterances and puts
    #    E / dE / phase on a single shared reference frame.
    # ==============================
    if per_channel:
        scale = np.percentile(np.abs(signals), norm_percentile, axis=1, keepdims=True)
    else:
        scale = np.percentile(np.abs(signals), norm_percentile)
    sig_n = np.clip(signals / (scale + eps), -clip_val, clip_val)

    # ==============================
    # 4. Inner Hair Cell Compression (full-wave)
    # ==============================
    E = np.log1p(alpha * np.abs(sig_n))

    # ==============================
    # 5. Onset detection (positive temporal derivative of E)
    # ==============================
    dE = np.diff(E, axis=1, prepend=E[:, :1])
    dE[dE < 0] = 0

    # ==============================
    # 6. Phase signal: negative half-wave of the normalized signal.
    #    Complementary in polarity to E's full-wave energy → not redundant with E.
    # ==============================
    phase_signal = np.maximum(-sig_n, 0)

    return {
        "E": E,
        "dE": dE,
        "phase": phase_signal,
        "cf": cf,
        "sr": sr,
    }

