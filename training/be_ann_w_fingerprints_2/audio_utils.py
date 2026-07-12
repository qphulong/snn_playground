import shutil
import subprocess

import numpy as np
import librosa
import soundfile as sf
from gammatone.filters import centre_freqs, make_erb_filters, erb_filterbank


def _ffprobe_sample_rate(path):
    """Native sample rate of `path` via ffprobe, or None if it can't be determined."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            check=True, capture_output=True, text=True,
        ).stdout.strip().splitlines()
        return int(out[0]) if out else None
    except (subprocess.CalledProcessError, ValueError):
        return None


def _ffmpeg_decode(path, sr):
    """Decode any ffmpeg-readable container (m4a/aac/mp3/...) to a mono float32
    waveform. If `sr` is None the native rate is probed and preserved; otherwise
    ffmpeg's resampler outputs directly at `sr`."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            f"Cannot decode {path!r}: libsndfile failed and ffmpeg is not on PATH. "
            "Install ffmpeg to read m4a/aac audio."
        )
    target_sr = sr if sr is not None else (_ffprobe_sample_rate(path) or 16000)
    proc = subprocess.run(
        [ffmpeg, "-nostdin", "-loglevel", "error", "-i", path,
         "-ac", "1", "-ar", str(target_sr),
         "-f", "f32le", "-acodec", "pcm_f32le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to decode {path!r}: "
            f"{proc.stderr.decode('utf-8', 'ignore').strip()}"
        )
    y = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
    return y, target_sr


def load_audio(path, sr=16000):
    """Load `path` to a mono float32 waveform at `sr` Hz (native rate if `sr` is None).

    Format-robust replacement for ``librosa.load``: wav/flac/ogg are decoded by
    libsndfile (soundfile); m4a/aac and any container libsndfile can't open are
    decoded via ffmpeg. This avoids librosa's audioread m4a fallback, which is
    deprecated and slated for removal in librosa 1.0 (and emits a warning per file).
    """
    try:
        y, sr_native = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        return _ffmpeg_decode(path, sr)
    if y.ndim > 1:                       # multi-channel -> mono (match librosa default)
        y = y.mean(axis=1)
    if sr is not None and sr_native != sr:
        y = librosa.resample(y, orig_sr=sr_native, target_sr=sr)
    return np.ascontiguousarray(y, dtype=np.float32), (sr if sr is not None else sr_native)


def load_mel_spectrogram(
    wav_path: str,
    n_mels: int = 256,
    fmax: int | None = 8000,
    target_frames_per_second: int = 1000,
    normalize: bool = True,
):
    audio, sr = load_audio(wav_path, sr=None)

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
    Encode an audio waveform into auditory-inspired spike features

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
    signal, sr = load_audio(audio_path, sr=sr)

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
