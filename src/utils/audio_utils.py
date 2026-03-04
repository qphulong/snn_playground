import numpy as np
import librosa

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

