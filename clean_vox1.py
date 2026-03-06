"""
clean_vox1_batch.py

Batch processor for the VoxCeleb1 dataset.

Processing the entire VoxCeleb1 dataset in a single run can often lead to
crashes or interrupted jobs due to the dataset's size and the long runtime
required. To make the cleaning process more reliable, this script processes
the dataset in small batches of speakers and can resume from where the
previous run stopped.

Instead of processing all speakers at once, the script:
  1. Checks which speakers already exist in the destination directory
     (datasets/vox1_cleaned).
  2. Selects the next N unprocessed speakers.
  3. Processes only that batch.

The output is written to a temporary directory so the results can be inspected
before committing them to the final dataset location.

Usage:
    python3 clean_vox1_batch.py <batch_size>

Example:
    python3 clean_vox1_batch.py 100

Processing order:
    1. All speakers from wav_dev (sorted)
    2. Then all speakers from wav_test (sorted)

Resume behaviour:
    The script determines completed speakers by checking:
        datasets/vox1_cleaned/<split>/<speaker_id>

    Any speaker already present there will be skipped automatically.

Output location:
    temp_process/

Example output structure:

    temp_process/
        wav_dev/
            id10001/
                id10001_00001/   -> 5-second .wav chunks
                id10001_00002/
                ...
        wav_test/
            ...

IMPORTANT:
    The script does NOT automatically move results to the final dataset.

    After verifying the processed audio, you must manually move each
    processed speaker directory from:

        temp_process/<split>/<speaker_id>

    into:

        datasets/vox1_cleaned/<split>/<speaker_id>

    This manual step prevents accidental corruption of the cleaned dataset
    if a batch fails midway.
"""

import sys
import wave
from pathlib import Path
from tqdm import tqdm


VOX1_ROOT    = Path("datasets/vox1")
DEST_ROOT    = Path("datasets/vox1_cleaned")   # checked for already-done speakers
TEMP_ROOT    = Path("temp_process")            # output goes here
SPLITS       = ["wav_dev", "wav_test"]
CORRUPT_LOG  = Path("corrupted_files.log")

CHUNK_DURATION = 5.0   # seconds
MIN_LAST_CHUNK = 2.5   # seconds

_corrupt_log_handle = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log_corrupted(path: Path, reason: str) -> None:
    msg = f"{path}\t{reason}"
    tqdm.write(f"[CORRUPT] {msg}")
    if _corrupt_log_handle is not None:
        _corrupt_log_handle.write(msg + "\n")
        _corrupt_log_handle.flush()


def read_wav(path: Path) -> tuple[bytes, wave._wave_params]:
    with wave.open(str(path), "rb") as f:
        params = f.getparams()
        frames = f.readframes(f.getnframes())
    return frames, params


def write_wav(path: Path, frames: bytes, params) -> None:
    with wave.open(str(path), "wb") as f:
        f.setparams(params)
        f.writeframes(frames)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_session(src_session: Path, dst_session: Path) -> None:
    wav_files = sorted(src_session.glob("*.wav"))
    if not wav_files:
        return

    buffer = bytearray()
    params_ref = None
    chunk_idx = 1

    for wav_path in wav_files:
        try:
            frames, params = read_wav(wav_path)
        except Exception as e:
            log_corrupted(wav_path, str(e))
            continue

        if params_ref is None:
            params_ref = params
            bpf            = params.nchannels * params.sampwidth
            chunk_bytes    = int(params.framerate * CHUNK_DURATION) * bpf
            min_last_bytes = int(params.framerate * MIN_LAST_CHUNK) * bpf
            dst_session.mkdir(parents=True, exist_ok=True)
        else:
            if (params.nchannels != params_ref.nchannels or
                    params.framerate != params_ref.framerate or
                    params.sampwidth != params_ref.sampwidth):
                log_corrupted(
                    wav_path,
                    f"WAV param mismatch (expected "
                    f"{params_ref.nchannels}ch/"
                    f"{params_ref.framerate}Hz/"
                    f"{params_ref.sampwidth * 8}bit)"
                )
                continue

        buffer.extend(frames)

        while len(buffer) >= chunk_bytes:
            write_wav(
                dst_session / f"{chunk_idx:05d}.wav",
                bytes(buffer[:chunk_bytes]),
                params_ref,
            )
            buffer = buffer[chunk_bytes:]
            chunk_idx += 1

    # write final chunk if long enough
    if params_ref is not None and len(buffer) > min_last_bytes:
        write_wav(
            dst_session / f"{chunk_idx:05d}.wav",
            bytes(buffer),
            params_ref,
        )


def process_speaker(speaker_dir: Path, dst_speaker: Path) -> None:
    session_dirs = sorted(d for d in speaker_dir.iterdir() if d.is_dir())
    speaker_id   = speaker_dir.name

    for idx, session_dir in enumerate(session_dirs, start=1):
        dst_session_name = f"{speaker_id}_{idx:05d}"
        dst_session      = dst_speaker / dst_session_name
        try:
            process_session(session_dir, dst_session)
        except Exception as e:
            tqdm.write(f"[ERROR] {session_dir}: {e}")


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------

def already_done_speakers(split: str) -> set[str]:
    """Return speaker IDs already present in DEST_ROOT for this split."""
    dest_split = DEST_ROOT / split
    if not dest_split.exists():
        return set()
    return {d.name for d in dest_split.iterdir() if d.is_dir()}


def build_global_todo(batch_size: int) -> list[tuple[str, Path]]:
    """
    Build an ordered list of (split, speaker_dir) pairs that still need
    processing, stopping after batch_size entries.

    Order: wav_dev speakers (sorted) first, then wav_test speakers (sorted).
    """
    todo: list[tuple[str, Path]] = []

    for split in SPLITS:
        src_split = VOX1_ROOT / split
        if not src_split.exists():
            print(f"[SKIP] {src_split} does not exist")
            continue

        done = already_done_speakers(split)
        pending = sorted(
            d for d in src_split.iterdir()
            if d.is_dir() and d.name not in done
        )

        for spk_dir in pending:
            todo.append((split, spk_dir))
            if len(todo) == batch_size:
                return todo

    return todo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _corrupt_log_handle

    if len(sys.argv) != 2 or not sys.argv[1].isdigit():
        print("Usage: python3 clean_vox1_batch.py <batch_size>")
        print("  e.g. python3 clean_vox1_batch.py 100")
        sys.exit(1)

    batch_size = int(sys.argv[1])
    if batch_size <= 0:
        print("batch_size must be a positive integer")
        sys.exit(1)

    todo = build_global_todo(batch_size)

    if not todo:
        print("Nothing to process — all speakers are already in the destination directory.")
        sys.exit(0)

    # Summarise what we're about to do
    from collections import Counter
    split_counts = Counter(split for split, _ in todo)
    print(f"\nBatch size : {batch_size}")
    print(f"To process : {len(todo)} speakers")
    for split in SPLITS:
        if split_counts[split]:
            print(f"  {split}: {split_counts[split]} speakers")
    print(f"Output dir : {TEMP_ROOT}/")
    print(f"(Move to {DEST_ROOT}/ manually after checking)\n")

    # Open corrupt log in append mode so successive batches accumulate
    _corrupt_log_handle = CORRUPT_LOG.open("a", encoding="utf-8")
    print(f"Appending corrupted-file log to: {CORRUPT_LOG}")

    with tqdm(total=len(todo), unit="speaker") as pbar:
        for split, speaker_dir in todo:
            speaker_id  = speaker_dir.name
            dst_speaker = TEMP_ROOT / split / speaker_id

            pbar.set_description(f"{split}/{speaker_id}")
            process_speaker(speaker_dir, dst_speaker)
            pbar.update(1)

    _corrupt_log_handle.close()
    _corrupt_log_handle = None

    print(f"\nDone. Batch written to: {TEMP_ROOT}/")
    print(f"Inspect, then move each speaker directory to {DEST_ROOT}/<split>/")
    print(f"Corrupted files logged to: {CORRUPT_LOG}")


if __name__ == "__main__":
    main()