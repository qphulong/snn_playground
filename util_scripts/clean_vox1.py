"""
clean_vox1.py

Shard and chunk the VoxCeleb1 `dev` pool for upload to Kaggle.

`datasets/vox1/dev` holds the full VoxCeleb1 pool — 1,251 speakers in
`speaker_id/session_id/utterance.wav` layout (PCM s16le, 16 kHz, mono). This
script reorganises it into speaker shards of 200 and consolidates the many short
utterances per session into uniform ~5-second clips, re-encoded to AAC `.m4a`,
so the dataset is small enough (per shard) and tidy enough to upload.

What it does
------------
  * Groups speakers (sorted) into shards of 200, named dev_01, dev_02, ...
  * For each session, concatenates its utterances in order, then cuts the
    stream into ~5-second clips renumbered 00001.m4a, 00002.m4a, ...
  * Unlike the VoxCeleb2 pipeline, the VoxCeleb1 source is PCM `.wav`, so the
    cut RE-ENCODES to AAC (-c:a aac -b:a 64k). Stream-copy is not possible from
    PCM into an m4a container. Segments land on AAC packet boundaries, so clips
    are ~5 s (a hair over) rather than exactly 5 s.
  * A trailing clip shorter than 2.5 s is dropped.

Output layout (inside datasets/vox1/):

    datasets/vox1/
        dev_01/
            id10001/
                1zcIwhmdeo4/
                    00001.m4a
                    00002.m4a
                ...
        dev_02/
            ...

Disk / deletion
---------------
Only one copy of the data fits on disk, so in real mode each speaker is
processed atomically: all of its clips are written into the target shard, then
its ORIGINAL directory under datasets/vox1/dev is DELETED to free space. The
job is resumable — interrupt it any time and re-run; state is re-derived purely
from what is on disk.

Dummy mode
----------
    python util_scripts/clean_vox1.py --dummy

Processes only the first 4 speakers into a shard named `dev_dummy` and DELETES
NOTHING. Use this to eyeball the output before committing to the destructive
real run.

Real run
--------
    python util_scripts/clean_vox1.py

Processes all remaining speakers, deleting originals as it goes.

Corrupted files are skipped and logged to corrupted_files_vox1.log.
"""

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SRC_DEV        = Path("datasets/vox1/dev")     # source speaker pool
SHARD_ROOT     = Path("datasets/vox1")         # shards written here: dev_01, ...
SHARD_SIZE     = 200                           # speakers per shard
DUMMY_SIZE     = 4                             # speakers in the dummy shard
DUMMY_SHARD    = "dev_dummy"
CHUNK_DURATION = 5.0                           # seconds (target, not strict)
MIN_LAST_CHUNK = 2.5                           # drop a trailing clip shorter than this
AAC_BITRATE    = "64k"                         # wav -> m4a re-encode bitrate
CORRUPT_LOG    = Path("corrupted_files_vox1.log")

SHARD_RE = re.compile(r"^dev_(\d{2})$")        # matches dev_01, dev_02, ...

_corrupt_log_handle = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log_corrupted(path: Path, reason: str) -> None:
    msg = f"{path}\t{reason.strip()}"
    tqdm.write(f"[CORRUPT] {msg}")
    if _corrupt_log_handle is not None:
        _corrupt_log_handle.write(msg + "\n")
        _corrupt_log_handle.flush()


def probe_duration(path: Path) -> float | None:
    """Return the duration of an audio file in seconds, or None on failure."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(path)],
            capture_output=True, text=True, check=True,
        )
        return float(out.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_session(src_session: Path, dst_session: Path) -> int:
    """
    Concatenate every utterance in `src_session` and cut the result into
    ~5-second AAC clips written to `dst_session`. Returns the number of clips kept.
    """
    utterances = sorted(src_session.glob("*.wav"))
    if not utterances:
        return 0

    dst_session.mkdir(parents=True, exist_ok=True)

    # ffmpeg concat demuxer needs a list file of `file '<path>'` lines.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as lf:
        for utt in utterances:
            # single-quote escaping per the concat demuxer's rules
            safe = str(utt.resolve()).replace("'", r"'\''")
            lf.write(f"file '{safe}'\n")
        list_path = Path(lf.name)

    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "concat", "-safe", "0", "-i", str(list_path),
             "-map", "0:a", "-c:a", "aac", "-b:a", AAC_BITRATE,
             "-fflags", "+genpts", "-avoid_negative_ts", "make_zero",
             "-f", "segment", "-segment_time", str(CHUNK_DURATION),
             "-segment_start_number", "1", "-reset_timestamps", "1",
             str(dst_session / "%05d.m4a")],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            log_corrupted(src_session, result.stderr or "ffmpeg failed")
    finally:
        list_path.unlink(missing_ok=True)

    clips = sorted(dst_session.glob("*.m4a"))

    # Drop a too-short trailing clip.
    if clips:
        dur = probe_duration(clips[-1])
        if dur is not None and dur < MIN_LAST_CHUNK:
            clips[-1].unlink()
            clips = clips[:-1]

    # Remove an empty session dir (e.g. total audio < MIN_LAST_CHUNK).
    if not clips:
        try:
            dst_session.rmdir()
        except OSError:
            pass

    return len(clips)


def process_speaker(speaker_dir: Path, dst_speaker: Path, delete_original: bool) -> int:
    """Process all sessions of a speaker. Returns total clips kept."""
    sessions = sorted(d for d in speaker_dir.iterdir() if d.is_dir())
    total = 0
    for session in sessions:
        try:
            total += process_session(session, dst_speaker / session.name)
        except Exception as e:  # noqa: BLE001 — never let one session abort the run
            log_corrupted(session, str(e))

    if delete_original:
        shutil.rmtree(speaker_dir)

    return total


# ---------------------------------------------------------------------------
# Shard bookkeeping (real mode)
# ---------------------------------------------------------------------------

def existing_shards() -> list[Path]:
    """Return existing dev_NN shard dirs, sorted by index."""
    shards = [d for d in SHARD_ROOT.iterdir()
              if d.is_dir() and SHARD_RE.match(d.name)]
    return sorted(shards, key=lambda d: int(SHARD_RE.match(d.name).group(1)))


def shard_speaker_ids() -> set[str]:
    """Speaker IDs already present in any dev_NN shard."""
    ids: set[str] = set()
    for shard in existing_shards():
        ids.update(d.name for d in shard.iterdir() if d.is_dir())
    return ids


def current_fill_target() -> tuple[Path, int]:
    """
    Return (shard_dir, count_so_far) for the shard currently being filled.

    Uses the highest existing dev_NN; if it is already full, starts the next.
    """
    shards = existing_shards()
    if shards:
        last = shards[-1]
        count = sum(1 for d in last.iterdir() if d.is_dir())
        if count < SHARD_SIZE:
            return last, count
        next_idx = int(SHARD_RE.match(last.name).group(1)) + 1
    else:
        next_idx = 1
    return SHARD_ROOT / f"dev_{next_idx:02d}", 0


def next_shard(shard_dir: Path) -> Path:
    idx = int(SHARD_RE.match(shard_dir.name).group(1)) + 1
    return SHARD_ROOT / f"dev_{idx:02d}"


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def run_dummy() -> None:
    speakers = sorted(d for d in SRC_DEV.iterdir() if d.is_dir())[:DUMMY_SIZE]
    if not speakers:
        print(f"No speakers found under {SRC_DEV}")
        return

    dst_shard = SHARD_ROOT / DUMMY_SHARD
    print(f"\nMode       : DUMMY (no originals deleted)")
    print(f"Speakers   : {len(speakers)}")
    print(f"Output dir : {dst_shard}/\n")

    with tqdm(total=len(speakers), unit="speaker") as pbar:
        for spk in speakers:
            dst_speaker = dst_shard / spk.name
            pbar.set_description(spk.name)
            if dst_speaker.exists():
                pbar.update(1)
                continue  # idempotent re-run
            process_speaker(spk, dst_speaker, delete_original=False)
            pbar.update(1)

    print(f"\nDone. Inspect {dst_shard}/ — source data under {SRC_DEV} is untouched.")


def run_real() -> None:
    done = shard_speaker_ids()
    remaining = [d for d in sorted(SRC_DEV.iterdir())
                 if d.is_dir() and d.name not in done]

    if not remaining:
        print(f"Nothing to process — no unsharded speakers left under {SRC_DEV}.")
        return

    shard_dir, fill = current_fill_target()

    print(f"\nMode       : REAL (originals deleted as we go)")
    print(f"Remaining  : {len(remaining)} speakers")
    print(f"Resuming at: {shard_dir.name} ({fill}/{SHARD_SIZE} filled)")
    print(f"Output dir : {SHARD_ROOT}/dev_NN/\n")

    with tqdm(total=len(remaining), unit="speaker") as pbar:
        for spk in remaining:
            if fill >= SHARD_SIZE:
                shard_dir = next_shard(shard_dir)
                fill = 0

            dst_speaker = shard_dir / spk.name
            pbar.set_description(f"{shard_dir.name}/{spk.name}")

            if dst_speaker.exists():
                # Crash recovery: clips already written, just drop the original.
                shutil.rmtree(spk)
            else:
                process_speaker(spk, dst_speaker, delete_original=True)

            fill += 1
            pbar.update(1)

    print(f"\nDone. Shards written under {SHARD_ROOT}/; processed originals removed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _corrupt_log_handle

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dummy", action="store_true",
        help=f"Process only the first {DUMMY_SIZE} speakers into "
             f"{DUMMY_SHARD}/ and delete nothing.",
    )
    args = parser.parse_args()

    if not SRC_DEV.exists():
        parser.error(f"source pool not found: {SRC_DEV}")

    _corrupt_log_handle = CORRUPT_LOG.open("a", encoding="utf-8")
    try:
        if args.dummy:
            run_dummy()
        else:
            run_real()
    finally:
        _corrupt_log_handle.close()
        _corrupt_log_handle = None


if __name__ == "__main__":
    main()
