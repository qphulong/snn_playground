"""
Prepare datasets/vox1_10person_fingerprint from datasets/vox1_cleaned/wav_dev.

Selects N random speakers that each have at least --n-records utterance folders
with >= --n-samples wav files. From each qualifying record the first --n-samples
files are taken: indices 0..(n_samples-3) go to wav_dev, the last 2 go to wav_test.
"""

import argparse
import random
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="datasets/vox1_cleaned/wav_dev",
                   help="Source speaker directory")
    p.add_argument("--dst", default="datasets/vox1_10person_fingerprint",
                   help="Output dataset root")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-persons", type=int, default=10)
    p.add_argument("--n-records", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=10)
    return p.parse_args()


def find_qualifying_records(speaker_dir: Path, n_records: int, n_samples: int):
    """Return up to n_records utterance dirs that each contain >= n_samples wavs."""
    records = []
    for utt_dir in sorted(speaker_dir.iterdir()):
        if not utt_dir.is_dir():
            continue
        wavs = sorted(utt_dir.glob("*.wav"))
        if len(wavs) >= n_samples:
            records.append((utt_dir, wavs))
        if len(records) == n_records:
            break
    return records


def copy_split(wavs, n_samples, dst_dev_record, dst_test_record):
    selected = wavs[:n_samples]
    dev_files = selected[:-2]
    test_files = selected[-2:]
    dst_dev_record.mkdir(parents=True, exist_ok=True)
    dst_test_record.mkdir(parents=True, exist_ok=True)
    for f in dev_files:
        shutil.copy2(f, dst_dev_record / f.name)
    for f in test_files:
        shutil.copy2(f, dst_test_record / f.name)
    return len(dev_files), len(test_files)


def main():
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)

    speakers = [d for d in src.iterdir() if d.is_dir()]
    rng = random.Random(args.seed)
    rng.shuffle(speakers)

    accepted = []
    for spk in speakers:
        records = find_qualifying_records(spk, args.n_records, args.n_samples)
        if len(records) >= args.n_records:
            accepted.append((spk, records[:args.n_records]))
        if len(accepted) == args.n_persons:
            break

    if len(accepted) < args.n_persons:
        raise RuntimeError(
            f"Only found {len(accepted)} qualifying speakers (need {args.n_persons}). "
            "Try lowering --n-records or --n-samples."
        )

    print(f"{'Speaker':<14} {'Records':>7} {'Dev':>5} {'Test':>5} {'Total':>7}")
    print("-" * 40)
    for spk_dir, records in accepted:
        total_dev = total_test = 0
        for utt_dir, wavs in records:
            d, t = copy_split(
                wavs,
                args.n_samples,
                dst / "wav_dev" / spk_dir.name / utt_dir.name,
                dst / "wav_test" / spk_dir.name / utt_dir.name,
            )
            total_dev += d
            total_test += t
        print(f"{spk_dir.name:<14} {len(records):>7} {total_dev:>5} {total_test:>5} {total_dev+total_test:>7}")

    print(f"\nDone. Dataset written to: {dst.resolve()}")


if __name__ == "__main__":
    main()
