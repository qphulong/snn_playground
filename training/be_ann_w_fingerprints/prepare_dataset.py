"""
Prepare datasets/vox1_100person_fingerprint from datasets/vox1_cleaned/wav_dev.

Selects N_PERSONS random speakers that each have enough qualifying records:
  - n_shared records with >= n_dev_samples + n_test_samples wavs
      -> wav_dev gets files [0 : n_dev_samples]
      -> wav_test gets files [n_dev_samples : n_dev_samples + n_test_samples]
  - (n_dev_records - n_shared) dev-only records with >= n_dev_samples wavs
      -> wav_dev gets files [0 : n_dev_samples]
  - n_test_only_records records (not in dev) with >= n_test_samples wavs
      -> wav_test gets files [0 : n_test_samples]

Default target:
  dev  : 100 people x 4 records x 8 samples
  test : 100 people x 4 records x 2 samples
           (2 records shared with dev using next 2 files, 2 records test-only)
"""

import argparse
import random
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="datasets/vox1_cleaned/wav_dev")
    p.add_argument("--dst", default="datasets/vox1_100person_fingerprint")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-persons", type=int, default=100)
    p.add_argument("--n-dev-records", type=int, default=4,
                   help="Total records per person in dev")
    p.add_argument("--n-dev-samples", type=int, default=8,
                   help="Wav files per dev record")
    p.add_argument("--n-shared", type=int, default=2,
                   help="Dev records that also appear in test (files after dev samples)")
    p.add_argument("--n-test-only-records", type=int, default=2,
                   help="Additional test-only records per person (not in dev)")
    p.add_argument("--n-test-samples", type=int, default=2,
                   help="Wav files per test record (shared and test-only)")
    return p.parse_args()


def find_records_for_speaker(speaker_dir, n_dev, n_dev_samples, n_shared,
                              n_test_samples, n_test_only):
    """Return (shared, dev_only, test_only) or None if speaker doesn't qualify.

    shared    -- first n_shared records with >= n_dev_samples + n_test_samples wavs
    dev_only  -- next (n_dev - n_shared) records with >= n_dev_samples wavs
    test_only -- next n_test_only records (not in dev) with >= n_test_samples wavs
    """
    all_records = [
        (d, sorted(d.glob("*.wav")))
        for d in sorted(speaker_dir.iterdir()) if d.is_dir()
    ]
    used = set()
    shared, dev_only, test_only = [], [], []

    for utt_dir, wavs in all_records:
        if len(shared) == n_shared:
            break
        if len(wavs) >= n_dev_samples + n_test_samples:
            shared.append((utt_dir, wavs))
            used.add(utt_dir)

    if len(shared) < n_shared:
        return None

    n_dev_only = n_dev - n_shared
    for utt_dir, wavs in all_records:
        if utt_dir in used:
            continue
        if len(dev_only) == n_dev_only:
            break
        if len(wavs) >= n_dev_samples:
            dev_only.append((utt_dir, wavs))
            used.add(utt_dir)

    if len(dev_only) < n_dev_only:
        return None

    for utt_dir, wavs in all_records:
        if utt_dir in used:
            continue
        if len(test_only) == n_test_only:
            break
        if len(wavs) >= n_test_samples:
            test_only.append((utt_dir, wavs))
            used.add(utt_dir)

    if len(test_only) < n_test_only:
        return None

    return shared, dev_only, test_only


def copy_files(wavs, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in wavs:
        shutil.copy2(f, dst_dir / f.name)


def main():
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)

    speakers = [d for d in src.iterdir() if d.is_dir()]
    rng = random.Random(args.seed)
    rng.shuffle(speakers)

    accepted = []
    for spk in speakers:
        result = find_records_for_speaker(
            spk,
            n_dev=args.n_dev_records,
            n_dev_samples=args.n_dev_samples,
            n_shared=args.n_shared,
            n_test_samples=args.n_test_samples,
            n_test_only=args.n_test_only_records,
        )
        if result is not None:
            accepted.append((spk, result))
        if len(accepted) == args.n_persons:
            break

    if len(accepted) < args.n_persons:
        raise RuntimeError(
            f"Only found {len(accepted)} qualifying speakers (need {args.n_persons}). "
            "Try lowering requirements."
        )

    header = (f"{'Speaker':<14} {'Shared':>6} {'DevOnly':>7} {'TestOnly':>8} "
              f"{'DevFiles':>8} {'TestFiles':>9}")
    print(header)
    print("-" * len(header))

    for spk_dir, (shared, dev_only, test_only) in accepted:
        total_dev = total_test = 0

        for utt_dir, wavs in shared:
            dev_wavs  = wavs[:args.n_dev_samples]
            test_wavs = wavs[args.n_dev_samples:args.n_dev_samples + args.n_test_samples]
            copy_files(dev_wavs,  dst / "wav_dev"  / spk_dir.name / utt_dir.name)
            copy_files(test_wavs, dst / "wav_test" / spk_dir.name / utt_dir.name)
            total_dev  += len(dev_wavs)
            total_test += len(test_wavs)

        for utt_dir, wavs in dev_only:
            dev_wavs = wavs[:args.n_dev_samples]
            copy_files(dev_wavs, dst / "wav_dev" / spk_dir.name / utt_dir.name)
            total_dev += len(dev_wavs)

        for utt_dir, wavs in test_only:
            test_wavs = wavs[:args.n_test_samples]
            copy_files(test_wavs, dst / "wav_test" / spk_dir.name / utt_dir.name)
            total_test += len(test_wavs)

        print(f"{spk_dir.name:<14} {len(shared):>6} {len(dev_only):>7} {len(test_only):>8} "
              f"{total_dev:>8} {total_test:>9}")

    print(f"\nDone. Dataset written to: {dst.resolve()}")
    print(f"  Persons : {len(accepted)}")
    print(f"  Dev     : {len(accepted)} persons x "
          f"{args.n_dev_records} records x {args.n_dev_samples} samples")
    print(f"  Test    : {len(accepted)} persons x "
          f"{args.n_shared + args.n_test_only_records} records x {args.n_test_samples} samples "
          f"({args.n_shared} shared, {args.n_test_only_records} test-only)")


if __name__ == "__main__":
    main()
