"""
prepare_dataset.py
==================
Generate a YAML manifest of which wav files to include for fingerprint training —
WITHOUT copying any audio.  The fingerprint generator reads paths from this manifest
and loads audio in place from datasets/vox1_cleaned/.

Split (open-set: dev and test speaker sets are disjoint, separate source dirs):
  dev  : N_DEV_PERSONS  random qualifying persons from wav_dev
           qualifying = >= dev_records sessions each with >= dev_samples wavs
           per person  = dev_records sessions x dev_samples wavs
  test : N_TEST_PERSONS random qualifying persons from wav_test
           qualifying = >= test_records sessions each with >= test_samples wavs
           per person  = test_records sessions x test_samples wavs

Persons that don't qualify are skipped (we keep drawing until the target count is met).
Person selection is randomised (seeded); within a person, the first qualifying records
(sorted) and the first wavs (sorted) in each are taken, so output is deterministic per seed.

Output: training/be_ann_w_fingerprints/dataset_manifest.yaml
  dataset_root, seed, and per-split {wav_subdir, n_persons, records_per_person,
  samples_per_record, persons: {person_id: {record_id: [wav, ...]}}}.
  Reconstruct a path as: {dataset_root}/{wav_subdir}/{person_id}/{record_id}/{wav}

Usage:
  cd <repo_root>
  python training/be_ann_w_fingerprints/prepare_dataset.py
"""

import argparse
import os
import random
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="datasets/vox1_cleaned")
    p.add_argument("--dev-subdir",  default="wav_dev")
    p.add_argument("--test-subdir", default="wav_test")
    p.add_argument("--out", default="training/be_ann_w_fingerprints/dataset_manifest.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-dev-persons",  type=int, default=600)
    p.add_argument("--dev-records",    type=int, default=4, help="sessions per dev person")
    p.add_argument("--dev-samples",    type=int, default=8, help="wavs per dev session")
    p.add_argument("--n-test-persons", type=int, default=20)
    p.add_argument("--test-records",   type=int, default=4, help="sessions per test person")
    p.add_argument("--test-samples",   type=int, default=4, help="wavs per test session")
    return p.parse_args()


def select_person_records(person_dir, n_records, n_samples):
    """Return {record_id: [wav, ...]} for the first n_records sessions that each have
    >= n_samples wavs (first n_samples wavs taken, sorted), or None if too few qualify."""
    selected = {}
    for record_dir in sorted(d for d in person_dir.iterdir() if d.is_dir()):
        wavs = sorted(f.name for f in record_dir.glob("*.wav"))
        if len(wavs) >= n_samples:
            selected[record_dir.name] = wavs[:n_samples]
        if len(selected) == n_records:
            return selected
    return None


def build_split(src_dir, n_persons, n_records, n_samples, rng):
    """Randomly draw qualifying persons until n_persons collected.

    Returns {person_id: {record_id: [wav, ...]}}.
    """
    persons = sorted(d for d in src_dir.iterdir() if d.is_dir())
    rng.shuffle(persons)

    split = {}
    for person_dir in persons:
        recs = select_person_records(person_dir, n_records, n_samples)
        if recs is not None:
            split[person_dir.name] = recs
        if len(split) == n_persons:
            break

    if len(split) < n_persons:
        raise RuntimeError(
            f"Only {len(split)} qualifying persons in {src_dir} "
            f"(need {n_persons}; rule = >={n_records} records each with >={n_samples} wavs)."
        )
    return split


def main():
    args = parse_args()
    root = Path(args.dataset_root)
    rng  = random.Random(args.seed)

    dev = build_split(root / args.dev_subdir,
                      args.n_dev_persons, args.dev_records, args.dev_samples, rng)
    test = build_split(root / args.test_subdir,
                       args.n_test_persons, args.test_records, args.test_samples, rng)

    manifest = {
        "dataset_root": args.dataset_root,
        "seed": args.seed,
        "dev": {
            "wav_subdir": args.dev_subdir,
            "n_persons": args.n_dev_persons,
            "records_per_person": args.dev_records,
            "samples_per_record": args.dev_samples,
            "persons": dev,
        },
        "test": {
            "wav_subdir": args.test_subdir,
            "n_persons": args.n_test_persons,
            "records_per_person": args.test_records,
            "samples_per_record": args.test_samples,
            "persons": test,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, default_flow_style=False)

    def _count(split):
        return sum(len(wavs) for recs in split.values() for wavs in recs.values())

    print(f"Wrote manifest → {out_path}")
    print(f"  dev  : {len(dev):>3} persons x {args.dev_records} records x "
          f"{args.dev_samples} wavs = {_count(dev)} samples")
    print(f"  test : {len(test):>3} persons x {args.test_records} records x "
          f"{args.test_samples} wavs = {_count(test)} samples")
    print(f"  total samples: {_count(dev) + _count(test)}")


if __name__ == "__main__":
    main()
