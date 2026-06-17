"""
prepare_dataset.py
==================
Generate a YAML manifest of which wav files to include for fingerprint training —
WITHOUT copying any audio.  The fingerprint generator reads paths from this manifest
and loads audio in place from datasets/vox1_cleaned/.

Enumerates the FULL VoxCeleb1 dataset: every speaker, every recording session, capped
at --per-session wavs per session (default 2). Maximising session breadth (all sessions,
few wavs each) is what teaches the embedding channel/recording-invariance; capping per
session bounds the highly-correlated within-session redundancy.

The manifest doubles as the generation-status file: each split carries a `generated:`
list of completed `person/record/wav` labels. prepare_fingerprints.py appends to it as
it produces fingerprints, and resumes by skipping entries already in the set. Re-running
this script is NON-DESTRUCTIVE: it preserves existing `generated` entries that still
appear in the freshly enumerated list (use --reset to wipe them).

Split (open-set: dev and test speaker sets are disjoint, separate source dirs):
  dev  : all speakers under wav_dev
  test : all speakers under wav_test
For each speaker every session is included; within a session the first --per-session wavs
(sorted) are taken, so output is fully deterministic.

Output: training/be_ann_w_fingerprints/dataset_manifest.yaml
  dataset_root, and per-split {wav_subdir, n_persons, per_session,
  persons: {person_id: {record_id: [wav, ...]}}, generated: [label, ...]}.
  Reconstruct a path as: {dataset_root}/{wav_subdir}/{person_id}/{record_id}/{wav}

Usage:
  cd <repo_root>
  python training/be_ann_w_fingerprints/prepare_dataset.py                 # default 2/session
  python training/be_ann_w_fingerprints/prepare_dataset.py --per-session 4
  python training/be_ann_w_fingerprints/prepare_dataset.py --reset          # wipe generated status
"""

import argparse
import os
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="datasets/vox1_cleaned")
    p.add_argument("--dev-subdir",  default="wav_dev")
    p.add_argument("--test-subdir", default="wav_test")
    p.add_argument("--out", default="training/be_ann_w_fingerprints/dataset_manifest.yaml")
    p.add_argument("--per-session", type=int, default=2,
                   help="max wavs taken per recording session (first N, sorted)")
    p.add_argument("--reset", action="store_true",
                   help="discard any existing `generated` status in the output manifest")
    return p.parse_args()


def build_split(src_dir, per_session):
    """Enumerate every speaker and session under src_dir.

    Returns {person_id: {record_id: [wav, ...]}} taking at most per_session wavs
    (first, sorted) per session. Sessions/persons with no wavs are skipped.
    """
    if not src_dir.is_dir():
        raise RuntimeError(f"source dir not found: {src_dir}")

    split = {}
    for person_dir in sorted(d for d in src_dir.iterdir() if d.is_dir()):
        records = {}
        for record_dir in sorted(d for d in person_dir.iterdir() if d.is_dir()):
            wavs = sorted(f.name for f in record_dir.glob("*.wav"))[:per_session]
            if wavs:
                records[record_dir.name] = wavs
        if records:
            split[person_dir.name] = records
    return split


def labels_of(split):
    """Set of `person/record/wav` labels in a split's persons mapping."""
    return {f"{person}/{record}/{wav}"
            for person, records in split.items()
            for record, wavs in records.items()
            for wav in wavs}


def load_prev_generated(out_path):
    """Return {split_name: set(labels)} of previously-completed fingerprints, or {}."""
    if not out_path.exists():
        return {}
    with open(out_path) as f:
        prev = yaml.safe_load(f) or {}
    out = {}
    for split_name in ("dev", "test"):
        blk = prev.get(split_name)
        if isinstance(blk, dict):
            out[split_name] = set(blk.get("generated", []) or [])
    return out


def count_samples(split):
    return sum(len(wavs) for records in split.values() for wavs in records.values())


def main():
    args = parse_args()
    root = Path(args.dataset_root)

    dev  = build_split(root / args.dev_subdir,  args.per_session)
    test = build_split(root / args.test_subdir, args.per_session)

    prev = {} if args.reset else load_prev_generated(Path(args.out))

    def generated_for(split_name, split):
        # Keep only previously-done labels that still exist in the new enumeration.
        kept = prev.get(split_name, set()) & labels_of(split)
        return sorted(kept)

    manifest = {
        "dataset_root": args.dataset_root,
        "dev": {
            "wav_subdir":  args.dev_subdir,
            "n_persons":   len(dev),
            "per_session": args.per_session,
            "persons":     dev,
            "generated":   generated_for("dev", dev),
        },
        "test": {
            "wav_subdir":  args.test_subdir,
            "n_persons":   len(test),
            "per_session": args.per_session,
            "persons":     test,
            "generated":   generated_for("test", test),
        },
    }

    # Atomic write: dump to a temp file then replace, so a crash mid-write can't
    # corrupt the only manifest/status file.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, default_flow_style=False)
    os.replace(tmp_path, out_path)

    print(f"Wrote manifest → {out_path}")
    for name, split in (("dev", dev), ("test", test)):
        kept = len(manifest[name]["generated"])
        print(f"  {name:>4} : {len(split):>4} persons, "
              f"{sum(len(r) for r in split.values()):>6} sessions, "
              f"{count_samples(split):>7} samples"
              + (f"  ({kept} generated preserved)" if kept else ""))
    print(f"  per_session cap: {args.per_session}")
    print(f"  total samples  : {count_samples(dev) + count_samples(test)}")


if __name__ == "__main__":
    main()
