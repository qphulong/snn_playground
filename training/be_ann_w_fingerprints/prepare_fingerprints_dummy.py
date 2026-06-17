"""
prepare_fingerprints_dummy.py
=============================
Single-process smoke test for the reduced 4-neuron architecture + silence masking.
Takes the first --persons speakers (default 2) from the dev split of the manifest,
runs every one of their wavs through the same core pipeline as the production
generator, and writes dummy_fingerprints.npz in the EXACT production per-sample format:

  weights         (M, 4, 33, 384)  float16
  input_activity  (M, 384)         float16   per-sample [0,1]
  hidden_activity (M, 384)         float16   per-sample [0,1]
  person_ids / record_ids / labels (M,)      str

Use this to eyeball shapes / value ranges / silent-neuron counts / timing before
launching the full parallel run with prepare_fingerprints.py.

Usage:
  cd <repo_root>
  ./venv/bin/python training/be_ann_w_fingerprints/prepare_fingerprints_dummy.py
  ./venv/bin/python training/be_ann_w_fingerprints/prepare_fingerprints_dummy.py --persons 2
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import _fingerprint_core as C  # noqa: E402  (import core first; pins BLAS threads)

import argparse        # noqa: E402
import time            # noqa: E402

import numpy as np     # noqa: E402
import yaml            # noqa: E402

REPO_ROOT        = os.path.join(SCRIPT_DIR, '..', '..')
DEFAULT_MANIFEST = os.path.join(SCRIPT_DIR, 'dataset_manifest.yaml')
OUT_PATH         = os.path.join(SCRIPT_DIR, 'dummy_fingerprints.npz')


def discover_first_persons(split_blk, dataset_root, n_persons):
    """Ordered entries for the first n_persons speakers of a split block."""
    wav_subdir = split_blk['wav_subdir']
    persons    = split_blk['persons']
    entries = []
    for person_id in sorted(persons)[:n_persons]:
        records = persons[person_id]
        for record_id in sorted(records):
            for wav in sorted(records[record_id]):
                entries.append({
                    'person_id': person_id,
                    'record_id': record_id,
                    'label':     f"{person_id}/{record_id}/{wav}",
                    'wav_path':  os.path.join(dataset_root, wav_subdir,
                                              person_id, record_id, wav),
                })
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=DEFAULT_MANIFEST)
    ap.add_argument('--persons', type=int, default=2, help='first N dev speakers')
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = yaml.safe_load(f)
    dataset_root = os.path.join(REPO_ROOT, manifest['dataset_root'])
    entries = discover_first_persons(manifest['dev'], dataset_root, args.persons)

    print(f"{'='*64}")
    print(f"Dummy run: first {args.persons} dev person(s) → {len(entries)} wav(s)")
    print(f"{'='*64}")

    h = C.build_network()

    weights, ia, ha, pids, rids, labels = [], [], [], [], [], []
    for i, e in enumerate(entries):
        t0 = time.time()
        out = C.train_fingerprint(h, e['wav_path'])
        if out is None:
            continue
        fp, in_counts, hid_counts = out
        w, iaa, haa = C.fingerprint_to_sample(fp, in_counts, hid_counts)
        weights.append(w); ia.append(iaa); ha.append(haa)
        pids.append(e['person_id']); rids.append(e['record_id']); labels.append(e['label'])
        print(f"  [{i+1:03d}/{len(entries)}] {e['label']}  "
              f"in_silent={int((in_counts==0).sum())}/{C.N_IN} "
              f"hid_silent={int((hid_counts==0).sum())}/{C.N_H}  ({time.time()-t0:.1f}s)")

    if not weights:
        print("No fingerprints produced.")
        return

    weights = np.stack(weights); ia = np.stack(ia); ha = np.stack(ha)
    np.savez(
        OUT_PATH,
        weights=weights, input_activity=ia, hidden_activity=ha,
        person_ids=np.array(pids), record_ids=np.array(rids), labels=np.array(labels),
    )

    zeroed = int((weights == 0).sum())
    print(f"\nSaved → {os.path.relpath(OUT_PATH)}")
    print(f"  weights         {weights.shape} {weights.dtype}  "
          f"range [{weights.min():.4f}, {weights.max():.4f}]  zeroed {zeroed}/{weights.size}")
    print(f"  input_activity  {ia.shape} {ia.dtype}  range [{ia.min():.3f}, {ia.max():.3f}]")
    print(f"  hidden_activity {ha.shape} {ha.dtype}  range [{ha.min():.3f}, {ha.max():.3f}]")
    print(f"  samples         {len(labels)}")


if __name__ == '__main__':
    main()
