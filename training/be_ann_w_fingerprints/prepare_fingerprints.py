"""
prepare_fingerprints.py
=======================
Manifest-driven fingerprint generator for the reduced 4-neuron architecture with
silence masking. Parallel (multiprocessing), sharded, and crash-proof.

Per-sample output (see _fingerprint_core.fingerprint_to_sample):
  weights         (4, 33, 384)  float16  raw type-images, silent cells zeroed
  input_activity  (384,)        float16  per-sample [0,1]
  hidden_activity (384,)        float16  per-sample [0,1]

Sharding: entries are enumerated in a stable order (sorted person/record/wav) and
split into fixed-size shards. A shard is the unit of work, output, and crash-resume:
  {split}_fingerprints_{shard:04d}.npz
Within a shard, fingerprints are written incrementally to a memmap + meta sidecar so a
mid-shard crash resumes from the last checkpoint. A shard whose final npz exists is
skipped. One worker owns a shard at a time, so there is never write contention.

The manifest's per-split `generated:` list is updated (atomically, by the main process
only) as each shard completes — it is a convenience cache; the shard npz files on disk
are the ground truth for resume.

Usage:
  cd <repo_root>
  ./venv/bin/python training/be_ann_w_fingerprints/prepare_fingerprints.py
  ./venv/bin/python training/be_ann_w_fingerprints/prepare_fingerprints.py --workers 2 --shard-size 2000
  ./venv/bin/python training/be_ann_w_fingerprints/prepare_fingerprints.py --workers 2 --shard-size 50 --limit 120   # smoke
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.join(SCRIPT_DIR, '..', '..')
# Make the sibling core module importable whether run by path or re-imported by a
# spawned worker. Import the core FIRST so its BLAS-thread pinning beats numpy.
sys.path.insert(0, SCRIPT_DIR)
import _fingerprint_core as C  # noqa: E402

import argparse        # noqa: E402
import multiprocessing as mp  # noqa: E402
import time            # noqa: E402

import numpy as np     # noqa: E402
import yaml            # noqa: E402
DEFAULT_MANIFEST = os.path.join(SCRIPT_DIR, 'dataset_manifest.yaml')

CHECKPOINT_EVERY = 20   # flush memmap + meta every N fingerprints within a shard
_LOG_EVERY = 10         # per-worker progress print cadence (set in _init_worker)


# ═══════════════════════════════════════════════════════════════════════════════
# Manifest discovery
# ═══════════════════════════════════════════════════════════════════════════════

def discover_split(split_blk, dataset_root):
    """Ordered per-wav entries for one manifest split block.

    Returns list of {person_id, record_id, wav, label, wav_path}, sorted
    deterministically by person / record / wav.
    """
    wav_subdir = split_blk['wav_subdir']
    persons    = split_blk['persons']
    entries = []
    for person_id in sorted(persons):
        records = persons[person_id]
        for record_id in sorted(records):
            for wav in sorted(records[record_id]):
                entries.append({
                    'person_id': person_id,
                    'record_id': record_id,
                    'wav':       wav,
                    'label':     f"{person_id}/{record_id}/{wav}",
                    'wav_path':  os.path.join(dataset_root, wav_subdir,
                                              person_id, record_id, wav),
                })
    return entries


def shard_paths(split, shard_idx):
    base = os.path.join(SCRIPT_DIR, f"{split}_fingerprints_{shard_idx:04d}.npz")
    return {'out': base, 'mmap': base + '.mmap', 'meta': base + '.meta.npz'}


# ═══════════════════════════════════════════════════════════════════════════════
# Worker
# ═══════════════════════════════════════════════════════════════════════════════

_H = None   # per-process Brian2 network handle


def _init_worker(log_every):
    global _H, _LOG_EVERY
    _LOG_EVERY = log_every
    _H = C.build_network()


def process_shard(task):
    """Generate all fingerprints for one shard; write the shard npz. Resumable.

    Returns (split, shard_idx, labels_written, status).
    """
    split, shard_idx, n_shards, entries = task
    paths = shard_paths(split, shard_idx)
    tag = f"{split} shard {shard_idx + 1:02d}/{n_shards}"
    if os.path.exists(paths['out']):
        return (split, shard_idx, [e['label'] for e in entries], 'exists')

    n = len(entries)
    mmap_shape = (n,) + C.WEIGHTS_SHAPE

    count = 0
    ia, ha, pids, rids, labels = [], [], [], [], []

    # ── Resume mid-shard from memmap + meta if compatible ─────────────────────
    if os.path.exists(paths['mmap']) and os.path.exists(paths['meta']):
        m = np.load(paths['meta'], allow_pickle=True)
        if int(m['n']) == n:
            count  = int(m['count'])
            ia     = [r for r in m['input_activity']]
            ha     = [r for r in m['hidden_activity']]
            pids   = m['person_ids'].tolist()
            rids   = m['record_ids'].tolist()
            labels = m['labels'].tolist()
            weights_mm = np.memmap(paths['mmap'], dtype='float16', mode='r+', shape=mmap_shape)
        else:
            weights_mm = np.memmap(paths['mmap'], dtype='float16', mode='w+', shape=mmap_shape)
    else:
        weights_mm = np.memmap(paths['mmap'], dtype='float16', mode='w+', shape=mmap_shape)

    def _checkpoint():
        weights_mm.flush()
        np.savez(
            paths['meta'],
            n=n, count=count,
            input_activity=(np.stack(ia) if ia else np.zeros((0, C.N_IN), np.float16)),
            hidden_activity=(np.stack(ha) if ha else np.zeros((0, C.N_H), np.float16)),
            person_ids=np.array(pids), record_ids=np.array(rids), labels=np.array(labels),
        )

    produced = set(labels)
    if count:
        print(f"  [{tag}] resuming at {count}/{n} samples", flush=True)
    t_sh, since = time.time(), 0
    for e in entries:
        if e['label'] in produced:
            continue
        out = C.train_fingerprint(_H, e['wav_path'])
        if out is None:
            continue   # unencodable wav — not recorded
        w, iaa, haa = C.fingerprint_to_sample(*out)
        weights_mm[count] = w
        ia.append(iaa); ha.append(haa)
        pids.append(e['person_id']); rids.append(e['record_id']); labels.append(e['label'])
        produced.add(e['label'])
        count += 1
        since += 1
        if count % _LOG_EVERY == 0:
            rate = (time.time() - t_sh) / max(since, 1)
            print(f"  [{tag}] {count}/{n} samples ({100 * count // n}%)  "
                  f"{rate:.1f}s/sample", flush=True)
        if count % CHECKPOINT_EVERY == 0:
            _checkpoint()

    # ── Finalise: write the shard npz, drop the resume sidecars ───────────────
    np.savez(
        paths['out'],
        weights=np.ascontiguousarray(weights_mm[:count]),
        input_activity=(np.stack(ia) if ia else np.zeros((0, C.N_IN), np.float16)),
        hidden_activity=(np.stack(ha) if ha else np.zeros((0, C.N_H), np.float16)),
        person_ids=np.array(pids), record_ids=np.array(rids), labels=np.array(labels),
    )
    del weights_mm
    for p in (paths['mmap'], paths['meta']):
        if os.path.exists(p):
            os.unlink(p)
    return (split, shard_idx, labels, 'done')


# ═══════════════════════════════════════════════════════════════════════════════
# Manifest `generated:` status (main process only)
# ═══════════════════════════════════════════════════════════════════════════════

def update_manifest_generated(manifest_path, split, new_labels):
    """Atomically merge new_labels into manifest[split]['generated']."""
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)
    blk = manifest.get(split)
    if not isinstance(blk, dict):
        return
    done = set(blk.get('generated') or [])
    done.update(new_labels)
    blk['generated'] = sorted(done)
    tmp = manifest_path + '.tmp'
    with open(tmp, 'w') as f:
        yaml.safe_dump(manifest, f, sort_keys=False, default_flow_style=False)
    os.replace(tmp, manifest_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def build_tasks(manifest, dataset_root, shard_size, limit):
    """Return (tasks, stats). tasks = list of (split, shard_idx, entries) for
    shards whose final npz does not yet exist."""
    tasks, stats = [], {}
    for split in ('dev', 'test'):
        if split not in manifest:
            continue
        entries = discover_split(manifest[split], dataset_root)
        if limit:
            entries = entries[:limit]
        n_shards = (len(entries) + shard_size - 1) // shard_size
        done_shards = 0
        for s in range(n_shards):
            chunk = entries[s * shard_size:(s + 1) * shard_size]
            if os.path.exists(shard_paths(split, s)['out']):
                done_shards += 1
                continue
            tasks.append((split, s, n_shards, chunk))
        stats[split] = {'entries': len(entries), 'shards': n_shards, 'done': done_shards}
    return tasks, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=DEFAULT_MANIFEST)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help='parallel worker processes (default = physical cores)')
    ap.add_argument('--shard-size', type=int, default=2000,
                    help='fingerprints per shard npz')
    ap.add_argument('--limit', type=int, default=0,
                    help='process at most N entries per split (0 = all; smoke testing)')
    ap.add_argument('--log-every', type=int, default=10,
                    help='print within-shard progress every N samples')
    args = ap.parse_args()

    t_start = time.time()

    with open(args.manifest) as f:
        manifest = yaml.safe_load(f)
    dataset_root = os.path.join(REPO_ROOT, manifest['dataset_root'])

    tasks, stats = build_tasks(manifest, dataset_root, args.shard_size, args.limit)

    print(f"{'='*64}")
    for split, st in stats.items():
        print(f"{split:>4}: {st['entries']:>7} entries, {st['shards']:>4} shards "
              f"({st['done']} already complete)")
    print(f"workers={args.workers}  shard_size={args.shard_size}"
          + (f"  limit={args.limit}/split" if args.limit else ""))
    print(f"shards to process: {len(tasks)}")
    print(f"{'='*64}")

    if not tasks:
        print("Nothing to do — all shards complete.")
        return

    # Warm the Cython codegen cache once so spawned workers reuse the compiled
    # extension instead of compiling concurrently on first build.
    print("warming codegen cache ...")
    C.build_network()

    ctx = mp.get_context('spawn')
    n_done = 0
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(args.log_every,)) as pool:
        for split, shard_idx, labels, status in pool.imap_unordered(process_shard, tasks):
            if status == 'done':
                update_manifest_generated(args.manifest, split, labels)
            n_done += 1
            print(f"[{n_done}/{len(tasks)}] {split} shard {shard_idx:04d}: "
                  f"{len(labels)} fingerprints ({status})  "
                  f"[{time.time()-t_start:.0f}s elapsed]")

    print(f"\nDone — {len(tasks)} shard(s) in {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
