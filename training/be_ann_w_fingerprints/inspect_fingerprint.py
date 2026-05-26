"""Temporary script: inspect one fingerprint tensor from dev_fingerprints.npz."""

import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH   = os.path.join(SCRIPT_DIR, 'dev_fingerprints.npz')

data = np.load(NPZ_PATH, allow_pickle=True)
fp         = data['fingerprints'][0]   # (700, 700) float32
person_id  = data['person_ids'][0]
record_id  = data['record_ids'][0]

print(f"person_id : {person_id}")
print(f"record_id : {record_id}")
print(f"shape     : {fp.shape}")
print(f"dtype     : {fp.dtype}")
print(f"min       : {fp.min():.6f}")
print(f"max       : {fp.max():.6f}")
print(f"mean      : {fp.mean():.6f}")
print(f"std       : {fp.std():.6f}")
print(f"sparsity  : {(fp == 0).mean()*100:.1f}% zeros")

print("\n--- top-left 10×10 corner ---")
np.set_printoptions(precision=4, suppress=True, linewidth=120)
print(fp[:10, :10])

print("\n--- row sums (first 20 rows) ---")
row_sums = fp.sum(axis=1)
print(row_sums[:20])
print(f"row-sum range: [{row_sums.min():.4f}, {row_sums.max():.4f}]  mean: {row_sums.mean():.4f}")

print("\n--- ASCII histogram of non-zero values ---")
nz = fp[fp > 0].ravel()
if len(nz):
    bins = np.linspace(nz.min(), nz.max(), 11)
    counts, _ = np.histogram(nz, bins=bins)
    max_count  = counts.max()
    bar_width  = 40
    for i, (lo, hi, c) in enumerate(zip(bins, bins[1:], counts)):
        bar = '#' * int(c / max_count * bar_width)
        print(f"  [{lo:.3f}-{hi:.3f}]  {bar:<{bar_width}}  {c}")
else:
    print("  (all zeros)")
