"""
dummy_fingerprint_analysis.py
=============================
Render the dummy fingerprint tensor (dummy_fingerprint.npz) as PNGs.

The transform now lives in prepare_fingerprints_dummy.py, which saves the fingerprint
already in ANN-ready form:

  fingerprint : (2, 7, 33, 672) float32
    axis 0 (2)   norm channel: 0 = raw weight, 1 = z-scored per hidden column
    axis 1 (7)   within-channel input-neuron type (4 sustained, 2 onset, 1 phase)
    axis 2 (33)  channel offset -16..+16
    axis 3 (672) hidden neuron

This script just loads that tensor and writes one PNG per (type, norm) slice → 14 images,
each 33 x 672:
  *_raw.png      raw weight                          (cmap magma)
  *_zscore.png   z-scored per hidden column          (cmap coolwarm)

Output:
  training/be_ann_w_fingerprints/dummy_fingerprint_pngs/

Usage:
  cd <repo_root>
  python training/be_ann_w_fingerprints/dummy_fingerprint_analysis.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH   = os.path.join(SCRIPT_DIR, 'dummy_fingerprint.npz')
OUT_DIR    = os.path.join(SCRIPT_DIR, 'dummy_fingerprint_pngs')

# Within-channel input-neuron type names, in index order (axis 1 of the tensor):
# 4 sustained, 2 onset, 1 phase.
TYPE_NAMES = ['sus0', 'sus1', 'sus2', 'sus3', 'onset0', 'onset1', 'phase0']
TYPE_KIND  = ['sustained', 'sustained', 'sustained', 'sustained', 'onset', 'onset', 'phase']


def main():
    data   = np.load(NPZ_PATH, allow_pickle=True)
    tensor = data['fingerprint'].astype(np.float32)              # (2, 7, 33, 672)
    label  = str(data['label']) if 'label' in data.files else '(unknown)'
    os.makedirs(OUT_DIR, exist_ok=True)

    assert tensor.ndim == 4 and tensor.shape[:2] == (2, len(TYPE_NAMES)), \
        f"expected (2, 7, 33, 672), got {tensor.shape}"

    print(f"Tensor : {tensor.shape}  label={label}")
    print(f"Writing PNGs → {os.path.relpath(OUT_DIR)}\n")

    raw_all, z_all = tensor[0], tensor[1]                        # (7, 33, 672) each
    n = 0
    for t, (name, kind) in enumerate(zip(TYPE_NAMES, TYPE_KIND)):
        raw, zimg = raw_all[t], z_all[t]                        # (33, 672)

        raw_path = os.path.join(OUT_DIR, f'fp_type{t}_{name}_raw.png')
        z_path   = os.path.join(OUT_DIR, f'fp_type{t}_{name}_zscore.png')
        plt.imsave(raw_path, raw,  cmap='magma')
        plt.imsave(z_path,   zimg, cmap='coolwarm')
        n += 2

        print(f"  type{t} {name:<6} ({kind:<9})  "
              f"raw[{raw.min():.4f},{raw.max():.4f}]  "
              f"→ {os.path.basename(raw_path)}, {os.path.basename(z_path)}")

    print(f"\nWrote {n} PNG(s) to {os.path.relpath(OUT_DIR)}")


if __name__ == '__main__':
    main()
