"""
dummy_fingerprint_analysis.py
=============================
Render the dummy fingerprint output (dummy_fingerprints.npz, produced by
prepare_fingerprints_dummy.py) as PNGs, for the reduced 4-neuron architecture
with silence masking.

dummy_fingerprints.npz holds the production per-sample format:
  weights         (M, 4, 33, 384)  float16  raw type-images, silent cells zeroed
  input_activity  (M, 384)         float16  per-sample [0,1]
  hidden_activity (M, 384)         float16  per-sample [0,1]
  person_ids / record_ids / labels (M,)     str

For the chosen sample (--sample, default 0) this writes:
  weights_type{t}_{name}.png   one per input-neuron type (4 images, 33 x 384, magma)
  activity.png                 bar plots of input_activity (384) and hidden_activity (384)

Output:
  training/be_ann_w_fingerprints/dummy_fingerprint_pngs/

Usage:
  cd <repo_root>
  ./venv/bin/python training/be_ann_w_fingerprints/dummy_fingerprint_analysis.py
  ./venv/bin/python training/be_ann_w_fingerprints/dummy_fingerprint_analysis.py --sample 3
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH   = os.path.join(SCRIPT_DIR, 'dummy_fingerprints.npz')
OUT_DIR    = os.path.join(SCRIPT_DIR, 'dummy_fingerprint_pngs')

# Within-channel input-neuron types, in axis-1 order: 2 sustained, 1 onset, 1 phase.
TYPE_NAMES = ['sus0', 'sus1', 'onset0', 'phase0']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=0, help='which sample to render')
    args = ap.parse_args()

    data    = np.load(NPZ_PATH, allow_pickle=True)
    weights = data['weights'].astype(np.float32)        # (M, 4, 33, 384)
    in_act  = data['input_activity'].astype(np.float32) # (M, 384)
    hid_act = data['hidden_activity'].astype(np.float32)# (M, 384)
    labels  = data['labels']

    M = weights.shape[0]
    assert weights.shape[1:] == (len(TYPE_NAMES), 33, 384), \
        f"expected (M, 4, 33, 384), got {weights.shape}"
    s = args.sample
    if not (0 <= s < M):
        raise SystemExit(f"--sample {s} out of range (have {M} samples)")

    os.makedirs(OUT_DIR, exist_ok=True)
    label = str(labels[s])
    print(f"Samples: {M}  |  rendering sample {s}: {label}")
    print(f"Writing PNGs → {os.path.relpath(OUT_DIR)}\n")

    # ── Weight type-images ────────────────────────────────────────────────────
    w = weights[s]                                       # (4, 33, 384)
    for t, name in enumerate(TYPE_NAMES):
        img  = w[t]                                      # (33, 384)
        path = os.path.join(OUT_DIR, f'weights_type{t}_{name}.png')
        plt.imsave(path, img, cmap='magma')
        print(f"  type{t} {name:<7} raw[{img.min():.4f},{img.max():.4f}]  "
              f"zeroed {(img == 0).sum()}/{img.size}  → {os.path.basename(path)}")

    # ── Activity bars ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    for ax, vec, title, color in (
        (axes[0], in_act[s],  f"input_activity  (silent={(in_act[s]==0).sum()}/384)",  "steelblue"),
        (axes[1], hid_act[s], f"hidden_activity (silent={(hid_act[s]==0).sum()}/384)", "mediumseagreen"),
    ):
        ax.bar(np.arange(len(vec)), vec, width=1.0, color=color, linewidth=0)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Neuron index"); ax.set_ylabel("Activity [0,1]")
        ax.set_xlim(-0.5, len(vec) - 0.5); ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"Per-neuron activity — {label}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    act_path = os.path.join(OUT_DIR, 'activity.png')
    fig.savefig(act_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {os.path.basename(act_path)}")

    print(f"\nWrote {len(TYPE_NAMES) + 1} PNG(s) to {os.path.relpath(OUT_DIR)}")


if __name__ == '__main__':
    main()
