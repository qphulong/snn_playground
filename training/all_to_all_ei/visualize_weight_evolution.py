"""
visualize_weight_evolution.py
-----------------------------
Single-purpose visualizer for the in->hid weight-matrix evolution captured by
train.py (every 500 ms, after normalisation) into
`weight_evolution_epoch_{NNN}.npz`.

For the chosen epoch and every sample in the file it renders TWO things:

1. Whole-matrix evolution — one PNG per snapshot frame of the full
   (N_IN x N_H) = 672 x 672 matrix.

2. Role-sliced evolution — the 672 input neurons are split BY NEURON ROLE
   (stride-7: input ordering is channel-major, 7 neurons per channel =
   4 sustained, 2 onset, 1 phase). Role k = input rows k, 7+k, 14+k, ...
   (96 rows, one per channel). Each of the 7 roles gets its OWN PNG series of
   (96 x N_H) heatmaps -> 7 x n_frames PNGs per sample.

Color scale is FIXED GLOBALLY across every frame/role/sample so the evolution
is directly comparable frame to frame.

Usage:
    python visualize_weight_evolution.py            # epoch 1 (default)
    python visualize_weight_evolution.py 0          # epoch 0
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── resolve paths / epoch ─────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EPOCH      = int(sys.argv[1]) if len(sys.argv) > 1 else 1

npz_path = os.path.join(SCRIPT_DIR, f"weight_evolution_epoch_{EPOCH:03d}.npz")
if not os.path.exists(npz_path):
    sys.exit(f"No evolution file for epoch {EPOCH}: {npz_path}\n"
             f"Run train.py first (it writes weight_evolution_epoch_*.npz).")

OUT_ROOT = os.path.join(SCRIPT_DIR, "weight_evo_vizs", f"epoch_{EPOCH}")

print(f"Loading {npz_path}")
data = np.load(npz_path, allow_pickle=True)

N_IN              = int(data["N_IN"])
N_H               = int(data["N_H"])
sustained_per_band = int(data["sustained_per_band"])
onset_per_band     = int(data["onset_per_band"])
phase_per_band     = int(data["phase_per_band"])
neurons_per_band   = sustained_per_band + onset_per_band + phase_per_band  # = 7
wmax_exc           = float(data["wmax_exc"])

# ── discover samples present in this file ─────────────────────────────────────

sample_keys = sorted(
    int(k[len("sample"):-len("_frames")])
    for k in data.files if k.startswith("sample") and k.endswith("_frames")
)
if not sample_keys:
    sys.exit(f"No sampleN_frames arrays found in {npz_path}")
print(f"Samples in file: {sample_keys}")

# ── role index map (stride-{neurons_per_band}) ────────────────────────────────
# Within each channel the order is: sustained_0..N, onset_0..N, phase_0..N.

roles = []  # (label, within_band_offset)
for i in range(sustained_per_band):
    roles.append((f"sustained_{i}", i))
for i in range(onset_per_band):
    roles.append((f"onset_{i}", sustained_per_band + i))
for i in range(phase_per_band):
    roles.append((f"phase_{i}", sustained_per_band + onset_per_band + i))
assert len(roles) == neurons_per_band, (len(roles), neurons_per_band)

# ── fixed global color scale across ALL frames / samples ──────────────────────

gmin, gmax = np.inf, -np.inf
for n in sample_keys:
    frames = data[f"sample{n}_frames"]
    if frames.size:
        gmin = min(gmin, float(frames.min()))
        gmax = max(gmax, float(frames.max()))
if not np.isfinite(gmin) or not np.isfinite(gmax) or gmax <= gmin:
    gmin, gmax = 0.0, wmax_exc
print(f"Fixed global color scale: vmin={gmin:.4g}  vmax={gmax:.4g}")

CMAP = "viridis"


def _save_heatmap(mat, title, out_path):
    h, w = mat.shape
    fig, ax = plt.subplots(figsize=(10, 10 * h / w + 1.2))
    im = ax.imshow(mat, aspect="equal", interpolation="nearest",
                   cmap=CMAP, vmin=gmin, vmax=gmax, origin="lower")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("hidden neuron (post)")
    ax.set_ylabel("input neuron (pre)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="weight")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ── render ────────────────────────────────────────────────────────────────────

total_png = 0
for n in sample_keys:
    frames = data[f"sample{n}_frames"]   # (F, N_IN, N_H)
    times  = data[f"sample{n}_times"]    # (F,)
    F = frames.shape[0]
    print(f"\nSample {n}: {F} frames")

    sample_dir = os.path.join(OUT_ROOT, f"sample_{n}")

    # 1) whole-matrix evolution
    full_dir = os.path.join(sample_dir, "full")
    os.makedirs(full_dir, exist_ok=True)
    for f in range(F):
        t_ms = float(times[f])
        _save_heatmap(
            frames[f],
            f"in->hid full  | epoch {EPOCH} sample {n} | frame {f:03d}  t={t_ms:.0f}ms",
            os.path.join(full_dir, f"frame_{f:03d}.png"),
        )
        total_png += 1
    print(f"  full/ : {F} PNGs")

    # 2) role-sliced evolution (7 separate series)
    for label, off in roles:
        rows = np.arange(off, N_IN, neurons_per_band)   # 96 rows
        role_dir = os.path.join(sample_dir, label)
        os.makedirs(role_dir, exist_ok=True)
        for f in range(F):
            t_ms = float(times[f])
            _save_heatmap(
                frames[f][rows, :],
                f"in->hid {label} ({len(rows)} ch x {N_H}) | epoch {EPOCH} "
                f"sample {n} | frame {f:03d}  t={t_ms:.0f}ms",
                os.path.join(role_dir, f"frame_{f:03d}.png"),
            )
            total_png += 1
        print(f"  {label}/ : {F} PNGs")

print(f"\nDone — {total_png} PNGs under {OUT_ROOT}")
