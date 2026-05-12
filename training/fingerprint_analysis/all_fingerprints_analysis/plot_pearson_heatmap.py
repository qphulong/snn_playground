"""
plot_pearson_heatmap.py
=======================
Loads fingerprints.npz (produced by prepare_fingerprints.py) and plots a
Pearson correlation heatmap: rows = part-A fingerprints, columns = part-B
fingerprints.

Each cell is the mean per-neuron Pearson r: for each of the 700 hidden
neurons, Pearson r is computed between its incoming weight vectors in the two
fingerprints (fp_A[:, n] vs fp_B[:, n]), then averaged over all neurons.
This matches the metric reported by fingerprint_pair_analysis/visualize.py.

Output: vizs/pearson_heatmap_AB.png

Usage:
    cd <repo_root>
    python training/fingerprint_analysis/all_fingerprints_analysis/plot_pearson_heatmap.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FP_PATH    = os.path.join(SCRIPT_DIR, "fingerprints.npz")
OUT_PATH   = os.path.join(SCRIPT_DIR, "vizs", "pearson_heatmap_AB.png")

# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

if not os.path.exists(FP_PATH):
    sys.exit(f"ERROR: {FP_PATH} not found — run prepare_fingerprints.py first.")

data       = np.load(FP_PATH, allow_pickle=True)
fps        = data["fingerprints"].astype(np.float64)   # (30, 700, 700)
person_ids = data["person_ids"]
record_ids = data["record_ids"]
parts      = data["parts"]

mask_A = parts == "A"
mask_B = parts == "B"

fps_A      = fps[mask_A]           # (15, 700, 700)
fps_B      = fps[mask_B]           # (15, 700, 700)
labels_A   = [f"{p}\n{r}" for p, r in zip(person_ids[mask_A], record_ids[mask_A])]
labels_B   = [f"{p}\n{r}" for p, r in zip(person_ids[mask_B], record_ids[mask_B])]

n_A = len(fps_A)
n_B = len(fps_B)

# ─────────────────────────────────────────────────────────────────────────────
# Compute mean per-neuron Pearson r matrix  (n_A × n_B)
#
# For each pair (i, j): correlate the incoming weight vector of each hidden
# neuron n — fp_A[i, :, n] vs fp_B[j, :, n] — then average over N_H neurons.
# Vectorised per-i to avoid 15×15×700 individual scipy calls.
# ─────────────────────────────────────────────────────────────────────────────

r_matrix = np.zeros((n_A, n_B), dtype=np.float32)
for i in range(n_A):
    A   = fps_A[i]                             # (N_IN, N_H)
    A_c = A - A.mean(axis=0)                   # centre each column
    A_norm = np.sqrt((A_c ** 2).sum(axis=0))   # (N_H,)
    for j in range(n_B):
        B      = fps_B[j]
        B_c    = B - B.mean(axis=0)
        B_norm = np.sqrt((B_c ** 2).sum(axis=0))
        denom  = A_norm * B_norm
        r_per_neuron = np.where(denom > 0,
                                (A_c * B_c).sum(axis=0) / denom,
                                0.0)
        r_matrix[i, j] = float(r_per_neuron.mean())

print(f"Pearson r matrix  shape={r_matrix.shape}  "
      f"min={r_matrix.min():.4f}  max={r_matrix.max():.4f}  "
      f"mean={r_matrix.mean():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(r_matrix, aspect="auto", interpolation="nearest",
               cmap="RdYlGn", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, label="Pearson r")

ax.set_xticks(range(n_B))
ax.set_xticklabels(labels_B, fontsize=6, rotation=45, ha="right")
ax.set_yticks(range(n_A))
ax.set_yticklabels(labels_A, fontsize=6)

ax.set_xlabel("Part-B fingerprints", fontsize=11)
ax.set_ylabel("Part-A fingerprints", fontsize=11)
ax.set_title("Pearson correlation — Part A vs Part B fingerprints", fontsize=13, fontweight="bold")

# annotate each cell with the r value
for i in range(n_A):
    for j in range(n_B):
        ax.text(j, i, f"{r_matrix[i, j]:.2f}", ha="center", va="center",
                fontsize=5, color="black")

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Saved → {os.path.relpath(OUT_PATH)}")
