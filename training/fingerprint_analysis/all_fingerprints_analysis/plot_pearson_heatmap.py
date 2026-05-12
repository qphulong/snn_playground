"""
plot_pearson_heatmap.py
=======================
Loads fingerprints.npz (produced by prepare_fingerprints.py) and plots a
Pearson correlation heatmap: rows = part-A fingerprints, columns = part-B
fingerprints.  Each cell is the Pearson r between two flattened 700×700
weight matrices.

Output: vizs/pearson_heatmap_AB.png

Usage:
    cd <repo_root>
    python training/fingerprint_analysis/plot_pearson_heatmap.py
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
# Compute Pearson r matrix  (n_A × n_B)
# ─────────────────────────────────────────────────────────────────────────────

flat_A = fps_A.reshape(n_A, -1)   # (15, 490000)
flat_B = fps_B.reshape(n_B, -1)   # (15, 490000)

r_matrix = np.zeros((n_A, n_B), dtype=np.float32)
for i in range(n_A):
    for j in range(n_B):
        r_matrix[i, j], _ = stats.pearsonr(flat_A[i], flat_B[j])

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
