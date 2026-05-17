"""
plot_similarity_heatmap.py
==========================
Loads fingerprints.npz (produced by prepare_fingerprints.py) and plots a
weighted-similarity heatmap: rows = part-A fingerprints, columns = part-B
fingerprints.

All 30 fingerprints are first globally normalised to [0, 1] using the
dataset-wide min and max so absolute weight magnitudes share a common scale.

Similarity formula (x, y flattened to 1-D vectors of length 700*700):

    w_i     = max(|x_i|, |y_i|)
    sim(x,y) = sum(w_i * x_i * y_i)
               / ( sqrt(sum(w_i * x_i^2)) * sqrt(sum(w_i * y_i^2)) )

This is a weighted cosine similarity where positions dominated by large
values in either fingerprint are up-weighted.  If either weighted norm is
zero the score is set to 0.

Output: vizs/similarity_heatmap_AB.png

Usage:
    cd <repo_root>
    python training/fingerprint_analysis_to_be_deleted/all_fingerprints_analysis/plot_similarity_heatmap.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FP_PATH    = os.path.join(SCRIPT_DIR, "fingerprints.npz")
OUT_PATH   = os.path.join(SCRIPT_DIR, "vizs", "similarity_heatmap_AB.png")

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

# ─────────────────────────────────────────────────────────────────────────────
# Global normalise to [0, 1]
# ─────────────────────────────────────────────────────────────────────────────

g_min = fps.min()
g_max = fps.max()
fps   = (fps - g_min) / (g_max - g_min)

print(f"Global normalisation  min={g_min:.6f}  max={g_max:.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# Split A / B
# ─────────────────────────────────────────────────────────────────────────────

mask_A   = parts == "A"
mask_B   = parts == "B"

fps_A    = fps[mask_A]           # (15, 700, 700)
fps_B    = fps[mask_B]           # (15, 700, 700)
labels_A = [f"{p}\n{r}" for p, r in zip(person_ids[mask_A], record_ids[mask_A])]
labels_B = [f"{p}\n{r}" for p, r in zip(person_ids[mask_B], record_ids[mask_B])]

n_A = len(fps_A)
n_B = len(fps_B)

# ─────────────────────────────────────────────────────────────────────────────
# Compute similarity matrix  (n_A × n_B)
#
# Each fingerprint is flattened to a 490 000-element vector.
# w_i = max(|x_i|, |y_i|)
# sim = sum(w*x*y) / (sqrt(sum(w*x^2)) * sqrt(sum(w*y^2)))
# ─────────────────────────────────────────────────────────────────────────────

sim_matrix = np.zeros((n_A, n_B), dtype=np.float32)

for i in range(n_A):
    x = fps_A[i].ravel()          # (490_000,)
    for j in range(n_B):
        y     = fps_B[j].ravel()
        w     = np.maximum(np.abs(x), np.abs(y))
        norm_x = np.sqrt((w * x * x).sum())
        norm_y = np.sqrt((w * y * y).sum())
        denom  = norm_x * norm_y
        if denom == 0.0:
            sim_matrix[i, j] = 0.0
        else:
            sim_matrix[i, j] = float((w * x * y).sum() / denom)

print(f"Similarity matrix  shape={sim_matrix.shape}  "
      f"min={sim_matrix.min():.4f}  max={sim_matrix.max():.4f}  "
      f"mean={sim_matrix.mean():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(sim_matrix, aspect="auto", interpolation="nearest",
               cmap="RdYlGn", vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label="Similarity")

ax.set_xticks(range(n_B))
ax.set_xticklabels(labels_B, fontsize=6, rotation=45, ha="right")
ax.set_yticks(range(n_A))
ax.set_yticklabels(labels_A, fontsize=6)

ax.set_xlabel("Part-B fingerprints", fontsize=11)
ax.set_ylabel("Part-A fingerprints", fontsize=11)
ax.set_title("Weighted similarity — Part A vs Part B fingerprints",
             fontsize=13, fontweight="bold")

for i in range(n_A):
    for j in range(n_B):
        ax.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center",
                fontsize=5, color="black")

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Saved → {os.path.relpath(OUT_PATH)}")
