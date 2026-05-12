"""
plot_l1_heatmap.py
==================
Loads fingerprints.npz (produced by prepare_fingerprints.py) and plots a
similarity heatmap: rows = part-A fingerprints, columns = part-B fingerprints.

Each cell is 1 / (1 + d) where d is the mean per-neuron L1 distance:
  d(i, j) = mean over n of  ||fp_A[i, :, n] - fp_B[j, :, n]||_1

The 1/(1+d) transform maps distance to similarity in (0, 1]:
  d = 0  →  similarity = 1.0  (identical)
  d → ∞  →  similarity → 0.0

Output:
  vizs/l1_similarity_heatmap_AB.png      — full 15×15 matrix
  vizs/l1_similarity_heatmap_block.png   — 5×5 speaker-level block means

Usage:
    cd <repo_root>
    python training/fingerprint_analysis/all_fingerprints_analysis/plot_l1_heatmap.py
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

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
FP_PATH        = os.path.join(SCRIPT_DIR, "fingerprints.npz")
OUT_PATH       = os.path.join(SCRIPT_DIR, "vizs", "l1_similarity_heatmap_AB.png")
OUT_PATH_BLOCK = os.path.join(SCRIPT_DIR, "vizs", "l1_similarity_heatmap_block.png")

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

fps_A    = fps[mask_A]   # (15, 700, 700)
fps_B    = fps[mask_B]   # (15, 700, 700)
labels_A = [f"{p}\n{r}" for p, r in zip(person_ids[mask_A], record_ids[mask_A])]
labels_B = [f"{p}\n{r}" for p, r in zip(person_ids[mask_B], record_ids[mask_B])]

n_A = len(fps_A)
n_B = len(fps_B)

# ─────────────────────────────────────────────────────────────────────────────
# Compute similarity matrix  (n_A × n_B)
#
# d(i, j) = mean over neurons n of  ||fp_A[i, :, n] - fp_B[j, :, n]||_1
# s(i, j) = 1 / (1 + d(i, j))
# ─────────────────────────────────────────────────────────────────────────────

sim_matrix = np.zeros((n_A, n_B), dtype=np.float32)
d_matrix   = np.zeros((n_A, n_B), dtype=np.float32)

for i in range(n_A):
    A = fps_A[i]   # (N_IN, N_H)
    for j in range(n_B):
        B = fps_B[j]
        l1_per_neuron  = np.abs(A - B).sum(axis=0)   # (N_H,)
        d = float(l1_per_neuron.mean())
        d_matrix[i, j]   = d
        sim_matrix[i, j] = 1.0 / (1.0 + d)

diag_mask     = np.eye(n_A, n_B, dtype=bool)
diag_mean     = sim_matrix[diag_mask].mean()
offdiag_mean  = sim_matrix[~diag_mask].mean()

print(f"L1 distance   min={d_matrix.min():.4f}  max={d_matrix.max():.4f}  "
      f"mean={d_matrix.mean():.4f}")
print(f"Similarity    min={sim_matrix.min():.4f}  max={sim_matrix.max():.4f}  "
      f"mean={sim_matrix.mean():.4f}")
print(f"Diagonal      mean={diag_mean:.4f}")
print(f"Off-diagonal  mean={offdiag_mean:.4f}")
print(f"Gap (diag - off-diag) = {diag_mean - offdiag_mean:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(sim_matrix, aspect="auto", interpolation="nearest",
               cmap="viridis", vmin=sim_matrix.min(), vmax=sim_matrix.max())
plt.colorbar(im, ax=ax, label="Similarity  1 / (1 + mean L1)")

ax.set_xticks(range(n_B))
ax.set_xticklabels(labels_B, fontsize=6, rotation=45, ha="right")
ax.set_yticks(range(n_A))
ax.set_yticklabels(labels_A, fontsize=6)

ax.set_xlabel("Part-B fingerprints", fontsize=11)
ax.set_ylabel("Part-A fingerprints", fontsize=11)
ax.set_title(
    f"L1 similarity  1/(1+d)  — Part A vs Part B fingerprints\n"
    f"diagonal mean={diag_mean:.4f}   off-diagonal mean={offdiag_mean:.4f}   "
    f"gap={diag_mean - offdiag_mean:+.4f}",
    fontsize=11, fontweight="bold",
)

for i in range(n_A):
    for j in range(n_B):
        ax.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center",
                fontsize=5, color="black")

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Saved → {os.path.relpath(OUT_PATH)}")

# ─────────────────────────────────────────────────────────────────────────────
# Block-mean heatmap  (5×5 speaker level)
#
# The 15×15 matrix is ordered as 5 speakers × 3 recordings.
# Each 3×3 block [i*3:(i+1)*3, j*3:(j+1)*3] corresponds to speaker pair (i,j).
# Averaging each block collapses recording variance → speaker-level similarity.
# ─────────────────────────────────────────────────────────────────────────────

BLOCK = 3   # recordings per speaker
N_SPK = n_A // BLOCK  # = 5

speaker_labels_A = sorted(set(person_ids[mask_A]))
speaker_labels_B = sorted(set(person_ids[mask_B]))

block_matrix = np.zeros((N_SPK, N_SPK), dtype=np.float32)
for si in range(N_SPK):
    for sj in range(N_SPK):
        block = sim_matrix[si*BLOCK:(si+1)*BLOCK, sj*BLOCK:(sj+1)*BLOCK]
        block_matrix[si, sj] = block.mean()

block_diag_mask    = np.eye(N_SPK, dtype=bool)
block_diag_mean    = block_matrix[block_diag_mask].mean()
block_offdiag_mean = block_matrix[~block_diag_mask].mean()

print(f"\nBlock matrix  min={block_matrix.min():.4f}  max={block_matrix.max():.4f}  "
      f"mean={block_matrix.mean():.4f}")
print(f"Block diagonal      mean={block_diag_mean:.4f}")
print(f"Block off-diagonal  mean={block_offdiag_mean:.4f}")
print(f"Block gap (diag - off-diag) = {block_diag_mean - block_offdiag_mean:.4f}")

fig2, ax2 = plt.subplots(figsize=(6, 5))

im2 = ax2.imshow(block_matrix, aspect="auto", interpolation="nearest",
                 cmap="viridis", vmin=block_matrix.min(), vmax=block_matrix.max())
plt.colorbar(im2, ax=ax2, label="Mean similarity  1 / (1 + mean L1)")

ax2.set_xticks(range(N_SPK))
ax2.set_xticklabels(speaker_labels_B, fontsize=8, rotation=45, ha="right")
ax2.set_yticks(range(N_SPK))
ax2.set_yticklabels(speaker_labels_A, fontsize=8)

ax2.set_xlabel("Speaker (Part-B)", fontsize=11)
ax2.set_ylabel("Speaker (Part-A)", fontsize=11)
ax2.set_title(
    f"Speaker-level L1 similarity  (mean of 3×3 blocks)\n"
    f"diagonal mean={block_diag_mean:.4f}   off-diagonal mean={block_offdiag_mean:.4f}   "
    f"gap={block_diag_mean - block_offdiag_mean:+.4f}",
    fontsize=10, fontweight="bold",
)

for si in range(N_SPK):
    for sj in range(N_SPK):
        ax2.text(sj, si, f"{block_matrix[si, sj]:.4f}", ha="center", va="center",
                 fontsize=7, color="white" if block_matrix[si, sj] < block_matrix.mean() else "black")

plt.tight_layout()
fig2.savefig(OUT_PATH_BLOCK, dpi=150, bbox_inches="tight")
plt.close(fig2)

print(f"Saved → {os.path.relpath(OUT_PATH_BLOCK)}")
