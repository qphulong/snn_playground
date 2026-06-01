"""
heatmap.py
----------
Loads fingerprints.npz and plots a pairwise log-absolute-distance heatmap.

Output: vizs/heatmap_log_absolute_distance.png

Usage:
    cd <repo_root>
    python training/tonotopic_plasticity_bound_fingerprint_analysis/heatmap.py
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
OUT_DIR    = os.path.join(SCRIPT_DIR, "vizs")

# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

if not os.path.exists(FP_PATH):
    sys.exit(f"ERROR: {FP_PATH} not found — run prepare_fingerprints.py first.")

data       = np.load(FP_PATH, allow_pickle=True)
fps        = data["fingerprints"].astype(np.float64)   # (N, 672, 672)
person_ids = data["person_ids"]
record_ids = data["record_ids"]
parts      = data["parts"]

N = len(fps)
print(f"Loaded {N} fingerprints  shape={fps.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# Split into part A (rows) and part B (columns)
# ─────────────────────────────────────────────────────────────────────────────

mask_A = parts == "A"
mask_B = parts == "B"

vecs_A = fps[mask_A].reshape(mask_A.sum(), -1)   # (n_A, 672*672)
vecs_B = fps[mask_B].reshape(mask_B.sum(), -1)   # (n_B, 672*672)

labels_A = [f"{p}\n{r}" for p, r in zip(person_ids[mask_A], record_ids[mask_A])]
labels_B = [f"{p}\n{r}" for p, r in zip(person_ids[mask_B], record_ids[mask_B])]

# Person group boundaries for visual separation
def _boundaries(pid_list):
    bounds = []
    for k in range(1, len(pid_list)):
        if pid_list[k] != pid_list[k - 1]:
            bounds.append(k - 0.5)
    return bounds

boundaries_A = _boundaries(list(person_ids[mask_A]))
boundaries_B = _boundaries(list(person_ids[mask_B]))

# ─────────────────────────────────────────────────────────────────────────────
# Similarity metrics registry
#
# Each function receives two 1-D float64 vectors (same length) and returns a
# scalar.  Add new metrics here — the rest of the script is automatic.
# ─────────────────────────────────────────────────────────────────────────────

def log_absolute_distance(x, y):
    return float(np.log(np.sum(np.abs(x - y)) + 1.0))


# (fn, is_distance)  — is_distance=True reverses the colormap and uses dynamic range
METRICS = {
    "log_absolute_distance": (log_absolute_distance, True),
}

# ─────────────────────────────────────────────────────────────────────────────
# Compute and plot one heatmap per metric
# ─────────────────────────────────────────────────────────────────────────────

def compute_matrix(metric_fn, vecs_row, vecs_col):
    """Compute the n_row × n_col similarity matrix for a given metric."""
    n_row, n_col = len(vecs_row), len(vecs_col)
    mat = np.zeros((n_row, n_col), dtype=np.float32)
    for i in range(n_row):
        for j in range(n_col):
            mat[i, j] = metric_fn(vecs_row[i], vecs_col[j])
    return mat


def plot_heatmap(mat, metric_name, labels_row, labels_col,
                 boundaries_row, boundaries_col, out_path, is_distance=False,
                 subtitle=None):
    n_row, n_col = mat.shape
    fig_w = max(8, n_col * 0.7 + 2)
    fig_h = max(6, n_row * 0.7 + 2)
    font_size = max(4, min(9, 80 / max(n_row, n_col)))

    cmap  = "RdYlGn_r" if is_distance else "RdYlGn"
    vmin  = mat.min() if is_distance else 0
    vmax  = mat.max() if is_distance else 1
    cbar_label = "Distance" if is_distance else "Similarity"

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mat, aspect="auto", interpolation="nearest",
                   cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.03, pad=0.02)

    ax.set_xticks(range(n_col))
    ax.set_xticklabels(labels_col, fontsize=font_size, rotation=45, ha="right")
    ax.set_yticks(range(n_row))
    ax.set_yticklabels(labels_row, fontsize=font_size)

    for b in boundaries_row:
        ax.axhline(b, color="white", linewidth=1.5)
    for b in boundaries_col:
        ax.axvline(b, color="white", linewidth=1.5)

    if n_row * n_col <= 1600:
        for i in range(n_row):
            for j in range(n_col):
                ax.text(j, i, f"{mat[i, j]:.2f}",
                        ha="center", va="center",
                        fontsize=max(3, font_size - 2), color="black")

    kind  = "Distance" if is_distance else "Similarity"
    title = f"{kind} ({metric_name}) — rows: Part A, cols: Part B"
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Part-B fingerprints", fontsize=10)
    ax.set_ylabel("Part-A fingerprints", fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.relpath(out_path, SCRIPT_DIR)}")


os.makedirs(OUT_DIR, exist_ok=True)

for name, (fn, is_distance) in METRICS.items():
    print(f"\nMetric: {name}")
    mat = compute_matrix(fn, vecs_A, vecs_B)
    print(f"  matrix  shape={mat.shape}  min={mat.min():.4f}  max={mat.max():.4f}  "
          f"mean={mat.mean():.4f}")
    out_path = os.path.join(OUT_DIR, f"heatmap_{name}.png")
    plot_heatmap(mat, name, labels_A, labels_B,
                 boundaries_A, boundaries_B, out_path, is_distance=is_distance)

# ── 5×5 aggregated heatmap ────────────────────────────────────────────────────
BLOCK = 3          # recordings per speaker
N_BLK = mat.shape[0] // BLOCK   # = 5 speakers

diag_mask = ~np.eye(BLOCK, dtype=bool)   # True for the 6 off-diagonal cells in a 3×3 block

mat5 = np.zeros((N_BLK, N_BLK), dtype=np.float32)
for bi in range(N_BLK):
    for bj in range(N_BLK):
        block = mat[bi*BLOCK:(bi+1)*BLOCK, bj*BLOCK:(bj+1)*BLOCK]
        mat5[bi, bj] = block[diag_mask].mean() if bi == bj else block.mean()

labels5_A = [person_ids[mask_A][bi * BLOCK] for bi in range(N_BLK)]
labels5_B = [person_ids[mask_B][bj * BLOCK] for bj in range(N_BLK)]

print(f"\nMetric: log_absolute_distance_5x5  shape={mat5.shape}  "
      f"min={mat5.min():.4f}  max={mat5.max():.4f}  mean={mat5.mean():.4f}")

diag_mean    = np.diag(mat5).mean()
offdiag_mean = mat5[~np.eye(N_BLK, dtype=bool)].mean()

out5 = os.path.join(OUT_DIR, "heatmap_log_absolute_distance_5x5.png")
plot_heatmap(mat5,
             "log_absolute_distance (5×5 block mean, same-recording excluded on diag)",
             labels5_A, labels5_B, [], [], out5, is_distance=True,
             subtitle=f"diag mean = {diag_mean:.4f}   off-diag mean = {offdiag_mean:.4f}")

print(f"\nDone — {len(METRICS) + 1} heatmap(s) saved to vizs/")
