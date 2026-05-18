"""
heatmap.py
----------
Loads fingerprints.npz and plots a pairwise similarity heatmap for each
registered metric.  Each metric produces one PNG in vizs/.

To add a new metric: add a function to METRICS that takes two 1-D float64
vectors and returns a scalar similarity score.

Output: vizs/heatmap_<metric_name>.png

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
fps        = data["fingerprints"].astype(np.float64)   # (N, 700, 700)
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

vecs_A = fps[mask_A].reshape(mask_A.sum(), -1)   # (n_A, 700*700)
vecs_B = fps[mask_B].reshape(mask_B.sum(), -1)   # (n_B, 700*700)

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

def cosine(x, y):
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if nx == 0 or ny == 0:
        return 0.0
    return float(np.dot(x, y) / (nx * ny))


def dot_product(x, y):
    return float(np.dot(x, y))


def weighted_cosine(x, y):
    w = np.maximum(np.abs(x), np.abs(y)) ** 2
    norm_x = np.sqrt((w * x * x).sum())
    norm_y = np.sqrt((w * y * y).sum())
    denom  = norm_x * norm_y
    if denom == 0.0:
        return 0.0
    return float((w * x * y).sum() / denom)


def squared_error(x, y):
    return float(np.sum((x - y) ** 2))


def absolute_error(x, y):
    return float(np.sum(np.abs(x - y)))


def rooted_error(x, y):
    return float(np.sqrt(np.sum((x - y) ** 2)))


# (fn, is_distance)  — is_distance=True reverses the colormap and uses dynamic range
METRICS = {
    "cosine":          (cosine,          False),
    "dot_product":     (dot_product,     False),
    "weighted_cosine": (weighted_cosine, False),
    "squared_error":   (squared_error,   True),
    "absolute_error":  (absolute_error,  True),
    "rooted_error":    (rooted_error,    True),
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
                 boundaries_row, boundaries_col, out_path, is_distance=False):
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
    ax.set_title(f"{kind} ({metric_name}) — rows: Part A, cols: Part B",
                 fontsize=13, fontweight="bold", pad=10)
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

print(f"\nDone — {len(METRICS)} heatmap(s) saved to vizs/")
