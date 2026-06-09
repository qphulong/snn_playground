"""
analyze_specialization.py
-------------------------
Do STDP hidden neurons learn a CONSISTENT and SPECIALIZED incoming-weight
pattern across two utterances of the SAME recording session?

Each hidden neuron j's incoming weights are a 672-vector = column j of the
in->hid matrix W (shape N_IN x N_H = 672 x 672, rows=input/pre, cols=hidden/post).
Two such matrices are stored per epoch (one per sample) in
`history_epoch_{NNN}.npz` under `in_to_hid__weight_per_sample`.

All columns share a strong common profile (role/channel structure + column-sum=2
normalisation), so raw similarity is ~1 and useless. We therefore work on
MEAN-CENTERED RESIDUALS: subtract each sample's population-mean column, leaving
only each neuron's deviation-from-population (its specialization). We then ask:
  * consistency  — is neuron j's residual the same in both samples?
  * specialized  — are different neurons' residuals distinct (identifiable)?

Control: the seed-42 initial matrix (regenerated exactly as in train.py), so we
can show consistency is LEARNED rather than leftover init structure.

Primary visualization: a PCA embedding of all residual vectors (both samples).
Expectation: the two points of a neuron-pair sit together (consistency) while the
population spreads across the space (specialization).

Usage:
    python analyze_specialization.py            # epoch 1 (default)
    python analyze_specialization.py 0          # epoch 0
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# ── constants mirrored from train.py (so the regenerated init is exact) ───────
N_IN          = 672
N_H           = 672
EXC_SUM_LIMIT = 2.0
WMIN_EXC      = 0.0
WMAX_EXC      = 1.0
SEED          = 42
CONSISTENCY_THRESH = 0.5   # "% of neurons with residual cosine above this"

# ── paths / epoch ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EPOCH      = int(sys.argv[1]) if len(sys.argv) > 1 else 1

npz_path = os.path.join(SCRIPT_DIR, f"history_epoch_{EPOCH:03d}.npz")
if not os.path.exists(npz_path):
    sys.exit(f"No history file for epoch {EPOCH}: {npz_path}")

OUT_DIR = os.path.join(SCRIPT_DIR, "specialization_vizs", f"epoch_{EPOCH}")
os.makedirs(OUT_DIR, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def l2_normalize_cols(M, eps=1e-12):
    """L2-normalize each column; zero columns stay zero."""
    norms = np.linalg.norm(M, axis=0, keepdims=True)
    return M / np.maximum(norms, eps)


def residualize(W):
    """Subtract the population-mean column -> per-neuron deviation from population."""
    return W - W.mean(axis=1, keepdims=True)


def regenerate_init():
    """Reproduce train.py's seed-42 init in->hid matrix exactly (lines 112-116)."""
    rng_state = np.random.get_state()
    try:
        np.random.seed(SEED)
        W = np.random.uniform(0, 1, (N_IN, N_H)).astype(float)
        cs = W.sum(axis=0)
        cs[cs == 0] = 1.0
        W = W / cs * EXC_SUM_LIMIT
        W = np.clip(W, WMIN_EXC, WMAX_EXC)
        return W
    finally:
        np.random.set_state(rng_state)


# ── load weights ──────────────────────────────────────────────────────────────
print(f"Loading {npz_path}")
data = np.load(npz_path, allow_pickle=True)
wps  = data["in_to_hid__weight_per_sample"]
if len(wps) < 2:
    sys.exit(f"Need >=2 samples, found {len(wps)}")
WA = np.asarray(wps[0], dtype=float)   # sample 0 final in->hid  (N_IN, N_H)
WB = np.asarray(wps[1], dtype=float)   # sample 1 final in->hid
Winit = regenerate_init()
print(f"WA {WA.shape}, WB {WB.shape}, Winit {Winit.shape}")

# ── residuals (columns = per-neuron residual vectors) ─────────────────────────
RA, RB, Rinit = residualize(WA), residualize(WB), residualize(Winit)

RA_n   = l2_normalize_cols(RA)
RB_n   = l2_normalize_cols(RB)
Rin_n  = l2_normalize_cols(Rinit)

# ── 1) per-neuron consistency (diagonal cosine) + init control ────────────────
cos_AB    = np.sum(RA_n * RB_n,  axis=0)   # (N_H,)
cos_Ainit = np.sum(RA_n * Rin_n, axis=0)
cos_Binit = np.sum(RB_n * Rin_n, axis=0)

# ── 2) full cross-sample cosine matrix S[i,j] = cos(RA_i, RB_j) ───────────────
S = RA_n.T @ RB_n                          # (N_H, N_H)
off = S[~np.eye(N_H, dtype=bool)]          # off-diagonal = "different neuron" null

# ── 3) identity matching ──────────────────────────────────────────────────────
top1_match  = np.mean(np.argmax(S, axis=1) == np.arange(N_H))
row_ind, col_ind = linear_sum_assignment(-S)          # maximize total similarity
hungarian_id = np.mean(col_ind[row_ind] == row_ind)
chance = 1.0 / N_H

# ── 4) pairs-together-vs-spread (residual space) ──────────────────────────────
# pair distance ||RA_j - RB_j||  vs  median distance from RA_j to OTHER B columns.
pair_dist = np.linalg.norm(RA - RB, axis=0)                       # (N_H,)
# distances RA_j to every RB_k:  ||RA_j||^2 + ||RB_k||^2 - 2 RA_j.RB_k
G   = RA.T @ RB                                                   # (N_H, N_H)
aa  = np.sum(RA * RA, axis=0)[:, None]
bb  = np.sum(RB * RB, axis=0)[None, :]
D   = np.sqrt(np.maximum(aa + bb - 2 * G, 0.0))                   # (N_H, N_H)
Doff = D.copy(); np.fill_diagonal(Doff, np.nan)
spread = np.nanmedian(Doff, axis=1)                              # typical other-neuron dist
ratio  = pair_dist / np.maximum(spread, 1e-12)                   # << 1 == pairs together

# ── report ────────────────────────────────────────────────────────────────────
report = {
    "epoch": EPOCH,
    "n_hidden": int(N_H),
    "consistency_cos_AB": {
        "mean": float(cos_AB.mean()), "median": float(np.median(cos_AB)),
        "std": float(cos_AB.std()),
        f"frac_above_{CONSISTENCY_THRESH}": float(np.mean(cos_AB > CONSISTENCY_THRESH)),
    },
    "control_cos_vs_init": {
        "mean_A_init": float(cos_Ainit.mean()), "mean_B_init": float(cos_Binit.mean()),
    },
    "offdiag_null_cos": {"mean": float(off.mean()), "std": float(off.std())},
    "identity_matching": {
        "top1_match_rate": float(top1_match),
        "hungarian_identity_rate": float(hungarian_id),
        "chance_rate": float(chance),
    },
    "pairs_vs_spread": {
        "mean_pair_distance": float(pair_dist.mean()),
        "mean_spread": float(np.nanmean(spread)),
        "mean_ratio": float(np.nanmean(ratio)),
        "median_ratio": float(np.nanmedian(ratio)),
    },
    "caveat": ("Sample 1 weights continue from sample 0 (persistent, sequential "
               "within epoch); the two are not independent, which inflates "
               "consistency. Stricter test: train each sample independently from "
               "the same init + add a different-speaker control."),
}

print("\n================= specialization / consistency =================")
print(f"  residual cosine A vs B : mean={cos_AB.mean():.3f} median={np.median(cos_AB):.3f} "
      f"({100*np.mean(cos_AB>CONSISTENCY_THRESH):.1f}% > {CONSISTENCY_THRESH})")
print(f"  control vs init        : A={cos_Ainit.mean():.3f}  B={cos_Binit.mean():.3f}  (expect ~0)")
print(f"  off-diagonal null      : mean={off.mean():.3f} std={off.std():.3f}")
print(f"  top-1 identity match   : {100*top1_match:.1f}%   (chance {100*chance:.2f}%)")
print(f"  Hungarian identity     : {100*hungarian_id:.1f}%")
print(f"  pair/spread ratio      : mean={np.nanmean(ratio):.3f} median={np.nanmedian(ratio):.3f} (<<1 = pairs together)")
print("================================================================\n")

with open(os.path.join(OUT_DIR, "specialization_report.json"), "w") as f:
    json.dump(report, f, indent=2)


# ── PCA embedding (primary viz) ───────────────────────────────────────────────
# Stack residual vectors: each row a (neuron, sample) point.
X = np.vstack([RA.T, RB.T])                 # (2*N_H, N_IN)
Xc = X - X.mean(axis=0, keepdims=True)
U, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
PC = U[:, :2] * sv[:2]                       # (2*N_H, 2) scores
evr = (sv**2) / np.sum(sv**2)
embA, embB = PC[:N_H], PC[N_H:]

fig, ax = plt.subplots(figsize=(11, 10))
# pair-connecting segments first (thin, behind points)
seg = np.stack([embA, embB], axis=1)         # (N_H, 2, 2)
from matplotlib.collections import LineCollection
ax.add_collection(LineCollection(seg, colors="0.6", linewidths=0.3, alpha=0.5, zorder=1))
ax.scatter(embA[:, 0], embA[:, 1], s=10, c="tab:blue",   label="sample 0", alpha=0.8, zorder=2)
ax.scatter(embB[:, 0], embB[:, 1], s=10, c="tab:orange", label="sample 1", alpha=0.8, zorder=2)
ax.set_xlabel(f"PC1 ({100*evr[0]:.1f}% var)")
ax.set_ylabel(f"PC2 ({100*evr[1]:.1f}% var)")
ax.set_title(f"Residual incoming-weight PCA — epoch {EPOCH}\n"
             f"each segment = one hidden neuron's (sample0 ↔ sample1) pair  "
             f"(pair/spread median={np.nanmedian(ratio):.2f})")
ax.legend(loc="best")
ax.set_aspect("equal", adjustable="datalim")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pca_pairs.png"), dpi=130)
plt.close(fig)

# ── consistency histogram (diagonal vs null vs init control) ──────────────────
fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(-1, 1, 81)
ax.hist(off,       bins=bins, density=True, alpha=0.45, color="0.6",
        label=f"off-diagonal null (mean {off.mean():.2f})")
ax.hist(cos_Ainit, bins=bins, density=True, alpha=0.45, color="tab:green",
        label=f"A vs init control (mean {cos_Ainit.mean():.2f})")
ax.hist(cos_AB,    bins=bins, density=True, alpha=0.65, color="tab:red",
        label=f"same-neuron A↔B (mean {cos_AB.mean():.2f})")
ax.axvline(cos_AB.mean(), color="tab:red", ls="--", lw=1)
ax.set_xlabel("residual cosine similarity")
ax.set_ylabel("density")
ax.set_title(f"Per-neuron cross-sample consistency — epoch {EPOCH}")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "consistency_hist.png"), dpi=130)
plt.close(fig)

# ── cross-sample similarity matrix ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(S, aspect="equal", origin="lower", cmap="RdBu_r",
               vmin=-np.abs(S).max(), vmax=np.abs(S).max(), interpolation="nearest")
ax.set_xlabel("hidden neuron (sample 1)")
ax.set_ylabel("hidden neuron (sample 0)")
ax.set_title(f"Cross-sample residual cosine S[i,j] — epoch {EPOCH}\n"
             f"bright diagonal = same-index neurons match  "
             f"(top-1 {100*top1_match:.0f}%, Hungarian {100*hungarian_id:.0f}%)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "similarity_matrix.png"), dpi=130)
plt.close(fig)

# ── pair-distance / spread ratio ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(ratio, bins=40, color="tab:purple", alpha=0.8)
ax.axvline(1.0, color="k", ls="--", lw=1, label="ratio = 1 (no consistency)")
ax.axvline(np.nanmedian(ratio), color="tab:red", ls="--", lw=1,
           label=f"median {np.nanmedian(ratio):.2f}")
ax.set_xlabel("pair distance / median other-neuron distance  (<<1 = pairs together)")
ax.set_ylabel("count")
ax.set_title(f"Pairs-together vs population spread — epoch {EPOCH}")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pair_distance_vs_spread.png"), dpi=130)
plt.close(fig)

print(f"Wrote 4 PNGs + specialization_report.json -> {OUT_DIR}")
