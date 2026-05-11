"""
fingerprint_analysis/visualize.py
===================================
Loads recorder epoch data and fingerprints, then produces:

  vizs/A/epoch_NNN/          — raster plots for run A
  vizs/B/epoch_NNN/          — raster plots for run B
  vizs/fingerprint_analysis/ — fingerprint comparison plots:
      neuron_{N}_weights.png
      difference_heatmap.png
      per_neuron_l2.png
      difference_histogram.png
      per_neuron_pearson.png
      stats.txt

Configuration is driven by visualize_A and visualize_B sections of
record_and_visualize_config.yaml.  weights_per_neuron from visualize_A
is used for the per-neuron fingerprint comparison plots.

Usage:
    python visualize.py
"""

import os
import sys
import glob as glob_mod
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_PATH   = os.path.join(SCRIPT_DIR, "record_and_visualize_config.yaml")
FP_PATH    = os.path.join(SCRIPT_DIR, "fingerprints.npz")
DIR_A      = os.path.join(SCRIPT_DIR, "run_A")
DIR_B      = os.path.join(SCRIPT_DIR, "run_B")
OUT_DIR    = os.path.join(SCRIPT_DIR, "vizs")
OUT_A      = os.path.join(OUT_DIR, "A")
OUT_B      = os.path.join(OUT_DIR, "B")
OUT_FP     = os.path.join(OUT_DIR, "fingerprint_analysis")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

with open(CFG_PATH) as f:
    _cfg = yaml.safe_load(f)

viz_cfg_A = _cfg.get("visualize_A", {})
viz_cfg_B = _cfg.get("visualize_B", {})

saved = []

GROUP_COLORS = {"input": "steelblue", "hidden": "mediumseagreen"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    print(f"  Saved: {os.path.relpath(path, SCRIPT_DIR)}")


def _pfx(name):
    return name.replace("->", "_to_")


def _k(name, field):
    return f"{_pfx(name)}__{field}"


def _has(keys, name, field):
    return _k(name, field) in keys


def _get(data, name, field):
    return data[_k(name, field)]


# ─────────────────────────────────────────────────────────────────────────────
# Raster plot helper
# ─────────────────────────────────────────────────────────────────────────────

def _plot_spike_raster(data, keys, name, epoch_idx, sample_idxs, color, out_base):
    if not (_has(keys, name, "raster_i") and _has(keys, name, "raster_t")):
        return
    raster_i  = _get(data, name, "raster_i")
    raster_t  = _get(data, name, "raster_t")
    n_total   = int(_get(data, name, "raster_n_samples"))
    n_neurons = int(_get(data, name, "raster_n_neurons"))

    for s in sample_idxs:
        if s >= n_total:
            continue
        sp_i = raster_i[s]
        sp_t = raster_t[s]

        fig, ax = plt.subplots(figsize=(12, 5))
        if len(sp_t) > 0:
            ax.scatter(sp_t, sp_i, s=0.5, c=color, linewidths=0, rasterized=True)
        ax.set_title(
            f"Spike Raster — {name}  |  Epoch {epoch_idx}, Sample {s}"
            f"  ({len(sp_t):,} spikes)",
            fontsize=12, fontweight="bold",
        )
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron index")
        ax.set_xlim(left=0)
        ax.set_ylim(-0.5, n_neurons - 0.5)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()

        fname = f"raster_{_pfx(name)}_sample{s:03d}.png"
        _save(fig, os.path.join(out_base, f"epoch_{epoch_idx:03d}", fname))


# ─────────────────────────────────────────────────────────────────────────────
# Per-run visualization
# ─────────────────────────────────────────────────────────────────────────────

def visualize_run(run_label, run_dir, viz_cfg, out_base):
    group_cfg         = viz_cfg.get("groups", {}) or {}
    visualize_samples = viz_cfg.get("visualize_samples") or None
    visualize_epoch   = viz_cfg.get("visualize_epoch") or []

    epoch_files = sorted(glob_mod.glob(os.path.join(run_dir, "history_epoch_*.npz")))
    if not epoch_files:
        print(f"  [skip] No epoch files in {run_dir}")
        return

    n_epochs      = len(epoch_files)
    epochs_to_viz = ([e for e in visualize_epoch if e < n_epochs]
                     if visualize_epoch else list(range(n_epochs)))

    for epoch_idx, npz_path in enumerate(epoch_files):
        if epoch_idx not in epochs_to_viz:
            continue
        print(f"\n  Epoch {epoch_idx} — {os.path.basename(npz_path)}")
        data = np.load(npz_path, allow_pickle=True)
        keys = set(data.files)

        n_samples = 0
        for gname in group_cfg:
            if _has(keys, gname, "raster_n_samples"):
                n_samples = int(_get(data, gname, "raster_n_samples"))
                break

        sample_idxs = ([s for s in visualize_samples if s < n_samples]
                       if visualize_samples else list(range(n_samples)))

        for name, gcfg in group_cfg.items():
            if gcfg.get("spike_raster"):
                color = GROUP_COLORS.get(name, "mediumpurple")
                _plot_spike_raster(data, keys, name, epoch_idx,
                                   sample_idxs, color, out_base)


# ─────────────────────────────────────────────────────────────────────────────
# Visualize both runs
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== Visualizing Run A ===")
visualize_run("A", DIR_A, viz_cfg_A, OUT_A)

print("\n=== Visualizing Run B ===")
visualize_run("B", DIR_B, viz_cfg_B, OUT_B)


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprint comparison
# ─────────────────────────────────────────────────────────────────────────────

if not os.path.exists(FP_PATH):
    print(f"\nERROR: {FP_PATH} not found — skipping fingerprint comparison plots.")
    print(f"\n{len(saved)} PNG(s) saved to: {OUT_DIR}")
    sys.exit(0)

os.makedirs(OUT_FP, exist_ok=True)

fp_data = np.load(FP_PATH, allow_pickle=True)
fp_A    = fp_data["fingerprint_A"].astype(np.float64)
fp_B    = fp_data["fingerprint_B"].astype(np.float64)
diff    = fp_A - fp_B
N_IN, N_H = fp_A.shape
print(f"\nLoaded fingerprints: shape={fp_A.shape}")


def _save_fp(fig, name):
    _save(fig, os.path.join(OUT_FP, name))


# ── Per-neuron incoming weight plots ──────────────────────────────────────────
# Neuron IDs come from visualize_A's weights_per_neuron config.

weights_per_neuron = viz_cfg_A.get("weights_per_neuron", {}) or {}

for syn_name, neuron_ids in weights_per_neuron.items():
    for nid in (neuron_ids or []):
        nid = int(nid)
        if nid >= N_H:
            continue

        w_A = fp_A[:, nid]
        w_B = fp_B[:, nid]
        d   = w_A - w_B
        x   = np.arange(N_IN)

        fig = plt.figure(figsize=(15, 9))
        gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)
        fig.suptitle(f"Hidden neuron {nid} — incoming weights comparison",
                     fontsize=13, fontweight="bold")

        ax0 = fig.add_subplot(gs[0, :])
        ax0.bar(x, w_A, width=1.0, color="steelblue",  linewidth=0, label="Run A fp")
        ax0.bar(x, w_B, width=1.0, color="darkorange", linewidth=0, alpha=0.7,
                label="Run B fp")
        ax0.set_title("Incoming weights (overlaid)", fontsize=11)
        ax0.set_xlabel("Input neuron index")
        ax0.set_ylabel("Weight")
        ax0.set_xlim(-0.5, N_IN - 0.5)
        ax0.legend(fontsize=9)
        ax0.grid(True, axis="y", alpha=0.3)

        colors_d = np.where(d >= 0, "steelblue", "crimson")
        ax1 = fig.add_subplot(gs[1, :])
        ax1.bar(x, d, width=1.0, color=colors_d, linewidth=0)
        ax1.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
        ax1.set_title("Difference  (A − B)", fontsize=11)
        ax1.set_xlabel("Input neuron index")
        ax1.set_ylabel("Δ weight")
        ax1.set_xlim(-0.5, N_IN - 0.5)
        ax1.grid(True, axis="y", alpha=0.3)

        ax2 = fig.add_subplot(gs[2, 0])
        ax2.hist(w_A[w_A > 0], bins=40, color="steelblue",  alpha=0.7,
                 label="Run A", density=True)
        ax2.hist(w_B[w_B > 0], bins=40, color="darkorange", alpha=0.7,
                 label="Run B", density=True)
        ax2.set_title("Weight distribution (non-zero)", fontsize=10)
        ax2.set_xlabel("Weight value")
        ax2.set_ylabel("Density")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[2, 1])
        ax3.hist(d, bins=40, color="mediumpurple", edgecolor="none", density=True)
        ax3.axvline(0, color="black", lw=0.8, ls="--")
        ax3.set_title("Difference distribution", fontsize=10)
        ax3.set_xlabel("Δ weight")
        ax3.set_ylabel("Density")
        ax3.grid(True, alpha=0.3)

        r, p = stats.pearsonr(w_A, w_B)
        l2   = float(np.linalg.norm(d))
        fig.text(0.5, 0.01,
                 f"neuron {nid} — L2: {l2:.5f}   Pearson r: {r:.5f}  (p={p:.3g})",
                 ha="center", fontsize=10, color="dimgray")
        _save_fp(fig, f"neuron_{nid:04d}_weights.png")


# ── Difference heatmap ────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Fingerprint difference  fp_A − fp_B", fontsize=13, fontweight="bold")

vabs = np.abs(diff).max()
im0  = axes[0].imshow(diff, aspect="auto", interpolation="nearest",
                      cmap="RdBu_r", vmin=-vabs, vmax=vabs, origin="lower")
axes[0].set_title("Signed difference matrix")
axes[0].set_xlabel("Hidden neuron index")
axes[0].set_ylabel("Input neuron index")
plt.colorbar(im0, ax=axes[0], label="fp_A − fp_B")

im1  = axes[1].imshow(np.abs(diff), aspect="auto", interpolation="nearest",
                      cmap="hot_r", vmin=0, origin="lower")
axes[1].set_title("|difference| matrix")
axes[1].set_xlabel("Hidden neuron index")
axes[1].set_ylabel("Input neuron index")
plt.colorbar(im1, ax=axes[1], label="|fp_A − fp_B|")

plt.tight_layout()
_save_fp(fig, "difference_heatmap.png")


# ── Per-neuron L2 distance ────────────────────────────────────────────────────

l2_per_neuron = np.linalg.norm(diff, axis=0)

fig, ax = plt.subplots(figsize=(13, 4))
ax.bar(np.arange(N_H), l2_per_neuron, width=1.0, color="steelblue", linewidth=0)
ax.axhline(l2_per_neuron.mean(), color="crimson", lw=1.2, ls="--",
           label=f"mean = {l2_per_neuron.mean():.5f}")
ax.set_title("Per-hidden-neuron L2 distance  ||fp_A[:,n] − fp_B[:,n]||₂",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Hidden neuron index")
ax.set_ylabel("L2 distance")
ax.set_xlim(-0.5, N_H - 0.5)
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
_save_fp(fig, "per_neuron_l2.png")


# ── Element-wise difference histogram ────────────────────────────────────────

flat_diff = diff.flatten()

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Element-wise difference distribution  (fp_A − fp_B)",
             fontsize=12, fontweight="bold")

axes[0].hist(flat_diff, bins=100, color="mediumpurple", edgecolor="none", density=True)
axes[0].axvline(0, color="black", lw=1.0, ls="--", label="0")
axes[0].axvline(flat_diff.mean(), color="crimson", lw=1.0, ls="-",
                label=f"mean={flat_diff.mean():.5f}")
axes[0].set_xlabel("fp_A − fp_B")
axes[0].set_ylabel("Density")
axes[0].set_title("Full range")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

p1, p99 = np.percentile(flat_diff, [1, 99])
mask = (flat_diff >= p1) & (flat_diff <= p99)
axes[1].hist(flat_diff[mask], bins=80, color="mediumpurple", edgecolor="none", density=True)
axes[1].axvline(0, color="black", lw=1.0, ls="--")
axes[1].set_xlabel("fp_A − fp_B")
axes[1].set_ylabel("Density")
axes[1].set_title("Central 98 % (p1–p99)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
_save_fp(fig, "difference_histogram.png")


# ── Per-neuron Pearson correlation ────────────────────────────────────────────

pearson_r = np.array([
    stats.pearsonr(fp_A[:, n], fp_B[:, n])[0] for n in range(N_H)
])

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Per-hidden-neuron Pearson correlation  r(fp_A[:,n], fp_B[:,n])",
             fontsize=12, fontweight="bold")

axes[0].hist(pearson_r, bins=60, color="mediumseagreen", edgecolor="none", density=True)
axes[0].axvline(pearson_r.mean(), color="crimson", lw=1.2, ls="--",
                label=f"mean r = {pearson_r.mean():.5f}")
axes[0].set_xlabel("Pearson r")
axes[0].set_ylabel("Density")
axes[0].set_title("Distribution across all hidden neurons")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].bar(np.arange(N_H), pearson_r, width=1.0, color="mediumseagreen", linewidth=0)
axes[1].axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.5)
axes[1].axhline(pearson_r.mean(), color="crimson", lw=1.2, ls="--",
                label=f"mean = {pearson_r.mean():.5f}")
axes[1].set_xlabel("Hidden neuron index")
axes[1].set_ylabel("Pearson r")
axes[1].set_title("Per-neuron correlation bar chart")
axes[1].set_xlim(-0.5, N_H - 0.5)
axes[1].legend(fontsize=8)
axes[1].grid(True, axis="y", alpha=0.3)

plt.tight_layout()
_save_fp(fig, "per_neuron_pearson.png")


# ── Statistical summary ───────────────────────────────────────────────────────

frob_abs = float(np.linalg.norm(diff, "fro"))
frob_fpA = float(np.linalg.norm(fp_A,  "fro"))
frob_rel = frob_abs / frob_fpA if frob_fpA > 0 else float("nan")

wilcox_stat, wilcox_p = stats.wilcoxon(flat_diff)

lines = [
    "=" * 60,
    "FINGERPRINT COMPARISON — STATISTICAL SUMMARY",
    "=" * 60,
    "",
    f"Fingerprint shape          : {fp_A.shape}",
    f"Run A training             : {fp_data['collected_A'].shape[0]} weight matrices averaged",
    f"Run B training             : {fp_data['collected_B'].shape[0]} weight matrices averaged",
    "",
    "── Difference metrics ──────────────────────────────────────",
    f"Frobenius norm |fp_A-fp_B| : {frob_abs:.6f}",
    f"Frobenius norm |fp_A|      : {frob_fpA:.6f}",
    f"Relative error             : {frob_rel:.4%}",
    "",
    "── Per-neuron L2 distance ──────────────────────────────────",
    f"Mean L2 per neuron         : {l2_per_neuron.mean():.6f}",
    f"Median L2 per neuron       : {np.median(l2_per_neuron):.6f}",
    f"Max L2 per neuron          : {l2_per_neuron.max():.6f}",
    "",
    "── Per-neuron Pearson correlation ──────────────────────────",
    f"Mean Pearson r             : {pearson_r.mean():.6f}",
    f"Median Pearson r           : {np.median(pearson_r):.6f}",
    f"Min Pearson r              : {pearson_r.min():.6f}",
    f"Neurons with r > 0.9       : {(pearson_r > 0.9).sum()}/{N_H}",
    f"Neurons with r > 0.95      : {(pearson_r > 0.95).sum()}/{N_H}",
    "",
    "── Wilcoxon signed-rank test (H0: differences symmetric about 0) ─",
    f"Statistic                  : {wilcox_stat:.4f}",
    f"p-value                    : {wilcox_p:.4g}",
    ("Result: NOT significant (p > 0.05) — differences indistinguishable from zero"
     if wilcox_p > 0.05 else
     f"Result: SIGNIFICANT (p = {wilcox_p:.4g}) — systematic bias detected"),
    "",
    "── Element-wise difference distribution ────────────────────",
    f"Mean                       : {flat_diff.mean():.6f}",
    f"Std                        : {flat_diff.std():.6f}",
    f"Max |diff|                 : {np.abs(flat_diff).max():.6f}",
    "=" * 60,
]

summary = "\n".join(lines)
print(f"\n{summary}")

stats_path = os.path.join(OUT_FP, "stats.txt")
with open(stats_path, "w") as f:
    f.write(summary + "\n")
print(f"\n  Stats saved → {os.path.relpath(stats_path, SCRIPT_DIR)}")

print(f"\n{len(saved)} PNG(s) saved to: {OUT_DIR}")
