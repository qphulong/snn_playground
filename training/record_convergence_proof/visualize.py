"""
convergence_proof/visualize.py
==============================
Loads recorder epoch data from run_3sample/ and run_1sample/ plus the averaged
fingerprints from fingerprints.npz, then produces:

  Raster plots (driven by record_and_visualize_config.yaml > visualize:)
    vizs/rasters/3sample_epoch{E}_input_sample{S}.png
    vizs/rasters/3sample_epoch{E}_hidden_sample{S}.png
    vizs/rasters/1sample_epoch{E}_input_sample{S}.png
    vizs/rasters/1sample_epoch{E}_hidden_sample{S}.png
    vizs/rasters/compare_hidden_sample0.png   — side-by-side last epoch, sample 0

  Fingerprint comparison (from fingerprints.npz)
    vizs/neuron_{N}_weights.png          — per-neuron incoming weights (yaml config)
    vizs/difference_heatmap.png
    vizs/per_neuron_l2.png
    vizs/difference_histogram.png
    vizs/per_neuron_pearson.png
    vizs/stats.txt

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
# Paths & config
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CFG_PATH     = os.path.join(SCRIPT_DIR, "record_and_visualize_config.yaml")
FP_PATH      = os.path.join(SCRIPT_DIR, "fingerprints.npz")
DIR_3        = os.path.join(SCRIPT_DIR, "run_3sample")
DIR_1        = os.path.join(SCRIPT_DIR, "run_1sample")
OUT_DIR      = os.path.join(SCRIPT_DIR, "vizs")
OUT_RASTERS  = os.path.join(OUT_DIR, "rasters")

os.makedirs(OUT_DIR,     exist_ok=True)
os.makedirs(OUT_RASTERS, exist_ok=True)

with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)

viz_cfg            = cfg.get("visualize", {})
visualize_samples  = viz_cfg.get("visualize_samples") or None   # None = all
group_viz          = viz_cfg.get("groups",           {}) or {}
weights_per_neuron = viz_cfg.get("weights_per_neuron", {}) or {}

saved = []


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


def _sample_indices(n_total):
    if visualize_samples:
        return [s for s in visualize_samples if s < n_total]
    return list(range(n_total))


def _load_last_epoch(run_dir):
    """Return (epoch_idx, data) for the last epoch npz in run_dir, or None."""
    files = sorted(glob_mod.glob(os.path.join(run_dir, "history_epoch_*.npz")))
    if not files:
        return None, None
    path = files[-1]
    idx  = int(os.path.basename(path).split("_")[-1].split(".")[0])
    return idx, np.load(path, allow_pickle=True)


def _load_all_epochs(run_dir):
    """Return list of (epoch_idx, data) for all epoch npz files in run_dir."""
    files = sorted(glob_mod.glob(os.path.join(run_dir, "history_epoch_*.npz")))
    result = []
    for path in files:
        idx = int(os.path.basename(path).split("_")[-1].split(".")[0])
        result.append((idx, np.load(path, allow_pickle=True)))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Raster plot helper  (same style as standard_template/visualize.py)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_spike_raster(data, keys, name, epoch_idx, sample_indices, color, out_subdir, run_label):
    if not (_has(keys, name, "raster_i") and _has(keys, name, "raster_t")):
        return
    raster_i  = _get(data, name, "raster_i")
    raster_t  = _get(data, name, "raster_t")
    n_total   = int(_get(data, name, "raster_n_samples"))
    n_neurons = int(_get(data, name, "raster_n_neurons"))

    for s in sample_indices:
        if s >= n_total:
            continue
        sp_i = raster_i[s]
        sp_t = raster_t[s]

        fig, ax = plt.subplots(figsize=(12, 5))
        if len(sp_t) > 0:
            ax.scatter(sp_t, sp_i, s=0.5, c=color, linewidths=0, rasterized=True)
        ax.set_title(
            f"Spike Raster — {name}  |  {run_label}, Epoch {epoch_idx}, Sample {s}"
            f"  ({len(sp_t):,} spikes)",
            fontsize=12, fontweight="bold",
        )
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron index")
        ax.set_xlim(left=0)
        ax.set_ylim(-0.5, n_neurons - 0.5)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()

        fname = f"{run_label}_epoch{epoch_idx:03d}_{_pfx(name)}_sample{s:03d}.png"
        _save(fig, os.path.join(out_subdir, fname))


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Raster plots from recorder epoch files
# ─────────────────────────────────────────────────────────────────────────────

RUNS = [
    ("3sample", DIR_3, "steelblue"),
    ("1sample", DIR_1, "darkorange"),
]

GROUP_COLORS = {"input": "steelblue", "hidden": "mediumseagreen"}

last_epoch_data = {}   # run_label -> (epoch_idx, data)

for run_label, run_dir, _ in RUNS:
    epochs = _load_all_epochs(run_dir)
    if not epochs:
        print(f"  [skip] No epoch files in {run_dir}")
        continue

    last_epoch_idx, last_data = epochs[-1]
    last_epoch_data[run_label] = (last_epoch_idx, last_data)

    for epoch_idx, data in epochs:
        keys     = set(data.files)
        n_total  = 0

        # find number of recorded samples from any group
        for gname in group_viz:
            if _has(keys, gname, "raster_n_samples"):
                n_total = int(_get(data, gname, "raster_n_samples"))
                break

        sample_idxs = _sample_indices(n_total)

        for gname, gcfg in group_viz.items():
            if not gcfg.get("spike_raster"):
                continue
            color = GROUP_COLORS.get(gname, "mediumpurple")
            _plot_spike_raster(data, keys, gname, epoch_idx,
                               sample_idxs, color, OUT_RASTERS, run_label)

# ── Side-by-side hidden raster comparison (last epoch, sample 0) ──────────────

has_both = ("3sample" in last_epoch_data and "1sample" in last_epoch_data)
if has_both and group_viz.get("hidden", {}).get("spike_raster"):
    e3, d3 = last_epoch_data["3sample"]
    e1, d1 = last_epoch_data["1sample"]
    keys3, keys1 = set(d3.files), set(d1.files)

    if (_has(keys3, "hidden", "raster_i") and _has(keys1, "hidden", "raster_i")):
        n3 = int(_get(d3, "hidden", "raster_n_samples"))
        n1 = int(_get(d1, "hidden", "raster_n_samples"))

        for s in [0]:   # compare sample 0 only
            if s >= n3 or s >= n1:
                continue

            sp_i3 = d3[_k("hidden", "raster_i")][s]
            sp_t3 = d3[_k("hidden", "raster_t")][s]
            sp_i1 = d1[_k("hidden", "raster_i")][s]
            sp_t1 = d1[_k("hidden", "raster_t")][s]
            n_neurons = int(_get(d3, "hidden", "raster_n_neurons"))

            fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, sharey=True)
            fig.suptitle(
                f"Hidden raster comparison — sample {s}\n"
                f"(3-sample: epoch {e3}, 1-sample: epoch {e1})",
                fontsize=12, fontweight="bold",
            )
            for ax, sp_i, sp_t, color, lbl, n_sp in [
                (axes[0], sp_i3, sp_t3, "mediumseagreen",
                 f"3-sample fingerprint  ({len(sp_t3):,} spikes)", n_neurons),
                (axes[1], sp_i1, sp_t1, "darkorange",
                 f"1-sample fingerprint  ({len(sp_t1):,} spikes)", n_neurons),
            ]:
                if len(sp_t) > 0:
                    ax.scatter(sp_t, sp_i, s=0.5, c=color, linewidths=0, rasterized=True)
                ax.set_title(lbl, fontsize=10)
                ax.set_ylabel("Hidden neuron index")
                ax.set_ylim(-0.5, n_sp - 0.5)
                ax.grid(True, alpha=0.2)
            axes[-1].set_xlabel("Time (ms)")
            plt.tight_layout()
            _save(fig, os.path.join(OUT_RASTERS, f"compare_hidden_sample{s:03d}.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Load fingerprints for comparison plots
# ─────────────────────────────────────────────────────────────────────────────

if not os.path.exists(FP_PATH):
    print(f"\nERROR: {FP_PATH} not found — skipping fingerprint comparison plots.")
    print(f"\n{len(saved)} PNG(s) saved to: {OUT_DIR}")
    sys.exit(0)

fp_data = np.load(FP_PATH, allow_pickle=True)
fp3     = fp_data["fingerprint_3"].astype(np.float64)
fp1     = fp_data["fingerprint_1"].astype(np.float64)
diff    = fp3 - fp1
N_IN, N_H = fp3.shape
print(f"\nLoaded fingerprints: shape={fp3.shape}")


def _save_fp(fig, name):
    _save(fig, os.path.join(OUT_DIR, name))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Per-neuron incoming weight plots
# ─────────────────────────────────────────────────────────────────────────────

for syn_name, neuron_ids in weights_per_neuron.items():
    for nid in (neuron_ids or []):
        nid = int(nid)
        if nid >= N_H:
            continue

        w3 = fp3[:, nid]
        w1 = fp1[:, nid]
        d  = w3 - w1
        x  = np.arange(N_IN)

        fig = plt.figure(figsize=(15, 9))
        gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)
        fig.suptitle(f"Hidden neuron {nid} — incoming weights comparison",
                     fontsize=13, fontweight="bold")

        ax0 = fig.add_subplot(gs[0, :])
        ax0.bar(x, w3, width=1.0, color="steelblue",  linewidth=0, label="3-sample fp")
        ax0.bar(x, w1, width=1.0, color="darkorange", linewidth=0, alpha=0.7, label="1-sample fp")
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
        ax1.set_title("Difference  (3-sample − 1-sample)", fontsize=11)
        ax1.set_xlabel("Input neuron index")
        ax1.set_ylabel("Δ weight")
        ax1.set_xlim(-0.5, N_IN - 0.5)
        ax1.grid(True, axis="y", alpha=0.3)

        ax2 = fig.add_subplot(gs[2, 0])
        ax2.hist(w3[w3 > 0], bins=40, color="steelblue",  alpha=0.7, label="3-sample", density=True)
        ax2.hist(w1[w1 > 0], bins=40, color="darkorange", alpha=0.7, label="1-sample", density=True)
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

        r, p = stats.pearsonr(w3, w1)
        l2   = float(np.linalg.norm(d))
        fig.text(0.5, 0.01,
                 f"neuron {nid} — L2 distance: {l2:.5f}   Pearson r: {r:.5f}  (p={p:.3g})",
                 ha="center", fontsize=10, color="dimgray")

        _save_fp(fig, f"neuron_{nid:04d}_weights.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Difference heatmap
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Fingerprint difference  fp3 − fp1", fontsize=13, fontweight="bold")

vabs = np.abs(diff).max()
im0  = axes[0].imshow(diff, aspect="auto", interpolation="nearest",
                      cmap="RdBu_r", vmin=-vabs, vmax=vabs, origin="lower")
axes[0].set_title("Signed difference matrix")
axes[0].set_xlabel("Hidden neuron index")
axes[0].set_ylabel("Input neuron index")
plt.colorbar(im0, ax=axes[0], label="fp3 − fp1")

im1  = axes[1].imshow(np.abs(diff), aspect="auto", interpolation="nearest",
                      cmap="hot_r", vmin=0, origin="lower")
axes[1].set_title("|difference| matrix")
axes[1].set_xlabel("Hidden neuron index")
axes[1].set_ylabel("Input neuron index")
plt.colorbar(im1, ax=axes[1], label="|fp3 − fp1|")

plt.tight_layout()
_save_fp(fig, "difference_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Per-neuron L2 distance
# ─────────────────────────────────────────────────────────────────────────────

l2_per_neuron = np.linalg.norm(diff, axis=0)

fig, ax = plt.subplots(figsize=(13, 4))
ax.bar(np.arange(N_H), l2_per_neuron, width=1.0, color="steelblue", linewidth=0)
ax.axhline(l2_per_neuron.mean(), color="crimson", lw=1.2, ls="--",
           label=f"mean = {l2_per_neuron.mean():.5f}")
ax.set_title("Per-hidden-neuron L2 distance  ||fp3[:,n] − fp1[:,n]||₂",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Hidden neuron index")
ax.set_ylabel("L2 distance")
ax.set_xlim(-0.5, N_H - 0.5)
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
_save_fp(fig, "per_neuron_l2.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Histogram of element-wise differences
# ─────────────────────────────────────────────────────────────────────────────

flat_diff = diff.flatten()

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Element-wise difference distribution  (fp3 − fp1)",
             fontsize=12, fontweight="bold")

axes[0].hist(flat_diff, bins=100, color="mediumpurple", edgecolor="none", density=True)
axes[0].axvline(0, color="black", lw=1.0, ls="--", label="0")
axes[0].axvline(flat_diff.mean(), color="crimson", lw=1.0, ls="-",
                label=f"mean={flat_diff.mean():.5f}")
axes[0].set_xlabel("fp3 − fp1")
axes[0].set_ylabel("Density")
axes[0].set_title("Full range")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

p1, p99 = np.percentile(flat_diff, [1, 99])
mask = (flat_diff >= p1) & (flat_diff <= p99)
axes[1].hist(flat_diff[mask], bins=80, color="mediumpurple", edgecolor="none", density=True)
axes[1].axvline(0, color="black", lw=1.0, ls="--")
axes[1].set_xlabel("fp3 − fp1")
axes[1].set_ylabel("Density")
axes[1].set_title("Central 98 % (p1–p99)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
_save_fp(fig, "difference_histogram.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Per-neuron Pearson correlation
# ─────────────────────────────────────────────────────────────────────────────

pearson_r = np.array([
    stats.pearsonr(fp3[:, n], fp1[:, n])[0] for n in range(N_H)
])

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Per-hidden-neuron Pearson correlation  r(fp3[:,n], fp1[:,n])",
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


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Statistical summary
# ─────────────────────────────────────────────────────────────────────────────

frob_abs = float(np.linalg.norm(diff, "fro"))
frob_fp3 = float(np.linalg.norm(fp3,  "fro"))
frob_rel = frob_abs / frob_fp3 if frob_fp3 > 0 else float("nan")

wilcox_stat, wilcox_p = stats.wilcoxon(flat_diff)

lines = [
    "=" * 60,
    "CONVERGENCE PROOF — STATISTICAL SUMMARY",
    "=" * 60,
    "",
    f"Fingerprint shape          : {fp3.shape}",
    f"3-sample training          : {fp_data['collected_3'].shape[0]} weight matrices averaged",
    f"1-sample training          : {fp_data['collected_1'].shape[0]} weight matrices averaged",
    "",
    "── Difference metrics ──────────────────────────────────────",
    f"Frobenius norm |fp3-fp1|   : {frob_abs:.6f}",
    f"Frobenius norm |fp3|       : {frob_fp3:.6f}",
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

stats_path = os.path.join(OUT_DIR, "stats.txt")
with open(stats_path, "w") as f:
    f.write(summary + "\n")
print(f"\n  Stats saved → {os.path.relpath(stats_path, SCRIPT_DIR)}")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{len(saved)} PNG(s) saved to: {OUT_DIR}")
