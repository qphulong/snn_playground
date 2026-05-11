"""
fingerprint_analysis/visualize.py
===================================
Loads recorder epoch data and fingerprints, then produces:

  vizs/A/epoch_NNN/          — all enabled plots for run A
  vizs/B/epoch_NNN/          — all enabled plots for run B
  vizs/fingerprint_analysis/ — fingerprint comparison plots:
      neuron_{N}_weights.png
      difference_heatmap.png
      per_neuron_l2.png
      difference_histogram.png
      per_neuron_pearson.png
      stats.txt

Per-run plots mirror training/standard_template/visualize.py exactly.
Configuration is driven by visualize_A and visualize_B sections of
record_and_visualize_config.yaml.

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

GROUP_COLORS = {"input": "steelblue", "hidden": "mediumseagreen", "output": "darkorange"}


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


def _window_mask(t, t_start, t_end):
    mask = np.ones(len(t), dtype=bool)
    if t_start >= 0:
        mask &= t >= t_start
    if t_end >= 0:
        mask &= t <= t_end
    return mask


def _sample_indices(n_total, visualize_samples):
    if visualize_samples:
        return [s for s in visualize_samples if s < n_total]
    return list(range(n_total))


def _epochs_to_visualize(n_total, visualize_epoch):
    if visualize_epoch:
        return [e for e in visualize_epoch if e < n_total]
    return list(range(n_total))


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_weight_matrix(W, title, path):
    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1], figure=fig)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax_heat = fig.add_subplot(gs[0])
    im = ax_heat.imshow(
        W, aspect="auto", interpolation="nearest", cmap="viridis",
        vmin=0, vmax=W.max() if W.max() > 0 else 1, origin="lower",
    )
    ax_heat.set_xlabel("Post-synaptic neuron index")
    ax_heat.set_ylabel("Pre-synaptic neuron index")
    plt.colorbar(im, ax=ax_heat, label="Weight")

    ax_hist = fig.add_subplot(gs[1])
    w_flat = W.flatten()
    ax_hist.hist(w_flat[w_flat > 0], bins=60, color="steelblue", edgecolor="none", density=True)
    ax_hist.set_xlabel("Weight value")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Distribution\n(non-zero)")
    ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, path)


def _plot_firing_rate(rates, title, path, color="darkorange"):
    n = len(rates)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(np.arange(n), rates, width=1.0, color=color, linewidth=0)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Neuron index")
    ax.set_ylabel("Mean rate (Hz)")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Per-group plot routines
# ─────────────────────────────────────────────────────────────────────────────

def _plot_spike_raster(data, keys, name, epoch_idx, sample_idxs, color, epoch_out):
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
        _save(fig, os.path.join(epoch_out, f"raster_{_pfx(name)}_sample{s:03d}.png"))


def _plot_spike_counts(data, keys, name, epoch_idx, sample_idxs, color, epoch_out):
    if not _has(keys, name, "raster_i"):
        return
    raster_i  = _get(data, name, "raster_i")
    n_total   = int(_get(data, name, "raster_n_samples"))
    n_neurons = int(_get(data, name, "raster_n_neurons"))

    for s in sample_idxs:
        if s >= n_total:
            continue
        sp_i = raster_i[s]
        counts = (np.bincount(sp_i.astype(np.int32), minlength=n_neurons)
                  if len(sp_i) > 0 else np.zeros(n_neurons, dtype=np.int32))
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(np.arange(n_neurons), counts, width=1.0, color=color, linewidth=0)
        ax.set_title(
            f"Spike Count — {name}  |  Epoch {epoch_idx}, Sample {s}",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Neuron index")
        ax.set_ylabel("Spike count")
        ax.set_xlim(-0.5, n_neurons - 0.5)
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        _save(fig, os.path.join(epoch_out, f"spike_count_{_pfx(name)}_sample{s:03d}.png"))


def _plot_mean_firing_rate(data, keys, name, epoch_idx, sample_idxs, color, epoch_out):
    pfx = _pfx(name)

    if _has(keys, name, "mfr"):
        _plot_firing_rate(
            _get(data, name, "mfr"),
            f"Mean Firing Rate — {name}  |  Epoch {epoch_idx}, all samples",
            os.path.join(epoch_out, f"mean_firing_rate_{pfx}_all.png"),
            color=color,
        )

    if (_has(keys, name, "mfr_sample_counts") and _has(keys, name, "mfr_sample_dur_s")):
        sample_counts = _get(data, name, "mfr_sample_counts")
        sample_durs   = _get(data, name, "mfr_sample_dur_s")
        n_total       = len(sample_durs)

        for s in sample_idxs:
            if s >= n_total:
                continue
            dur   = float(sample_durs[s])
            rates = (sample_counts[s] / dur if dur > 0
                     else np.zeros_like(sample_counts[s], dtype=np.float32))
            _plot_firing_rate(
                rates.astype(np.float32),
                f"Mean Firing Rate — {name}  |  Epoch {epoch_idx}, Sample {s}",
                os.path.join(epoch_out, f"mean_firing_rate_{pfx}_sample{s:03d}.png"),
                color=color,
            )


def _plot_membrane_potential(data, keys, name, epoch_idx, sample_idxs, epoch_out):
    v_key = _k(name, "vmon_v_all")
    t_key = _k(name, "vmon_t_all")
    if v_key not in keys or t_key not in keys:
        return

    neurons = _get(data, name, "vmon_indices")
    t_all   = _get(data, name, "vmon_t_all")
    windows = _get(data, name, "vmon_windows")
    n_total = len(t_all)
    pfx     = _pfx(name)

    var_keys = {}
    for k in keys:
        tag = f"{_pfx(name)}__vmon_"
        if k.startswith(tag) and k.endswith("_all"):
            var = k[len(tag):-4]
            if var != "t":
                var_keys[var] = k

    if not var_keys:
        return

    for s in sample_idxs:
        if s >= n_total:
            continue
        t = t_all[s]

        for k_idx, nid in enumerate(neurons):
            t_start = float(windows[k_idx, 1]) if windows[k_idx, 1] >= 0 else -1.0
            t_end   = float(windows[k_idx, 2]) if windows[k_idx, 2] >= 0 else -1.0
            mask    = _window_mask(t, t_start, t_end)
            t_w     = t[mask]

            n_vars = len(var_keys)
            fig, axes = plt.subplots(n_vars, 1, figsize=(11, 3 * n_vars), sharex=True,
                                     squeeze=False)
            win_str = f"  [{t_start:.0f}–{t_end:.0f} ms]" if t_start >= 0 else ""
            fig.suptitle(
                f"State Variables — {name}, Neuron {nid}  |  "
                f"Epoch {epoch_idx}, Sample {s}{win_str}",
                fontsize=11, fontweight="bold",
            )

            colors_list = ["steelblue", "crimson", "darkorange", "mediumseagreen",
                           "mediumpurple", "saddlebrown"]

            for ax_i, (var, full_key) in enumerate(sorted(var_keys.items())):
                ax  = axes[ax_i, 0]
                arr = data[full_key][s]
                y_w = arr[k_idx][mask]
                ax.plot(t_w, y_w, lw=0.8, color=colors_list[ax_i % len(colors_list)], label=var)
                ax.set_ylabel(var)
                ax.legend(fontsize=8, loc="upper right")
                ax.grid(True, alpha=0.3)

            axes[-1, 0].set_xlabel("Time (ms)")
            plt.tight_layout()
            win_suffix = (f"_window{t_start:.0f}_{t_end:.0f}ms" if t_start >= 0 else "")
            _save(fig, os.path.join(epoch_out,
                                    f"vmon_{pfx}_sample{s:03d}_neuron{nid:04d}{win_suffix}.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Per-synapse plot routines
# ─────────────────────────────────────────────────────────────────────────────

def _plot_weight_evolution(data, keys, name, epoch_idx, sample_idxs, epoch_out):
    if not _has(keys, name, "we_pairs"):
        return
    pairs  = _get(data, name, "we_pairs")
    values = _get(data, name, "we_values")
    pfx    = _pfx(name)

    x_axis = np.arange(values.shape[1])
    for k, (pi, pj) in enumerate(pairs):
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(x_axis, values[k], lw=1.5, color=f"C{k % 10}")
        ax.set_title(
            f"Weight Evolution (epoch {epoch_idx}, all samples) — "
            f"{name}  pre[{pi}] → post[{pj}]",
            fontsize=12, fontweight="bold",
        )
        ax.set_xlabel("Snapshot index")
        ax.set_ylabel("Weight")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save(fig, os.path.join(epoch_out,
                                f"weight_evolution_{pfx}_all_pre{pi:04d}_post{pj:04d}.png"))

    if (_has(keys, name, "we_sample_values") and _has(keys, name, "we_sample_times")):
        sample_values = _get(data, name, "we_sample_values")
        sample_times  = _get(data, name, "we_sample_times")
        n_total       = len(sample_times)

        for s in sample_idxs:
            if s >= n_total:
                continue
            sv     = sample_values[s]
            st     = sample_times[s]
            if len(st) == 0:
                continue
            st_rel = st - st[0]

            for k, (pi, pj) in enumerate(pairs):
                fig, ax = plt.subplots(figsize=(11, 3))
                ax.plot(st_rel, sv[k], lw=1.5, color=f"C{k % 10}")
                ax.set_title(
                    f"Weight Evolution (epoch {epoch_idx}, sample {s}) — "
                    f"{name}  pre[{pi}] → post[{pj}]",
                    fontsize=12, fontweight="bold",
                )
                ax.set_xlabel("Time within sample (ms)")
                ax.set_ylabel("Weight")
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                _save(fig, os.path.join(epoch_out,
                                        f"weight_evolution_{pfx}_sample{s:03d}_"
                                        f"pre{pi:04d}_post{pj:04d}.png"))


def _plot_weight_delta(data, keys, name, epoch_idx, track_weight_delta, epoch_out):
    if not track_weight_delta:
        return
    if not _has(keys, name, "we_pairs"):
        return
    pairs  = _get(data, name, "we_pairs")
    values = _get(data, name, "we_values")
    pfx    = _pfx(name)

    x_axis = np.arange(values.shape[1])
    for k, (pi, pj) in enumerate(pairs):
        w_vals = values[k]
        deltas = np.diff(w_vals, prepend=w_vals[0])
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(x_axis, deltas, lw=1.5, color=f"C{k % 10}", marker="o", markersize=3)
        ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)
        ax.set_title(
            f"Weight Delta (epoch {epoch_idx}) — {name}  pre[{pi}] → post[{pj}]",
            fontsize=12, fontweight="bold",
        )
        ax.set_xlabel("Snapshot index")
        ax.set_ylabel("Δw")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save(fig, os.path.join(epoch_out,
                                f"weight_delta_{pfx}_all_pre{pi:04d}_post{pj:04d}.png"))


def _plot_synapse_weight_matrix(data, keys, name, epoch_idx, sample_idxs, epoch_out):
    pfx = _pfx(name)

    if _has(keys, name, "final_weights"):
        W = _get(data, name, "final_weights")
        _plot_weight_matrix(
            W,
            f"Final Weight Matrix — {name}  |  Epoch {epoch_idx}",
            os.path.join(epoch_out, f"final_weight_matrix_{pfx}.png"),
        )

    if (_has(keys, name, "weight_per_sample") and _has(keys, name, "weight_n_samples")):
        wm_per_sample = _get(data, name, "weight_per_sample")
        n_total       = int(_get(data, name, "weight_n_samples"))

        for s in sample_idxs:
            if s >= n_total:
                continue
            W = wm_per_sample[s]
            _plot_weight_matrix(
                W,
                f"Weight Matrix — {name}  |  Epoch {epoch_idx}, after sample {s}",
                os.path.join(epoch_out, f"weight_matrix_{pfx}_sample{s:03d}.png"),
            )


def _plot_weights_per_neuron(data, keys, name, epoch_idx, sample_idxs,
                              weights_per_neuron, epoch_out):
    neuron_ids = weights_per_neuron.get(name, [])
    if not neuron_ids:
        return
    if not (_has(keys, name, "weight_per_sample") and _has(keys, name, "weight_n_samples")):
        return
    wm_per_sample = _get(data, name, "weight_per_sample")
    n_total       = int(_get(data, name, "weight_n_samples"))

    for s in sample_idxs:
        if s >= n_total:
            continue
        W = wm_per_sample[s]

        for nid in neuron_ids:
            nid = int(nid)
            if nid >= W.shape[1]:
                continue
            weights = W[:, nid]

            fig = plt.figure(figsize=(14, 5))
            gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], figure=fig)
            fig.suptitle(
                f"Incoming Weights — {name}, post-neuron {nid}  |  "
                f"Epoch {epoch_idx}, Sample {s}",
                fontsize=13, fontweight="bold",
            )

            ax_bar = fig.add_subplot(gs[0])
            ax_bar.bar(np.arange(len(weights)), weights, width=0.8,
                       color="steelblue", linewidth=0)
            ax_bar.set_xlabel("Input neuron index")
            ax_bar.set_ylabel("Weight magnitude")
            ax_bar.set_xlim(-0.5, len(weights) - 0.5)
            ax_bar.set_ylim(bottom=0)
            ax_bar.grid(True, axis="y", alpha=0.3)

            ax_hist = fig.add_subplot(gs[1])
            w_nonzero = weights[weights > 0]
            if len(w_nonzero) > 0:
                ax_hist.hist(w_nonzero, bins=40, color="steelblue",
                             edgecolor="none", density=True)
            ax_hist.set_xlabel("Weight value")
            ax_hist.set_ylabel("Density")
            ax_hist.set_title("Distribution\n(non-zero)")
            ax_hist.grid(True, alpha=0.3)

            plt.tight_layout()
            _save(fig, os.path.join(epoch_out,
                                    f"weights_per_neuron_{_pfx(name)}_"
                                    f"sample{s:03d}_neuron{nid:04d}.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Per-run visualization
# ─────────────────────────────────────────────────────────────────────────────

def visualize_run(run_label, run_dir, viz_cfg, out_base):
    group_cfg          = viz_cfg.get("groups",            {}) or {}
    synapse_cfg        = viz_cfg.get("synapses",          {}) or {}
    weights_per_neuron = viz_cfg.get("weights_per_neuron",{}) or {}
    track_weight_delta = viz_cfg.get("track_weight_delta", False)
    visualize_samples  = viz_cfg.get("visualize_samples") or None
    visualize_epoch    = viz_cfg.get("visualize_epoch")   or []

    epoch_files = sorted(glob_mod.glob(os.path.join(run_dir, "history_epoch_*.npz")))
    if not epoch_files:
        print(f"  [skip] No epoch files in {run_dir}")
        return

    n_epochs      = len(epoch_files)
    epochs_to_viz = _epochs_to_visualize(n_epochs, visualize_epoch)

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

        sample_idxs = _sample_indices(n_samples, visualize_samples)
        epoch_out   = os.path.join(out_base, f"epoch_{epoch_idx:03d}")

        # ── Groups ────────────────────────────────────────────────────────────
        for name, gcfg in group_cfg.items():
            color = GROUP_COLORS.get(name, "mediumpurple")
            if gcfg.get("spike_raster"):
                _plot_spike_raster(data, keys, name, epoch_idx, sample_idxs, color, epoch_out)
                _plot_spike_counts(data, keys, name, epoch_idx, sample_idxs, color, epoch_out)
            if gcfg.get("mean_firing_rate"):
                _plot_mean_firing_rate(data, keys, name, epoch_idx, sample_idxs, color, epoch_out)
            if gcfg.get("membrane_potential"):
                _plot_membrane_potential(data, keys, name, epoch_idx, sample_idxs, epoch_out)

        # ── Synapses ──────────────────────────────────────────────────────────
        for name in synapse_cfg:
            _plot_weight_evolution(data, keys, name, epoch_idx, sample_idxs, epoch_out)
            _plot_weight_delta(data, keys, name, epoch_idx, track_weight_delta, epoch_out)
            _plot_synapse_weight_matrix(data, keys, name, epoch_idx, sample_idxs, epoch_out)
            _plot_weights_per_neuron(data, keys, name, epoch_idx, sample_idxs,
                                     weights_per_neuron, epoch_out)

        # ── Initial weight matrices (epoch 0 only) ────────────────────────────
        if epoch_idx == 0:
            init_out = os.path.join(out_base, "epoch_init")
            for name in synapse_cfg:
                if _has(keys, name, "init_weights"):
                    W = _get(data, name, "init_weights")
                    _plot_weight_matrix(
                        W,
                        f"Initial Weight Matrix — {name}",
                        os.path.join(init_out, f"init_weight_matrix_{_pfx(name)}.png"),
                    )


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

weights_per_neuron_fp = viz_cfg_A.get("weights_per_neuron", {}) or {}

for syn_name, neuron_ids in weights_per_neuron_fp.items():
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
