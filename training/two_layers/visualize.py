"""
visualize.py
------------
Load epoch history files (history_epoch_*.npz) and save all enabled plots as PNGs.

Each plot type has two versions:
  • Whole-dataset  — aggregated across all training samples
  • Per-sample     — one plot per index in visualize_samples []

Output structure:
  vizs/epoch_init/    — initial weight matrix
  vizs/epoch_0/       — plots for epoch 0
  vizs/epoch_1/       — plots for epoch 1
  ...
  vizs/final_top_k_weights.png  — top-k weights after all training

Usage:
    python visualize.py                                          # auto-locate history files + record_config.yaml
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml

# ── resolve paths ─────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
cfg_path = os.path.join(SCRIPT_DIR, "record_config.yaml")

# ── load config ───────────────────────────────────────────────────────────────

visualize_samples = None
visualize_epoch = None
if os.path.exists(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    visualize_samples = cfg.get("visualize_samples", None)
    visualize_epoch = cfg.get("visualize_epoch", [])

# If visualize_epoch is empty/None, visualize all epochs
if not visualize_epoch:
    visualize_epoch = None  # signal to use all epochs

# ── find epoch history files ──────────────────────────────────────────────────

import glob as glob_mod
epoch_files = sorted(glob_mod.glob(os.path.join(SCRIPT_DIR, "history_epoch_*.npz")))
init_file = os.path.join(SCRIPT_DIR, "history_init.npz")

if not epoch_files:
    print(f"ERROR: No history_epoch_*.npz files found in {SCRIPT_DIR}")
    sys.exit(1)

print(f"Found {len(epoch_files)} epoch file(s)")
print(f"Epochs: {[int(f.split('_')[-1].replace('.npz', '')) for f in epoch_files]}\n")

# ── output directory ──────────────────────────────────────────────────────────

OUT_DIR = os.path.join(SCRIPT_DIR, "vizs")
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Output base: {OUT_DIR}\n")

saved = []


def save(fig, name, epoch_dir=None):
    if epoch_dir is None:
        out_path = OUT_DIR
    else:
        out_path = os.path.join(OUT_DIR, epoch_dir)
        os.makedirs(out_path, exist_ok=True)
    path = os.path.join(out_path, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    print(f"  Saved: {os.path.relpath(path, SCRIPT_DIR)}")


def _sample_indices(n_total):
    """Return the list of sample indices to visualize."""
    if visualize_samples and len(visualize_samples) > 0:
        return [s for s in visualize_samples if s < n_total]
    return list(range(n_total))


def _epochs_to_visualize(total_epochs):
    """Return the list of epoch indices to visualize."""
    if visualize_epoch and len(visualize_epoch) > 0:
        return [e for e in visualize_epoch if e < total_epochs]
    return list(range(total_epochs))


def _window_mask(t, t_start, t_end):
    """Boolean mask for t within [t_start, t_end]. -1 means no bound."""
    mask = np.ones(len(t), dtype=bool)
    if t_start >= 0:
        mask &= t >= t_start
    if t_end >= 0:
        mask &= t <= t_end
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# Main epoch loop
# ══════════════════════════════════════════════════════════════════════════════

for epoch_idx, npz_path in enumerate(epoch_files):
    if visualize_epoch and epoch_idx not in _epochs_to_visualize(len(epoch_files)):
        continue

    epoch_dir = f"epoch_{epoch_idx}"
    print(f"\nProcessing Epoch {epoch_idx}...")

    data = np.load(npz_path, allow_pickle=True)
    keys = set(data.files)

    # ══════════════════════════════════════════════════════════════════════════════
    # 1. Weight evolution
    # ══════════════════════════════════════════════════════════════════════════════

    # ── 1a. Whole-dataset: x = sample index ──────────────────────────────────────
    if "we_pairs" in keys and "we_values" in keys and "we_times_ms" in keys:
        pairs    = data["we_pairs"]             # (n_pairs, 2)
        values   = data["we_values"]            # (n_pairs, n_total_snaps)
        times_ms = data["we_times_ms"]          # (n_total_snaps,)

    # Convert cumulative ms → sample index using boundary timestamps
    if "we_sample_boundaries_ms" in keys:
        boundaries = data["we_sample_boundaries_ms"]   # (n_samples+1,) or (n_samples,)
        # For each snapshot time, find which sample it belongs to
        sample_idx = np.searchsorted(boundaries, times_ms, side="right") - 1
        sample_idx = np.clip(sample_idx, 0, len(boundaries) - 1)
        x_axis     = sample_idx
        x_label    = "Sample index"
    else:
        # Fallback: just use snapshot number
        x_axis  = np.arange(values.shape[1])
        x_label = "Snapshot index"

        for k, (pi, pj) in enumerate(pairs):
            fig, ax = plt.subplots(figsize=(11, 3))
            ax.plot(x_axis, values[k], lw=1.5, color=f"C{k % 10}")
            ax.set_title(
                f"Weight Evolution (epoch {epoch_idx}, all samples) — synapse in[{pi}] → hid[{pj}]",
                fontsize=12, fontweight="bold"
            )
            ax.set_xlabel(x_label)
            ax.set_ylabel("Weight")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            save(fig, f"weight_evolution_all_in{pi:04d}_hid{pj:04d}.png", epoch_dir=epoch_dir)

        # ── 1b. Per-sample: one plot per synapse pair × per visualize_samples ─────
        if "we_sample_values" in keys and "we_sample_times" in keys and "we_n_samples" in keys:
            sample_values = data["we_sample_values"]   # object array
            sample_times  = data["we_sample_times"]    # object array
            n_total       = int(data["we_n_samples"])

            for s in _sample_indices(n_total):
                sv = sample_values[s]   # (n_pairs, n_snaps_this_sample)
                st = sample_times[s]    # (n_snaps_this_sample,)
                if len(st) == 0:
                    continue
                # Normalise times to start from 0 within the sample
                st_rel = st - st[0]

                for k, (pi, pj) in enumerate(pairs):
                    fig, ax = plt.subplots(figsize=(11, 3))
                    ax.plot(st_rel, sv[k], lw=1.5, color=f"C{k % 10}")
                    ax.set_title(
                        f"Weight Evolution (epoch {epoch_idx}, sample {s}) — synapse in[{pi}] → hid[{pj}]",
                        fontsize=12, fontweight="bold"
                    )
                    ax.set_xlabel("Time within sample (ms)")
                    ax.set_ylabel("Weight")
                    ax.set_ylim(0, 1)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    save(fig, f"weight_evolution_sample{s:03d}_in{pi:04d}_hid{pj:04d}.png", epoch_dir=epoch_dir)


    # ══════════════════════════════════════════════════════════════════════════════
    # 2. Membrane potential — input layer
    # ══════════════════════════════════════════════════════════════════════════════

    if "vmon_in_v_all" in keys and "vmon_in_t_all" in keys:
        neurons    = data["vmon_in_indices"]
        v_all      = data["vmon_in_v_all"]
        t_all      = data["vmon_in_t_all"]
        windows    = data["vmon_in_windows"]
        n_total    = int(data["vmon_in_n_samples"])
        v_th_in    = float(data["v_th_in"]) if "v_th_in" in keys else 1.0

        win_lookup = {int(row[0]): (float(row[1]), float(row[2])) for row in windows}

        for s in _sample_indices(n_total):
            v = v_all[s]
            t = t_all[s]

            for k, nid in enumerate(neurons):
                t_start, t_end = win_lookup.get(int(nid), (-1.0, -1.0))
                mask = _window_mask(t, t_start, t_end)
                t_w  = t[mask]
                v_w  = v[k][mask]

                spike_times = t_w[np.where((v_w[:-1] < v_th_in) & (v_w[1:] >= v_th_in))[0] + 1] \
                    if len(v_w) > 1 else np.array([])

                fig, ax = plt.subplots(figsize=(11, 3))
                ax.plot(t_w, v_w, lw=0.8, color="steelblue", zorder=2, label="v")
                ax.axhline(v_th_in, color="crimson", lw=1.0, ls="--", zorder=3, label=f"threshold ({v_th_in})")

                if len(spike_times) > 0:
                    ax.vlines(spike_times, ymin=v_th_in, ymax=v_th_in * 1.25,
                              color="crimson", lw=1.0, zorder=4, label=f"spikes ({len(spike_times)})")

                win_str = f"  [{t_start:.0f}–{t_end:.0f} ms]" if t_start >= 0 else ""
                ax.set_title(f"Membrane Potential — Input Neuron {nid}  |  Epoch {epoch_idx}, Sample {s}{win_str}",
                             fontsize=11, fontweight="bold")
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("v (a.u.)")
                ax.legend(fontsize=8, loc="upper right")
                ax.grid(True, alpha=0.3)
                y_lo = min(0.0, v_w.min() * 1.05) if len(v_w) else 0.0
                y_hi = max(v_th_in * 1.35, v_w.max() * 1.1) if len(v_w) else v_th_in * 1.5
                ax.set_ylim(y_lo, y_hi)
                plt.tight_layout()
                save(fig, f"vmon_input_sample{s:03d}_neuron{nid:04d}.png", epoch_dir=epoch_dir)


    # ══════════════════════════════════════════════════════════════════════════════
    # 3. Membrane potential — hidden layer
    # ══════════════════════════════════════════════════════════════════════════════

    if "vmon_hid_v_all" in keys and "vmon_hid_t_all" in keys:
        neurons  = data["vmon_hid_indices"]
        v_all    = data["vmon_hid_v_all"]
        vth_all  = data["vmon_hid_vth_all"]
        t_all    = data["vmon_hid_t_all"]
        windows  = data["vmon_hid_windows"]
        n_total  = int(data["vmon_hid_n_samples"])

        win_lookup = {int(row[0]): (float(row[1]), float(row[2])) for row in windows}

        for s in _sample_indices(n_total):
            v   = v_all[s]
            vth = vth_all[s]
            t   = t_all[s]

            for k, nid in enumerate(neurons):
                t_start, t_end = win_lookup.get(int(nid), (-1.0, -1.0))
                mask  = _window_mask(t, t_start, t_end)
                t_w   = t[mask]
                v_w   = v[k][mask]
                vth_w = vth[k][mask]

                crossed = (v_w[:-1] < vth_w[:-1]) & (v_w[1:] >= vth_w[1:])
                spike_times = t_w[np.where(crossed)[0] + 1] if len(v_w) > 1 else np.array([])

                fig, ax = plt.subplots(figsize=(11, 3))
                ax.plot(t_w, v_w,   lw=0.8, color="steelblue", zorder=2, label="v")
                ax.plot(t_w, vth_w, lw=1.0, color="crimson",   zorder=3, ls="--", label="vth (adaptive)")

                if len(spike_times) > 0:
                    spike_vth = np.interp(spike_times, t_w, vth_w)
                    ax.vlines(spike_times, ymin=spike_vth, ymax=spike_vth * 1.25,
                              color="crimson", lw=1.0, zorder=4, label=f"spikes ({len(spike_times)})")

                win_str = f"  [{t_start:.0f}–{t_end:.0f} ms]" if t_start >= 0 else ""
                ax.set_title(f"Membrane Potential — Hidden Neuron {nid}  |  Epoch {epoch_idx}, Sample {s}{win_str}",
                             fontsize=11, fontweight="bold")
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("v (a.u.)")
                ax.legend(fontsize=8, loc="upper right")
                ax.grid(True, alpha=0.3)
                y_lo = min(0.0, v_w.min() * 1.05) if len(v_w) else 0.0
                y_hi = max(vth_w.max() * 1.35, v_w.max() * 1.1) if len(v_w) else 2.0
                ax.set_ylim(y_lo, y_hi)
                plt.tight_layout()
                save(fig, f"vmon_hidden_sample{s:03d}_neuron{nid:04d}.png", epoch_dir=epoch_dir)


    # ══════════════════════════════════════════════════════════════════════════════
    # 4. Spike raster — input layer
    # ══════════════════════════════════════════════════════════════════════════════

    if "raster_in_i" in keys and "raster_in_t" in keys:
        raster_i  = data["raster_in_i"]
        raster_t  = data["raster_in_t"]
        n_total   = int(data["raster_in_n_samples"])
        n_neurons = int(data["raster_in_n_neurons"])

        for s in _sample_indices(n_total):
            sp_i = raster_i[s]
            sp_t = raster_t[s]

            fig, ax = plt.subplots(figsize=(12, 5))
            if len(sp_t) > 0:
                ax.scatter(sp_t, sp_i, s=0.5, c="steelblue", linewidths=0, rasterized=True)
            ax.set_title(f"Spike Raster — Input Layer  |  Epoch {epoch_idx}, Sample {s}  "
                         f"({len(sp_t):,} spikes)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Neuron index")
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0, top=n_neurons - 1)
            ax.grid(True, alpha=0.2)
            plt.tight_layout()
            save(fig, f"raster_input_sample{s:03d}.png", epoch_dir=epoch_dir)


    # ══════════════════════════════════════════════════════════════════════════════
    # 5. Spike raster — hidden layer
    # ══════════════════════════════════════════════════════════════════════════════

    if "raster_hid_i" in keys and "raster_hid_t" in keys:
        raster_i  = data["raster_hid_i"]
        raster_t  = data["raster_hid_t"]
        n_total   = int(data["raster_hid_n_samples"])
        n_neurons = int(data["raster_hid_n_neurons"])

        for s in _sample_indices(n_total):
            sp_i = raster_i[s]
            sp_t = raster_t[s]

            fig, ax = plt.subplots(figsize=(12, 5))
            if len(sp_t) > 0:
                ax.scatter(sp_t, sp_i, s=0.5, c="mediumseagreen", linewidths=0, rasterized=True)
            ax.set_title(f"Spike Raster — Hidden Layer  |  Epoch {epoch_idx}, Sample {s}  "
                         f"({len(sp_t):,} spikes)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Neuron index")
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0, top=n_neurons - 1)
            ax.grid(True, alpha=0.2)
            plt.tight_layout()
            save(fig, f"raster_hidden_sample{s:03d}.png", epoch_dir=epoch_dir)


    # ══════════════════════════════════════════════════════════════════════════════
    # 6. Weight matrix
    # ══════════════════════════════════════════════════════════════════════════════

    def _plot_weight_matrix(W, title, filename, epoch_dir=None):
        """Shared heatmap + histogram layout for a weight matrix."""
        fig = plt.figure(figsize=(13, 5))
        gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1], figure=fig)
        fig.suptitle(title, fontsize=14, fontweight="bold")

        ax_heat = fig.add_subplot(gs[0])
        im = ax_heat.imshow(
            W,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0,
            vmax=W.max() if W.max() > 0 else 1,
            origin="lower",
        )
        ax_heat.set_xlabel("Hidden neuron index")
        ax_heat.set_ylabel("Input neuron index")
        plt.colorbar(im, ax=ax_heat, label="Weight")

        ax_hist = fig.add_subplot(gs[1])
        w_flat = W.flatten()
        ax_hist.hist(w_flat[w_flat > 0], bins=60, color="steelblue", edgecolor="none", density=True)
        ax_hist.set_xlabel("Weight value")
        ax_hist.set_ylabel("Density")
        ax_hist.set_title("Distribution\n(non-zero)")
        ax_hist.grid(True, alpha=0.3)

        plt.tight_layout()
        save(fig, filename, epoch_dir=epoch_dir)


    # ── 6a. Whole-dataset: final weight matrix after all training ─────────────────
    if "final_weights_matrix" in keys:
        W = data["final_weights_matrix"]
        _plot_weight_matrix(W, f"Final Weight Matrix (epoch {epoch_idx})", "final_weight_matrix.png", epoch_dir=epoch_dir)

    # ── 6b. Per-sample: weight matrix state after each selected sample ────────────
    if "weight_matrix_per_sample" in keys and "weight_matrix_n_samples" in keys:
        wm_per_sample = data["weight_matrix_per_sample"]   # object array
        n_total       = int(data["weight_matrix_n_samples"])

        for s in _sample_indices(n_total):
            W = wm_per_sample[s]   # (N_IN, N_H)
            _plot_weight_matrix(
                W,
                f"Weight Matrix (epoch {epoch_idx}, after sample {s})",
                f"weight_matrix_sample{s:03d}.png",
                epoch_dir=epoch_dir
            )


    # ══════════════════════════════════════════════════════════════════════════════
    # 7. Mean firing rate — input layer
    # ══════════════════════════════════════════════════════════════════════════════

    def _plot_firing_rate(rates, title, filename, color="darkorange", epoch_dir=None):
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
        save(fig, filename, epoch_dir=epoch_dir)


    # ── 7a. Whole-dataset aggregate ───────────────────────────────────────────────
    if "mean_firing_rate_input" in keys:
        _plot_firing_rate(
            data["mean_firing_rate_input"],
            f"Mean Firing Rate — Input Layer (epoch {epoch_idx}, all samples)",
            "mean_firing_rate_input_all.png",
            color="darkorange",
            epoch_dir=epoch_dir
        )

    # ── 7b. Per-sample ────────────────────────────────────────────────────────────
    if "mfr_in_sample_counts" in keys and "mfr_in_sample_dur_s" in keys and "mfr_in_n_samples" in keys:
        sample_counts = data["mfr_in_sample_counts"]   # object array of (N_IN,) int arrays
        sample_durs   = data["mfr_in_sample_dur_s"]    # (n_samples,) float
        n_total       = int(data["mfr_in_n_samples"])

        for s in _sample_indices(n_total):
            dur = float(sample_durs[s])
            rates = (sample_counts[s] / dur).astype(np.float32) if dur > 0 else np.zeros_like(sample_counts[s], dtype=np.float32)
            _plot_firing_rate(
                rates,
                f"Mean Firing Rate — Input Layer  |  Epoch {epoch_idx}, Sample {s}",
                f"mean_firing_rate_input_sample{s:03d}.png",
                color="darkorange",
                epoch_dir=epoch_dir
            )


    # ══════════════════════════════════════════════════════════════════════════════
    # 8. Mean firing rate — hidden layer
    # ══════════════════════════════════════════════════════════════════════════════

    # ── 8a. Whole-dataset aggregate ───────────────────────────────────────────────
    if "mean_firing_rate_hidden" in keys:
        _plot_firing_rate(
            data["mean_firing_rate_hidden"],
            f"Mean Firing Rate — Hidden Layer (epoch {epoch_idx}, all samples)",
            "mean_firing_rate_hidden_all.png",
            color="mediumseagreen",
            epoch_dir=epoch_dir
        )

    # ── 8b. Per-sample ────────────────────────────────────────────────────────────
    if "mfr_hid_sample_counts" in keys and "mfr_hid_sample_dur_s" in keys and "mfr_hid_n_samples" in keys:
        sample_counts = data["mfr_hid_sample_counts"]
        sample_durs   = data["mfr_hid_sample_dur_s"]
        n_total       = int(data["mfr_hid_n_samples"])

        for s in _sample_indices(n_total):
            dur = float(sample_durs[s])
            rates = (sample_counts[s] / dur).astype(np.float32) if dur > 0 else np.zeros_like(sample_counts[s], dtype=np.float32)
            _plot_firing_rate(
                rates,
                f"Mean Firing Rate — Hidden Layer  |  Epoch {epoch_idx}, Sample {s}",
                f"mean_firing_rate_hidden_sample{s:03d}.png",
                color="mediumseagreen",
                epoch_dir=epoch_dir
            )
            
# ══════════════════════════════════════════════════════════════════════════════
# 0. Initial weight matrix
# ══════════════════════════════════════════════════════════════════════════════

if os.path.exists(init_file):
    print("Visualizing initial weight matrix...")
    init_data = np.load(init_file, allow_pickle=True)
    if "init_weight_matrix" in init_data.files:
        W_init = init_data["init_weight_matrix"]
        _plot_weight_matrix(W_init, "Initial Weight Matrix", "init_weight_matrix.png", epoch_dir="epoch_init")


# ── finish ────────────────────────────────────────────────────────────────────

print(f"\n{len(saved)} PNG(s) saved to: {OUT_DIR}")
if not saved:
    print("No recognised keys found — nothing was plotted.")