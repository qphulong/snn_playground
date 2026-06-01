"""
visualize.py
------------
Load epoch history files from run_A/ and run_B/ and save all enabled plots
as side-by-side PNGs (Group A on the left, Group B on the right).

Output structure:
  vizs/epoch_init/    — initial weight matrices
  vizs/epoch_0/       — plots for epoch 0
  vizs/epoch_1/       — plots for epoch 1
  ...
  vizs/fingerprints/  — final fingerprint comparison

Usage:
    python visualize.py
"""

import sys
import os
import glob as glob_mod
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml

# ── resolve paths ─────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
cfg_path   = os.path.join(SCRIPT_DIR, "record_and_visualize_config.yaml")
RUN_A_DIR  = os.path.join(SCRIPT_DIR, "run_A")
RUN_B_DIR  = os.path.join(SCRIPT_DIR, "run_B")

# ── load config ───────────────────────────────────────────────────────────────

visualize_samples    = None
visualize_epoch      = []
weights_per_neuron   = {}
track_weight_delta   = False
group_cfg            = {}
synapse_cfg          = {}

if os.path.exists(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    viz_cfg = cfg.get("visualize", cfg)
    visualize_samples  = viz_cfg.get("visualize_samples",  None)
    visualize_epoch    = viz_cfg.get("visualize_epoch",    [])
    weights_per_neuron = viz_cfg.get("weights_per_neuron", {}) or {}
    track_weight_delta = viz_cfg.get("track_weight_delta", False)
    group_cfg          = viz_cfg.get("groups",   {}) or {}
    synapse_cfg        = viz_cfg.get("synapses", {}) or {}

# ── find epoch file pairs ─────────────────────────────────────────────────────

epoch_files_A = sorted(glob_mod.glob(os.path.join(RUN_A_DIR, "history_epoch_*.npz")))
epoch_files_B = sorted(glob_mod.glob(os.path.join(RUN_B_DIR, "history_epoch_*.npz")))

if not epoch_files_A or not epoch_files_B:
    print(f"ERROR: Missing history files.\n  run_A: {len(epoch_files_A)} file(s)\n  run_B: {len(epoch_files_B)} file(s)")
    sys.exit(1)

n_epochs = min(len(epoch_files_A), len(epoch_files_B))
print(f"Found {n_epochs} epoch pair(s)")

# ── output directory ──────────────────────────────────────────────────────────

OUT_DIR = os.path.join(SCRIPT_DIR, "vizs")
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Output base: {OUT_DIR}\n")

saved = []


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def save(fig, name, epoch_dir=None):
    out = OUT_DIR if epoch_dir is None else os.path.join(OUT_DIR, epoch_dir)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    print(f"  Saved: {os.path.relpath(path, SCRIPT_DIR)}")


def _sample_indices(n_total):
    if visualize_samples:
        return [s for s in visualize_samples if s < n_total]
    return list(range(n_total))


def _epochs_to_visualize(total):
    if visualize_epoch:
        return [e for e in visualize_epoch if e < total]
    return list(range(total))


def _window_mask(t, t_start, t_end):
    mask = np.ones(len(t), dtype=bool)
    if t_start >= 0:
        mask &= t >= t_start
    if t_end >= 0:
        mask &= t <= t_end
    return mask


def _pfx(name):
    return name.replace("->", "_to_")


def _k(name, field):
    return f"{_pfx(name)}__{field}"


def _has(keys, name, field):
    return _k(name, field) in keys


def _get(data, name, field):
    return data[_k(name, field)]


# ══════════════════════════════════════════════════════════════════════════════
# Side-by-side plot helpers
# ══════════════════════════════════════════════════════════════════════════════

def _plot_weight_matrix_pair(W_A, W_B, title, filename, epoch_dir=None):
    vmax = max(W_A.max() if W_A.max() > 0 else 1,
               W_B.max() if W_B.max() > 0 else 1)

    fig = plt.figure(figsize=(26, 5))
    gs  = gridspec.GridSpec(1, 4, width_ratios=[3, 1, 3, 1], wspace=0.35)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for col_offset, (W, label) in enumerate([(W_A, "Group A"), (W_B, "Group B")]):
        ax_heat = fig.add_subplot(gs[col_offset * 2])
        ax_hist = fig.add_subplot(gs[col_offset * 2 + 1])

        ax_heat.set_title(label, fontsize=11)
        im = ax_heat.imshow(
            W, aspect="auto", interpolation="nearest", cmap="viridis",
            vmin=0, vmax=vmax, origin="lower",
        )
        ax_heat.set_xlabel("Post-synaptic neuron index")
        ax_heat.set_ylabel("Pre-synaptic neuron index")
        plt.colorbar(im, ax=ax_heat, label="Weight")

        w_flat = W.flatten()
        ax_hist.hist(w_flat[w_flat > 0], bins=60, color="steelblue",
                     edgecolor="none", density=True)
        ax_hist.set_xlabel("Weight value")
        ax_hist.set_ylabel("Density")
        ax_hist.set_title(f"Distribution\n(non-zero)")
        ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    save(fig, filename, epoch_dir=epoch_dir)


def _plot_firing_rate_pair(rates_A, rates_B, title, filename, color="darkorange", epoch_dir=None):
    fig, (ax_A, ax_B) = plt.subplots(1, 2, figsize=(22, 4), sharey=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, rates, label in [(ax_A, rates_A, "Group A"), (ax_B, rates_B, "Group B")]:
        n = len(rates)
        ax.bar(np.arange(n), rates, width=1.0, color=color, linewidth=0)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Neuron index")
        ax.set_ylabel("Mean rate (Hz)")
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save(fig, filename, epoch_dir=epoch_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Per-group plot routines  (paired A/B)
# ══════════════════════════════════════════════════════════════════════════════

def _plot_spike_raster(data_A, keys_A, data_B, keys_B, name, epoch_idx, epoch_dir, color):
    can_A = _has(keys_A, name, "raster_i") and _has(keys_A, name, "raster_t")
    can_B = _has(keys_B, name, "raster_i") and _has(keys_B, name, "raster_t")
    if not (can_A or can_B):
        return

    n_total_A   = int(_get(data_A, name, "raster_n_samples")) if can_A else 0
    n_total_B   = int(_get(data_B, name, "raster_n_samples")) if can_B else 0
    n_neurons_A = int(_get(data_A, name, "raster_n_neurons")) if can_A else 0
    n_neurons_B = int(_get(data_B, name, "raster_n_neurons")) if can_B else 0
    n_total     = max(n_total_A, n_total_B)

    for s in _sample_indices(n_total):
        fig, (ax_A, ax_B) = plt.subplots(1, 2, figsize=(22, 5))
        fig.suptitle(
            f"Spike Raster — {name}  |  Epoch {epoch_idx}, Sample {s}",
            fontsize=12, fontweight="bold"
        )

        for ax, data, keys, can, n_neurons, n_tot, label in [
            (ax_A, data_A, keys_A, can_A, n_neurons_A, n_total_A, "Group A"),
            (ax_B, data_B, keys_B, can_B, n_neurons_B, n_total_B, "Group B"),
        ]:
            ax.set_title(label, fontsize=11)
            if can and s < n_tot:
                sp_i = _get(data, name, "raster_i")[s]
                sp_t = _get(data, name, "raster_t")[s]
                if len(sp_t) > 0:
                    ax.scatter(sp_t, sp_i, s=0.5, c=color, linewidths=0, rasterized=True)
                ax.set_title(f"{label}  ({len(sp_t):,} spikes)", fontsize=11)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Neuron index")
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0, top=max(n_neurons - 1, 1))
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        save(fig, f"raster_{_pfx(name)}_sample{s:03d}.png", epoch_dir=epoch_dir)


def _plot_spike_counts(data_A, keys_A, data_B, keys_B, name, epoch_idx, epoch_dir, color):
    can_A = _has(keys_A, name, "raster_i")
    can_B = _has(keys_B, name, "raster_i")
    if not (can_A or can_B):
        return

    n_total_A   = int(_get(data_A, name, "raster_n_samples")) if can_A else 0
    n_total_B   = int(_get(data_B, name, "raster_n_samples")) if can_B else 0
    n_neurons_A = int(_get(data_A, name, "raster_n_neurons")) if can_A else 0
    n_neurons_B = int(_get(data_B, name, "raster_n_neurons")) if can_B else 0
    n_total     = max(n_total_A, n_total_B)
    n_neurons   = max(n_neurons_A, n_neurons_B)

    for s in _sample_indices(n_total):
        counts_A = np.zeros(n_neurons, dtype=np.int32)
        counts_B = np.zeros(n_neurons, dtype=np.int32)

        if can_A and s < n_total_A:
            sp_i = _get(data_A, name, "raster_i")[s]
            if len(sp_i) > 0:
                counts_A = np.bincount(sp_i.astype(np.int32), minlength=n_neurons)

        if can_B and s < n_total_B:
            sp_i = _get(data_B, name, "raster_i")[s]
            if len(sp_i) > 0:
                counts_B = np.bincount(sp_i.astype(np.int32), minlength=n_neurons)

        fig, (ax_A, ax_B) = plt.subplots(1, 2, figsize=(22, 4))
        fig.suptitle(
            f"Spike Count — {name}  |  Epoch {epoch_idx}, Sample {s}",
            fontsize=13, fontweight="bold"
        )

        for ax, counts, label in [(ax_A, counts_A, "Group A"), (ax_B, counts_B, "Group B")]:
            ax.bar(np.arange(n_neurons), counts, width=1.0, color=color, linewidth=0)
            ax.set_title(label, fontsize=11)
            ax.set_xlabel("Neuron index")
            ax.set_ylabel("Spike count")
            ax.set_xlim(-0.5, n_neurons - 0.5)
            ax.set_ylim(bottom=0)
            ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        save(fig, f"spike_count_{_pfx(name)}_sample{s:03d}.png", epoch_dir=epoch_dir)


def _plot_mean_firing_rate(data_A, keys_A, data_B, keys_B, name, epoch_idx, epoch_dir, color):
    pfx = _pfx(name)

    # Whole-epoch aggregate
    has_mfr_A = _has(keys_A, name, "mfr")
    has_mfr_B = _has(keys_B, name, "mfr")
    if has_mfr_A or has_mfr_B:
        rates_A = _get(data_A, name, "mfr") if has_mfr_A else np.array([])
        rates_B = _get(data_B, name, "mfr") if has_mfr_B else np.array([])
        if len(rates_A) == 0:
            rates_A = np.zeros_like(rates_B)
        if len(rates_B) == 0:
            rates_B = np.zeros_like(rates_A)
        _plot_firing_rate_pair(
            rates_A, rates_B,
            f"Mean Firing Rate — {name}  |  Epoch {epoch_idx}, all samples",
            f"mean_firing_rate_{pfx}_all.png",
            color=color, epoch_dir=epoch_dir
        )

    # Per-sample
    has_ps_A = _has(keys_A, name, "mfr_sample_counts") and _has(keys_A, name, "mfr_sample_dur_s")
    has_ps_B = _has(keys_B, name, "mfr_sample_counts") and _has(keys_B, name, "mfr_sample_dur_s")
    if has_ps_A or has_ps_B:
        n_total_A = len(_get(data_A, name, "mfr_sample_dur_s")) if has_ps_A else 0
        n_total_B = len(_get(data_B, name, "mfr_sample_dur_s")) if has_ps_B else 0
        n_total   = max(n_total_A, n_total_B)

        for s in _sample_indices(n_total):
            def _mfr_for(data, has, n_tot):
                if not has or s >= n_tot:
                    return np.array([])
                dur    = float(_get(data, name, "mfr_sample_dur_s")[s])
                counts = _get(data, name, "mfr_sample_counts")[s]
                return (counts / dur if dur > 0 else np.zeros_like(counts, dtype=np.float32))

            rates_A = _mfr_for(data_A, has_ps_A, n_total_A).astype(np.float32)
            rates_B = _mfr_for(data_B, has_ps_B, n_total_B).astype(np.float32)
            if len(rates_A) == 0:
                rates_A = np.zeros_like(rates_B)
            if len(rates_B) == 0:
                rates_B = np.zeros_like(rates_A)

            _plot_firing_rate_pair(
                rates_A, rates_B,
                f"Mean Firing Rate — {name}  |  Epoch {epoch_idx}, Sample {s}",
                f"mean_firing_rate_{pfx}_sample{s:03d}.png",
                color=color, epoch_dir=epoch_dir
            )


def _plot_membrane_potential(data_A, keys_A, data_B, keys_B, name, epoch_idx, epoch_dir):
    can_A = _k(name, "vmon_v_all") in keys_A and _k(name, "vmon_t_all") in keys_A
    can_B = _k(name, "vmon_v_all") in keys_B and _k(name, "vmon_t_all") in keys_B
    if not (can_A or can_B):
        return

    pfx = _pfx(name)

    def _var_keys(data, keys):
        tag = f"{_pfx(name)}__vmon_"
        out = {}
        for k in keys:
            if k.startswith(tag) and k.endswith("_all"):
                var = k[len(tag):-4]
                if var != "t":
                    out[var] = k
        return out

    var_keys_A = _var_keys(data_A, keys_A) if can_A else {}
    var_keys_B = _var_keys(data_B, keys_B) if can_B else {}
    all_vars   = sorted(set(list(var_keys_A.keys()) + list(var_keys_B.keys())))
    if not all_vars:
        return

    ref_data, ref_can = (data_A, can_A) if can_A else (data_B, can_B)
    neurons  = _get(ref_data, name, "vmon_indices")
    windows  = _get(ref_data, name, "vmon_windows")
    n_total_A = len(_get(data_A, name, "vmon_t_all")) if can_A else 0
    n_total_B = len(_get(data_B, name, "vmon_t_all")) if can_B else 0
    n_total   = max(n_total_A, n_total_B)

    colors = ["steelblue", "crimson", "darkorange", "mediumseagreen",
              "mediumpurple", "saddlebrown"]

    for s in _sample_indices(n_total):
        t_A = _get(data_A, name, "vmon_t_all")[s] if (can_A and s < n_total_A) else np.array([])
        t_B = _get(data_B, name, "vmon_t_all")[s] if (can_B and s < n_total_B) else np.array([])

        for k_idx, nid in enumerate(neurons):
            t_start = float(windows[k_idx, 1]) if windows[k_idx, 1] >= 0 else -1.0
            t_end   = float(windows[k_idx, 2]) if windows[k_idx, 2] >= 0 else -1.0

            n_vars  = len(all_vars)
            fig     = plt.figure(figsize=(22, 3 * n_vars))
            gs      = gridspec.GridSpec(n_vars, 2, figure=fig)
            win_str = f"  [{t_start:.0f}–{t_end:.0f} ms]" if t_start >= 0 else ""
            fig.suptitle(
                f"State Variables — {name}, Neuron {nid}  |  "
                f"Epoch {epoch_idx}, Sample {s}{win_str}",
                fontsize=11, fontweight="bold"
            )

            for ax_i, var in enumerate(all_vars):
                for col, (data, keys, can, n_tot, t_arr, label) in enumerate([
                    (data_A, keys_A, can_A, n_total_A, t_A, "Group A"),
                    (data_B, keys_B, can_B, n_total_B, t_B, "Group B"),
                ]):
                    ax = fig.add_subplot(gs[ax_i, col])
                    ax.set_ylabel(var)
                    ax.grid(True, alpha=0.3)
                    if ax_i == 0:
                        ax.set_title(label, fontsize=11)

                    var_keys = _var_keys(data, keys)
                    if can and s < n_tot and var in var_keys:
                        mask = _window_mask(t_arr, t_start, t_end)
                        t_w  = t_arr[mask]
                        arr  = data[var_keys[var]][s]
                        y_w  = arr[k_idx][mask]
                        ax.plot(t_w, y_w, lw=0.8, color=colors[ax_i % len(colors)], label=var)
                        ax.legend(fontsize=8, loc="upper right")

                    if ax_i == n_vars - 1:
                        ax.set_xlabel("Time (ms)")

            plt.tight_layout()
            win_suffix = (f"_window{t_start:.0f}_{t_end:.0f}ms" if t_start >= 0 else "")
            save(fig,
                 f"vmon_{pfx}_sample{s:03d}_neuron{nid:04d}{win_suffix}.png",
                 epoch_dir=epoch_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Per-synapse plot routines  (paired A/B)
# ══════════════════════════════════════════════════════════════════════════════

def _plot_weight_evolution(data_A, keys_A, data_B, keys_B, name, epoch_idx, epoch_dir):
    can_A = _has(keys_A, name, "we_pairs")
    can_B = _has(keys_B, name, "we_pairs")
    if not (can_A or can_B):
        return

    pfx    = _pfx(name)
    ref_d  = data_A if can_A else data_B
    ref_ks = keys_A if can_A else keys_B
    pairs  = _get(ref_d, name, "we_pairs")

    values_A = _get(data_A, name, "we_values") if can_A else None
    values_B = _get(data_B, name, "we_values") if can_B else None
    n_snaps  = (values_A.shape[1] if values_A is not None else values_B.shape[1])

    x_axis = np.arange(n_snaps)

    for k, (pi, pj) in enumerate(pairs):
        fig, (ax_A, ax_B) = plt.subplots(1, 2, figsize=(22, 3), sharey=True)
        fig.suptitle(
            f"Weight Evolution (epoch {epoch_idx}, all samples) — "
            f"{name}  pre[{pi}] → post[{pj}]",
            fontsize=12, fontweight="bold"
        )

        for ax, values, can, label in [
            (ax_A, values_A, can_A, "Group A"),
            (ax_B, values_B, can_B, "Group B"),
        ]:
            ax.set_title(label, fontsize=11)
            if can and values is not None:
                ax.plot(x_axis, values[k], lw=1.5, color=f"C{k % 10}")
            ax.set_xlabel("Snapshot index")
            ax.set_ylabel("Weight")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save(fig, f"weight_evolution_{pfx}_all_pre{pi:04d}_post{pj:04d}.png",
             epoch_dir=epoch_dir)

    # Per-sample
    has_ps_A = can_A and _has(keys_A, name, "we_sample_values") and _has(keys_A, name, "we_sample_times")
    has_ps_B = can_B and _has(keys_B, name, "we_sample_values") and _has(keys_B, name, "we_sample_times")
    if not (has_ps_A or has_ps_B):
        return

    n_total_A = len(_get(data_A, name, "we_sample_times")) if has_ps_A else 0
    n_total_B = len(_get(data_B, name, "we_sample_times")) if has_ps_B else 0
    n_total   = max(n_total_A, n_total_B)

    for s in _sample_indices(n_total):
        sv_A = _get(data_A, name, "we_sample_values")[s] if (has_ps_A and s < n_total_A) else None
        st_A = _get(data_A, name, "we_sample_times")[s]  if (has_ps_A and s < n_total_A) else np.array([])
        sv_B = _get(data_B, name, "we_sample_values")[s] if (has_ps_B and s < n_total_B) else None
        st_B = _get(data_B, name, "we_sample_times")[s]  if (has_ps_B and s < n_total_B) else np.array([])

        for k, (pi, pj) in enumerate(pairs):
            fig, (ax_A, ax_B) = plt.subplots(1, 2, figsize=(22, 3), sharey=True)
            fig.suptitle(
                f"Weight Evolution (epoch {epoch_idx}, sample {s}) — "
                f"{name}  pre[{pi}] → post[{pj}]",
                fontsize=12, fontweight="bold"
            )

            for ax, sv, st, label in [
                (ax_A, sv_A, st_A, "Group A"),
                (ax_B, sv_B, st_B, "Group B"),
            ]:
                ax.set_title(label, fontsize=11)
                if sv is not None and len(st) > 0:
                    st_rel = st - st[0]
                    ax.plot(st_rel, sv[k], lw=1.5, color=f"C{k % 10}")
                ax.set_xlabel("Time within sample (ms)")
                ax.set_ylabel("Weight")
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            save(fig,
                 f"weight_evolution_{pfx}_sample{s:03d}_pre{pi:04d}_post{pj:04d}.png",
                 epoch_dir=epoch_dir)


def _plot_weight_delta(data_A, keys_A, data_B, keys_B, name, epoch_idx, epoch_dir):
    if not track_weight_delta:
        return
    can_A = _has(keys_A, name, "we_pairs")
    can_B = _has(keys_B, name, "we_pairs")
    if not (can_A or can_B):
        return

    pfx    = _pfx(name)
    ref_d  = data_A if can_A else data_B
    pairs  = _get(ref_d, name, "we_pairs")

    values_A = _get(data_A, name, "we_values") if can_A else None
    values_B = _get(data_B, name, "we_values") if can_B else None
    n_snaps  = (values_A.shape[1] if values_A is not None else values_B.shape[1])
    x_axis   = np.arange(n_snaps)

    for k, (pi, pj) in enumerate(pairs):
        fig, (ax_A, ax_B) = plt.subplots(1, 2, figsize=(22, 3), sharey=True)
        fig.suptitle(
            f"Weight Delta (epoch {epoch_idx}) — {name}  pre[{pi}] → post[{pj}]",
            fontsize=12, fontweight="bold"
        )

        for ax, values, can, label in [
            (ax_A, values_A, can_A, "Group A"),
            (ax_B, values_B, can_B, "Group B"),
        ]:
            ax.set_title(label, fontsize=11)
            if can and values is not None:
                deltas = np.diff(values[k], prepend=values[k, 0])
                ax.plot(x_axis, deltas, lw=1.5, color=f"C{k % 10}", marker="o", markersize=3)
                ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)
            ax.set_xlabel("Snapshot index")
            ax.set_ylabel("Δw")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save(fig, f"weight_delta_{pfx}_all_pre{pi:04d}_post{pj:04d}.png",
             epoch_dir=epoch_dir)


def _plot_synapse_weight_matrix(data_A, keys_A, data_B, keys_B, name, epoch_idx, epoch_dir):
    pfx = _pfx(name)

    # Final weight matrix (end of epoch)
    has_fw_A = _has(keys_A, name, "final_weights")
    has_fw_B = _has(keys_B, name, "final_weights")
    if has_fw_A or has_fw_B:
        W_A = _get(data_A, name, "final_weights") if has_fw_A else np.zeros((1, 1))
        W_B = _get(data_B, name, "final_weights") if has_fw_B else np.zeros((1, 1))
        _plot_weight_matrix_pair(
            W_A, W_B,
            f"Final Weight Matrix — {name}  |  Epoch {epoch_idx}",
            f"final_weight_matrix_{pfx}.png",
            epoch_dir=epoch_dir
        )

    # Per-sample snapshots
    has_ps_A = _has(keys_A, name, "weight_per_sample") and _has(keys_A, name, "weight_n_samples")
    has_ps_B = _has(keys_B, name, "weight_per_sample") and _has(keys_B, name, "weight_n_samples")
    if not (has_ps_A or has_ps_B):
        return

    n_total_A = int(_get(data_A, name, "weight_n_samples")) if has_ps_A else 0
    n_total_B = int(_get(data_B, name, "weight_n_samples")) if has_ps_B else 0
    n_total   = max(n_total_A, n_total_B)

    for s in _sample_indices(n_total):
        W_A = (_get(data_A, name, "weight_per_sample")[s]
               if (has_ps_A and s < n_total_A) else np.zeros((1, 1)))
        W_B = (_get(data_B, name, "weight_per_sample")[s]
               if (has_ps_B and s < n_total_B) else np.zeros((1, 1)))
        _plot_weight_matrix_pair(
            W_A, W_B,
            f"Weight Matrix — {name}  |  Epoch {epoch_idx}, after sample {s}",
            f"weight_matrix_{pfx}_sample{s:03d}.png",
            epoch_dir=epoch_dir
        )


def _plot_weights_per_neuron(data_A, keys_A, data_B, keys_B, name, epoch_idx, epoch_dir):
    neuron_ids = weights_per_neuron.get(name, [])
    if not neuron_ids:
        return

    has_ps_A = _has(keys_A, name, "weight_per_sample") and _has(keys_A, name, "weight_n_samples")
    has_ps_B = _has(keys_B, name, "weight_per_sample") and _has(keys_B, name, "weight_n_samples")
    if not (has_ps_A or has_ps_B):
        return

    n_total_A = int(_get(data_A, name, "weight_n_samples")) if has_ps_A else 0
    n_total_B = int(_get(data_B, name, "weight_n_samples")) if has_ps_B else 0
    n_total   = max(n_total_A, n_total_B)

    for s in _sample_indices(n_total):
        W_A = (_get(data_A, name, "weight_per_sample")[s]
               if (has_ps_A and s < n_total_A) else None)
        W_B = (_get(data_B, name, "weight_per_sample")[s]
               if (has_ps_B and s < n_total_B) else None)

        for nid in neuron_ids:
            nid = int(nid)
            weights_A = W_A[:, nid] if (W_A is not None and nid < W_A.shape[1]) else np.array([])
            weights_B = W_B[:, nid] if (W_B is not None and nid < W_B.shape[1]) else np.array([])
            n_pre = max(len(weights_A), len(weights_B))

            fig = plt.figure(figsize=(26, 5))
            gs  = gridspec.GridSpec(1, 4, width_ratios=[2, 1, 2, 1], wspace=0.35)
            fig.suptitle(
                f"Incoming Weights — {name}, post-neuron {nid}  |  "
                f"Epoch {epoch_idx}, Sample {s}",
                fontsize=13, fontweight="bold"
            )

            for col_offset, (weights, label) in enumerate(
                [(weights_A, "Group A"), (weights_B, "Group B")]
            ):
                ax_bar  = fig.add_subplot(gs[col_offset * 2])
                ax_hist = fig.add_subplot(gs[col_offset * 2 + 1])
                ax_bar.set_title(label, fontsize=11)

                if len(weights) > 0:
                    ax_bar.bar(np.arange(len(weights)), weights,
                               width=0.8, color="steelblue", linewidth=0)
                    ax_bar.set_xlim(-0.5, len(weights) - 0.5)

                    w_nz = weights[weights > 0]
                    if len(w_nz) > 0:
                        ax_hist.hist(w_nz, bins=40, color="steelblue",
                                     edgecolor="none", density=True)

                ax_bar.set_xlabel("Input neuron index")
                ax_bar.set_ylabel("Weight magnitude")
                ax_bar.set_ylim(bottom=0)
                ax_bar.grid(True, axis="y", alpha=0.3)
                ax_hist.set_xlabel("Weight value")
                ax_hist.set_ylabel("Density")
                ax_hist.set_title("Distribution\n(non-zero)")
                ax_hist.grid(True, alpha=0.3)

            plt.tight_layout()
            save(fig,
                 f"weights_per_neuron_{_pfx(name)}_sample{s:03d}_neuron{nid:04d}.png",
                 epoch_dir=epoch_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Colour palette per group
# ══════════════════════════════════════════════════════════════════════════════

GROUP_COLORS = {
    "input":  "steelblue",
    "hidden": "mediumseagreen",
    "output": "darkorange",
}

def _group_color(name):
    return GROUP_COLORS.get(name, "mediumpurple")


# ══════════════════════════════════════════════════════════════════════════════
# Main epoch loop
# ══════════════════════════════════════════════════════════════════════════════

for epoch_idx in range(n_epochs):
    if epoch_idx not in _epochs_to_visualize(n_epochs):
        continue

    epoch_dir = f"epoch_{epoch_idx}"
    path_A    = epoch_files_A[epoch_idx]
    path_B    = epoch_files_B[epoch_idx]
    print(f"\nProcessing Epoch {epoch_idx}")
    print(f"  A: {os.path.basename(path_A)}")
    print(f"  B: {os.path.basename(path_B)}")

    data_A = np.load(path_A, allow_pickle=True)
    data_B = np.load(path_B, allow_pickle=True)
    keys_A = set(data_A.files)
    keys_B = set(data_B.files)

    # ── Groups ────────────────────────────────────────────────────────────────
    for name in group_cfg:
        color = _group_color(name)
        _plot_spike_raster(data_A, keys_A, data_B, keys_B, name, epoch_idx, epoch_dir, color)
        _plot_spike_counts(data_A, keys_A, data_B, keys_B, name, epoch_idx, epoch_dir, color)
        _plot_mean_firing_rate(data_A, keys_A, data_B, keys_B, name, epoch_idx, epoch_dir, color)
        _plot_membrane_potential(data_A, keys_A, data_B, keys_B, name, epoch_idx, epoch_dir)

    # ── Synapses ──────────────────────────────────────────────────────────────
    for name in synapse_cfg:
        _plot_weight_evolution(data_A, keys_A, data_B, keys_B, name, epoch_idx, epoch_dir)
        _plot_weight_delta(data_A, keys_A, data_B, keys_B, name, epoch_idx, epoch_dir)
        _plot_synapse_weight_matrix(data_A, keys_A, data_B, keys_B, name, epoch_idx, epoch_dir)
        _plot_weights_per_neuron(data_A, keys_A, data_B, keys_B, name, epoch_idx, epoch_dir)

    # ── Initial weight matrices (epoch 0 only) ────────────────────────────────
    if epoch_idx == 0:
        for name in synapse_cfg:
            has_iw_A = _has(keys_A, name, "init_weights")
            has_iw_B = _has(keys_B, name, "init_weights")
            if has_iw_A or has_iw_B:
                W_A = _get(data_A, name, "init_weights") if has_iw_A else np.zeros((1, 1))
                W_B = _get(data_B, name, "init_weights") if has_iw_B else np.zeros((1, 1))
                # Initial weights are shared (same seed), so just plot one; still side-by-side for consistency
                _plot_weight_matrix_pair(
                    W_A, W_B,
                    f"Initial Weight Matrix — {name}",
                    f"init_weight_matrix_{_pfx(name)}.png",
                    epoch_dir="epoch_init"
                )


# ══════════════════════════════════════════════════════════════════════════════
# Fingerprint comparison
# ══════════════════════════════════════════════════════════════════════════════

fp_path = os.path.join(SCRIPT_DIR, "fingerprints.npz")
if os.path.exists(fp_path):
    print("\nGenerating fingerprint comparison...")
    fp_data = np.load(fp_path, allow_pickle=True)
    fps     = fp_data["fingerprints"]   # (2, N_IN, N_H)
    W_A, W_B = fps[0], fps[1]

    # Side-by-side + absolute difference
    diff = np.abs(W_A - W_B)
    vmax = max(W_A.max(), W_B.max(), 1e-6)

    fig = plt.figure(figsize=(33, 5))
    gs  = gridspec.GridSpec(1, 6, width_ratios=[3, 1, 3, 1, 3, 1], wspace=0.35)
    fig.suptitle("Fingerprint Comparison — Group A vs Group B", fontsize=14, fontweight="bold")

    for col_offset, (W, label) in enumerate([(W_A, "Group A"), (W_B, "Group B"), (diff, "|A − B|")]):
        ax_heat = fig.add_subplot(gs[col_offset * 2])
        ax_hist = fig.add_subplot(gs[col_offset * 2 + 1])
        ax_heat.set_title(label, fontsize=11)

        cmap = "viridis" if label != "|A − B|" else "Reds"
        im = ax_heat.imshow(
            W, aspect="auto", interpolation="nearest", cmap=cmap,
            vmin=0, vmax=(vmax if label != "|A − B|" else diff.max() or 1),
            origin="lower",
        )
        ax_heat.set_xlabel("Post-synaptic neuron index")
        ax_heat.set_ylabel("Pre-synaptic neuron index")
        plt.colorbar(im, ax=ax_heat, label="Weight")

        w_flat = W.flatten()
        ax_hist.hist(w_flat[w_flat > 0], bins=60, color="steelblue",
                     edgecolor="none", density=True)
        ax_hist.set_xlabel("Weight value")
        ax_hist.set_ylabel("Density")
        ax_hist.set_title("Distribution\n(non-zero)")
        ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    fp_dir = os.path.join(OUT_DIR, "fingerprints")
    os.makedirs(fp_dir, exist_ok=True)
    fig.savefig(os.path.join(fp_dir, "fingerprint_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Scalar similarity metrics
    a_flat = W_A.flatten().astype(np.float64)
    b_flat = W_B.flatten().astype(np.float64)
    cosine = float(np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-12))
    mae    = float(np.mean(diff))
    rmse   = float(np.sqrt(np.mean(diff ** 2)))
    print(f"  Cosine similarity : {cosine:.6f}")
    print(f"  MAE               : {mae:.6f}")
    print(f"  RMSE              : {rmse:.6f}")
    saved.append(os.path.join(fp_dir, "fingerprint_comparison.png"))
    print(f"  Saved: vizs/fingerprints/fingerprint_comparison.png")


# ── finish ────────────────────────────────────────────────────────────────────

print(f"\n{len(saved)} PNG(s) saved to: {OUT_DIR}")
if not saved:
    print("No recognised keys found in npz files — nothing was plotted.")
