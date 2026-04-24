"""
visualize.py
------------
Reads history_epoch_NNN.npz files produced by train.py and generates
publication-quality PNG plots for every enabled recording type.

Output layout
-------------
  vizs/
    epoch_000/
      sample_000/
        spike_raster_<group>.png
        mean_firing_rate_<group>.png
        membrane_potential_<group>_neuron_<id>.png
        weights_evolution_<src>_to_<dst>_src<s>_dst<d>.png   ← one PNG per pair
        weights_per_neuron_<src>_to_<dst>_dst<id>.png
      sample_001/
        ...
    epoch_001/
      ...

Usage
-----
  python visualize.py                        # uses default paths next to this file
  python visualize.py --arch arch.yaml --cfg record_config.yaml --dir /path/to/run
"""

from __future__ import annotations

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import yaml

# ── colour palette ─────────────────────────────────────────────────────────────
_PALETTE = {
    "input_group": "#4C9BE8",
    "hidden_e":    "#E8834C",
    "hidden_i":    "#6DBE6D",
    "_default":    "#9B59B6",
}
_SYNAPSE_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12",
    "#9B59B6", "#1ABC9C", "#E67E22", "#95A5A6",
]

matplotlib.rcParams.update({
    "figure.dpi":         150,
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "lines.linewidth":    1.2,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})


# ============================================================
# Helpers
# ============================================================

def _color(group_name: str) -> str:
    return _PALETTE.get(group_name, _PALETTE["_default"])


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", s)


def _load_npz(path: str) -> dict:
    raw = np.load(path, allow_pickle=True)
    out = {}
    for k in raw.files:
        v = raw[k]
        if v.ndim == 0:
            v = v.item()
        out[k] = v
    return out


def _epoch_path(run_dir: str, epoch_idx: int) -> str:
    return os.path.join(run_dir, f"history_epoch_{epoch_idx:03d}.npz")


def _out_dir(viz_root: str, epoch_idx: int, sample_idx: int) -> str:
    base = os.path.join(viz_root, f"epoch_{epoch_idx:03d}", f"sample_{sample_idx:03d}")
    os.makedirs(base, exist_ok=True)
    return base


def _savefig(fig: plt.Figure, path: str):
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved -> {path}")


def _get_sample(obj_arr: np.ndarray, sample_idx: int) -> np.ndarray:
    return np.asarray(obj_arr[sample_idx])


def _infer_n_samples(data: dict) -> int:
    for k, v in data.items():
        if k.endswith("_n_samples") and isinstance(v, (int, np.integer)):
            return int(v)
    for k, v in data.items():
        if (
            isinstance(v, np.ndarray)
            and v.dtype == object
            and v.ndim == 1
            and len(v) > 0
        ):
            return len(v)
    return 0


def _thresh(group_name: str, group_cfgs: dict) -> float | None:
    thr_str = group_cfgs.get(group_name, {}).get("threshold", "")
    m = re.search(r"[-+]?\d*\.?\d+", thr_str)
    return float(m.group()) if m else None


def _src_dst(sname: str) -> tuple[str, str]:
    m = re.fullmatch(r"from_(.+)_to_(.+)", sname)
    return (m.group(1), m.group(2)) if m else ("?", "?")


# ============================================================
# Plot functions
# ============================================================

# ── 1. Spike raster ───────────────────────────────────────────────────────────

def plot_spike_raster(
    spk_i:         np.ndarray,
    spk_t_sim_ms:  np.ndarray,   # spike times in simulation ms
    n_neurons:     int,
    group_name:    str,
    audio_dur_s:   float,        # true audio duration  (for x-axis label + limit)
    sim_dur_s:     float,        # actual simulation duration (for rescaling)
    out_path:      str,
):
    audio_dur_ms = audio_dur_s * 1e3
    sim_dur_ms   = sim_dur_s   * 1e3

    # Rescale spike times from simulation-ms to audio-ms so they fill the axis.
    # Each sim timestep represents (audio_dur / sim_dur) of real audio time.
    scale = (audio_dur_ms / sim_dur_ms) if sim_dur_ms > 0 else 1.0
    spk_t_audio_ms = spk_t_sim_ms * scale

    fig, ax = plt.subplots(figsize=(10, 3.5))
    if len(spk_t_audio_ms) > 0:
        ax.scatter(spk_t_audio_ms, spk_i, s=0.8, c=_color(group_name),
                   linewidths=0, rasterized=True)

    ax.set_xlim(0, audio_dur_ms if audio_dur_ms > 0 else 1.0)
    ax.set_ylim(-0.5, n_neurons - 0.5)
    ax.set_xlabel("Time (ms, audio)")
    ax.set_ylabel("Neuron index")
    ax.set_title(
        f"Spike raster — {group_name}  "
        f"({len(spk_t_audio_ms):,} spikes / {audio_dur_ms:.0f} ms)"
    )
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
    fig.tight_layout()
    _savefig(fig, out_path)


# ── 2. Mean firing rate ───────────────────────────────────────────────────────

def plot_mean_firing_rate(
    spk_i:      np.ndarray,
    n_neurons:  int,
    group_name: str,
    duration_s: float,
    out_path:   str,
):
    counts      = np.bincount(spk_i, minlength=n_neurons).astype(float)
    rates       = counts / max(duration_s, 1e-9)
    mean_r      = rates.mean()
    active_frac = (rates > 0).mean() * 100

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(np.arange(n_neurons), rates, width=1.0,
           color=_color(group_name), linewidth=0, alpha=0.85)
    ax.set_xlabel("Neuron index")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_xlim(-0.5, n_neurons - 0.5)
    ax.set_title(
        f"Mean firing rate — {group_name}  |  "
        f"mean={mean_r:.2f} Hz  active={active_frac:.1f}%"
    )
    fig.tight_layout()
    _savefig(fig, out_path)


# ── 3. Membrane potential ─────────────────────────────────────────────────────

def plot_membrane_potential(
    t_ms:       np.ndarray,
    v:          np.ndarray,
    neuron_id:  int,
    group_name: str,
    threshold:  float | None,
    window_ms:  tuple[float, float] | None,
    out_path:   str,
):
    if window_ms is not None:
        t0, t1 = window_ms
        mask = (t_ms >= t0) & (t_ms <= t1)
        t_plot, v_plot = t_ms[mask], v[mask]
    else:
        t_plot, v_plot = t_ms, v

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t_plot, v_plot, color=_color(group_name), lw=0.9)

    if threshold is not None:
        ax.axhline(threshold, color="#E74C3C", lw=0.8, ls="--",
                   label=f"threshold = {threshold:.2f}")
        ax.legend(fontsize=8, frameon=False)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Membrane potential (a.u.)")
    win_str = (f"  [{window_ms[0]:.0f}-{window_ms[1]:.0f} ms]"
               if window_ms else "")
    ax.set_title(
        f"Membrane potential — {group_name} / neuron {neuron_id}{win_str}"
    )
    fig.tight_layout()
    _savefig(fig, out_path)


# ── 4. Weight evolution — one PNG per tracked pair ────────────────────────────

def plot_weight_evolution_pair(
    times_ms:   np.ndarray,   # (n_snaps,)
    values:     np.ndarray,   # (n_snaps,)
    src_id:     int,
    dst_id:     int,
    src_group:  str,
    dst_group:  str,
    sample_idx: int,
    color:      str,
    out_path:   str,
):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times_ms, values, color=color, lw=1.1)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Weight value")
    ax.set_ylim(0, None)
    ax.set_title(
        f"Weight evolution — {src_group}[{src_id}] -> {dst_group}[{dst_id}]  "
        f"[sample {sample_idx}]"
    )
    fig.tight_layout()
    _savefig(fig, out_path)


# ── 5. Weights per neuron — histogram (weight value on x, count on y) ─────────

def plot_weights_per_neuron(
    W_col:     np.ndarray,   # (n_src,) incoming weights to one dst neuron
    dst_id:    int,
    src_group: str,
    dst_group: str,
    out_path:  str,
):
    w_pos = W_col[W_col > 1e-9]
    w_sum = W_col.sum()
    w_max = W_col.max()
    w_nnz = len(w_pos)
    n_src = len(W_col)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    if len(w_pos) > 0:
        ax.hist(w_pos, bins=50, color=_color(src_group), alpha=0.85, edgecolor="none")
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Incoming weight distribution — {src_group} -> {dst_group}[{dst_id}]\n"
        f"sum={w_sum:.3f}  max={w_max:.4f}  nnz={w_nnz}/{n_src}"
    )
    fig.tight_layout()
    _savefig(fig, out_path)


# ============================================================
# Epoch visualizer
# ============================================================

def visualize_epoch(
    epoch_idx:   int,
    data:        dict,
    rec_cfg:     dict,
    arch_cfg:    dict,
    viz_root:    str,
    samples_sel: list[int],
):
    n_samples = _infer_n_samples(data)
    if n_samples == 0:
        print(f"  Epoch {epoch_idx}: no samples found in npz, skipping.")
        return

    sample_indices = (
        [s for s in samples_sel if s < n_samples]
        if samples_sel
        else list(range(n_samples))
    )

    durations  = data.get("sample_durations_s", np.zeros(n_samples))
    group_cfgs = arch_cfg.get("neuron_groups", {})

    for sidx in sample_indices:
        print(f"  epoch {epoch_idx} / sample {sidx}")
        sdir  = _out_dir(viz_root, epoch_idx, sidx)
        dur_s     = float(durations[sidx]) if sidx < len(durations) else 0.0
        sim_durs  = data.get("sample_sim_durations_s", None)
        sim_dur_s = float(sim_durs[sidx]) if (sim_durs is not None and sidx < len(sim_durs)) else dur_s

        # ── spike raster ──────────────────────────────────────────────────────
        for gname in (rec_cfg.get("spike_raster") or []):
            ki = f"spikes_{gname}_i"
            kt = f"spikes_{gname}_t"
            kn = f"spikes_{gname}_n_neurons"
            if ki not in data:
                continue
            plot_spike_raster(
                spk_i        = _get_sample(data[ki], sidx),
                spk_t_sim_ms = _get_sample(data[kt], sidx),
                n_neurons    = int(data.get(kn, 0)),
                group_name   = gname,
                audio_dur_s  = dur_s,
                sim_dur_s    = sim_dur_s,
                out_path     = os.path.join(sdir, f"spike_raster_{_safe_name(gname)}.png"),
            )

        # ── mean firing rate ──────────────────────────────────────────────────
        for gname in (rec_cfg.get("mean_firing_rate") or []):
            ki = f"spikes_{gname}_i"
            kn = f"spikes_{gname}_n_neurons"
            if ki not in data:
                continue
            plot_mean_firing_rate(
                spk_i      = _get_sample(data[ki], sidx),
                n_neurons  = int(data.get(kn, 0)),
                group_name = gname,
                duration_s = dur_s,
                out_path   = os.path.join(sdir, f"mean_firing_rate_{_safe_name(gname)}.png"),
            )

        # ── membrane potential ────────────────────────────────────────────────
        for gname in (rec_cfg.get("membrane_potential") or {}):
            kv = f"vmon_{gname}_v_all"
            kt = f"vmon_{gname}_t_all"
            ki = f"vmon_{gname}_indices"
            kw = f"vmon_{gname}_windows"
            if kv not in data:
                continue

            v_all    = data[kv]
            t_all    = data[kt]
            indices  = np.asarray(data[ki], dtype=int)
            win_rows = np.asarray(data[kw], dtype=float)   # (n_monitored, 3)

            v_sample = _get_sample(v_all, sidx)   # (n_monitored, T)
            t_sample = _get_sample(t_all, sidx)   # (T,)

            for row_k, nid in enumerate(indices):
                if row_k >= v_sample.shape[0]:
                    continue
                win_row   = win_rows[row_k]
                window_ms = (
                    None if win_row[1] < 0
                    else (float(win_row[1]), float(win_row[2]))
                )
                plot_membrane_potential(
                    t_ms       = t_sample,
                    v          = v_sample[row_k],
                    neuron_id  = int(nid),
                    group_name = gname,
                    threshold  = _thresh(gname, group_cfgs),
                    window_ms  = window_ms,
                    out_path   = os.path.join(
                        sdir,
                        f"membrane_potential_{_safe_name(gname)}_neuron_{nid}.png",
                    ),
                )

        # ── weight evolution — one file per tracked pair ──────────────────────
        for sname in (rec_cfg.get("weights_evolution") or {}):
            key_st = f"we_{sname}_sample_times"
            key_sv = f"we_{sname}_sample_vals"
            key_p  = f"we_{sname}_pairs"
            if key_st not in data:
                continue

            times_ms = _get_sample(data[key_st], sidx)      # (n_snaps,)
            vals     = _get_sample(data[key_sv], sidx)      # (n_pairs, n_snaps)
            pairs    = np.asarray(data[key_p], dtype=int)   # (n_pairs, 2)

            if vals.ndim == 1:
                vals = vals.reshape(1, -1)

            src_grp, dst_grp = _src_dst(sname)

            for k in range(len(pairs)):
                src_id = int(pairs[k, 0])
                dst_id = int(pairs[k, 1])
                fname  = (
                    f"weights_evolution"
                    f"_{_safe_name(src_grp)}_to_{_safe_name(dst_grp)}"
                    f"_src{src_id}_dst{dst_id}.png"
                )
                plot_weight_evolution_pair(
                    times_ms   = times_ms,
                    values     = vals[k],
                    src_id     = src_id,
                    dst_id     = dst_id,
                    src_group  = src_grp,
                    dst_group  = dst_grp,
                    sample_idx = sidx,
                    color      = _SYNAPSE_COLORS[k % len(_SYNAPSE_COLORS)],
                    out_path   = os.path.join(sdir, fname),
                )

        # ── weights per neuron ────────────────────────────────────────────────
        for sname, dst_ids in (rec_cfg.get("weights_per_neuron") or {}).items():
            key_ps = f"w_{sname}_per_sample"
            if key_ps not in data:
                print(f"  WARNING: '{key_ps}' not in npz — "
                      f"add '{sname}' to final_weights in record_config.yaml")
                continue

            W_sample         = _get_sample(data[key_ps], sidx)   # (N_src, N_dst)
            src_grp, dst_grp = _src_dst(sname)

            for dst_id in dst_ids:
                dst_id = int(dst_id)
                if dst_id >= W_sample.shape[1]:
                    print(f"  WARNING: dst_id {dst_id} >= N_dst "
                          f"{W_sample.shape[1]} for {sname}, skipping.")
                    continue
                fname = (
                    f"weights_per_neuron"
                    f"_{_safe_name(src_grp)}_to_{_safe_name(dst_grp)}"
                    f"_dst{dst_id}.png"
                )
                plot_weights_per_neuron(
                    W_col     = W_sample[:, dst_id],
                    dst_id    = dst_id,
                    src_group = src_grp,
                    dst_group = dst_grp,
                    out_path  = os.path.join(sdir, fname),
                )


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SNN training visualizer")
    parser.add_argument("--arch", default=None, help="Path to architecture.yaml")
    parser.add_argument("--cfg",  default=None, help="Path to record_config.yaml")
    parser.add_argument("--dir",  default=None, help="Directory containing npz files")
    args = parser.parse_args()

    HERE = os.path.dirname(os.path.abspath(__file__))

    arch_path = args.arch or os.path.join(HERE, "architecture.yaml")
    cfg_path  = args.cfg  or os.path.join(HERE, "record_config.yaml")
    run_dir   = args.dir  or HERE
    viz_root  = os.path.join(run_dir, "vizs")

    with open(arch_path) as f:
        arch_cfg = yaml.safe_load(f)
    with open(cfg_path) as f:
        rec_cfg = yaml.safe_load(f)

    total_epochs = int(rec_cfg.get("epochs", 1))
    epochs_sel   = rec_cfg.get("epochs_to_visualize") or []
    epochs_sel   = [int(e) for e in epochs_sel] if epochs_sel else list(range(total_epochs))

    samples_sel  = rec_cfg.get("samples_to_visualize") or []
    samples_sel  = [int(s) for s in samples_sel] if samples_sel else []

    print(f"Visualizing epochs:  {epochs_sel}")
    print(f"Visualizing samples: {samples_sel if samples_sel else 'all'}")

    for epoch_idx in epochs_sel:
        npz_path = _epoch_path(run_dir, epoch_idx)
        if not os.path.exists(npz_path):
            print(f"Epoch {epoch_idx}: npz not found at {npz_path}, skipping.")
            continue
        print(f"\n{'='*55}\nEpoch {epoch_idx} — {npz_path}\n{'='*55}")
        data = _load_npz(npz_path)
        visualize_epoch(epoch_idx, data, rec_cfg, arch_cfg, viz_root, samples_sel)

    print(f"\nDone. Plots saved under {viz_root}/")


if __name__ == "__main__":
    main()