import numpy as np
np.random.seed(42)
from brian2 import *
import glob
import os
import yaml

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.spike_encoding import compute_spike_input_current

import time
start = time.time()

# ============================================================
# Load recording config
# ============================================================

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "record_config.yaml")

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

EPOCHS                       = cfg.get("epochs", 1)
RECORD_WEIGHTS_EVOLUTION     = cfg.get("weights_evolution", [])
RECORD_VMON_INPUT_RAW        = cfg.get("membrane_potential_input", [])
RECORD_VMON_HIDDEN_RAW       = cfg.get("membrane_potential_hidden", [])
RECORD_SPIKE_RASTER_INPUT    = cfg.get("spike_raster_input", False)
RECORD_SPIKE_RASTER_HIDDEN   = cfg.get("spike_raster_hidden", False)
RECORD_FINAL_WEIGHTS         = cfg.get("final_weights_matrix", False)
RECORD_MEAN_RATE_INPUT       = cfg.get("mean_firing_rate_input", False)
RECORD_MEAN_RATE_HIDDEN      = cfg.get("mean_firing_rate_hidden", False)
TOP_K_WEIGHTS_VIZ            = cfg.get("top_k_weights_viz", 50)


def _parse_vmon_entries(raw):
    """Return (neuron_indices: list[int], windows: dict[int -> [a,b] or None])."""
    indices = []
    windows = {}
    for entry in (raw or []):
        if isinstance(entry, (int, float)):
            nid = int(entry)
            indices.append(nid)
            windows[nid] = None
        else:
            nid = int(entry[0])
            win = entry[1] if len(entry) > 1 else None
            indices.append(nid)
            windows[nid] = win
    return indices, windows

VMON_IN_INDICES,  VMON_IN_WINDOWS  = _parse_vmon_entries(RECORD_VMON_INPUT_RAW)
VMON_HID_INDICES, VMON_HID_WINDOWS = _parse_vmon_entries(RECORD_VMON_HIDDEN_RAW)

# ── Derived monitor flags ──────────────────────────────────────────────────────
NEED_IN_SPIKE_MON  = RECORD_MEAN_RATE_INPUT  or RECORD_SPIKE_RASTER_INPUT
NEED_HID_SPIKE_MON = RECORD_MEAN_RATE_HIDDEN or RECORD_SPIKE_RASTER_HIDDEN or bool(RECORD_WEIGHTS_EVOLUTION)
NEED_IN_V_MON      = bool(VMON_IN_INDICES)
NEED_HID_V_MON     = bool(VMON_HID_INDICES)
NEED_W_SNAPSHOT    = bool(RECORD_WEIGHTS_EVOLUTION)

# ============================================================
# Find dataset
# ============================================================

wav_files = sorted(glob.glob("datasets/vox1_small_test/**/*.wav", recursive=True))
print(f"Found {len(wav_files)} files in datasets/vox1_small_test")


# ============================================================
# Network Hyperparameters
# ============================================================

N_IN = 700
N_H  = 700

DT_SIM = 1 * ms

# -- Input layer (adaptive LIF) --
tau_m       = 40 * ms
tau_a       = 20 * ms
tau_current = 1 * ms
beta        = 0.25
v_th_in     = 1.0

# -- Hidden layer (adaptive-threshold LIF) --
tau_h        = 50 * ms
tau_vth      = 100 * ms
tau_elig     = 20 * ms
vth_rest     = 0.8
vth_init     = 0.8
vth_jump     = 1

# -- STDP --
taupre      = 20 * ms
taupost     = 20 * ms
Apre_delta  =  0.005
Apost_delta = -0.0055

# -- Synaptic weight bounds --
wmax       = 1.0
wmin       = 0.0
W_INIT_SUM = 4

# -- Weight initialization --
# (old uniform init:)
# w_matrix = np.random.uniform(0, 1, size=(N_IN, N_H))
# w_matrix = w_matrix / w_matrix.sum(axis=0, keepdims=True) * W_INIT_SUM
# w_matrix = np.clip(w_matrix, wmin, wmax)
W_INIT_SIGMA     = N_IN / 5   # Gaussian bell width
W_INIT_NOISE_STD = 0.005      # symmetry-breaking noise

# -- Homeostatic normalisation --
NORM_LIMIT = 4               # column weight-sum cap after each sample

# -- Lateral inhibition --
lat_inh = 1


# ============================================================
# Initialize weights
# ============================================================

i = np.arange(N_IN).reshape(-1, 1)   # input index
j = np.arange(N_H).reshape(1, -1)    # hidden index

# Circular distance (key: enables toroidal topology)
dist = np.abs(i - j)
dist = np.minimum(dist, N_IN - dist)

# Gaussian centered at j for each column
w_matrix = np.exp(-(dist ** 2) / (2 * W_INIT_SIGMA ** 2))

# Add small symmetry-breaking noise
w_matrix += np.random.normal(0, W_INIT_NOISE_STD, size=w_matrix.shape)
w_matrix = np.clip(w_matrix, 0, None)

# Normalize each column to W_INIT_SUM
w_matrix = w_matrix / w_matrix.sum(axis=0, keepdims=True) * W_INIT_SUM
w_matrix = np.clip(w_matrix, wmin, wmax)

# ── Record storage ─────────────────────────────────────────────────────────────
records = {}

if RECORD_WEIGHTS_EVOLUTION:
    records["we_pairs"]          = [list(p) for p in RECORD_WEIGHTS_EVOLUTION]
    records["we_values"]         = [[] for _ in RECORD_WEIGHTS_EVOLUTION]  # global timeline
    records["we_times_ms"]       = []
    # Per-sample: list[sample] of (n_pairs, n_snaps_this_sample) and times
    records["we_sample_values"]  = []   # list[sample] of array (n_pairs, n_snaps)
    records["we_sample_times"]   = []   # list[sample] of array (n_snaps,)

if NEED_IN_V_MON:
    records["vmon_in_indices"]  = VMON_IN_INDICES
    records["vmon_in_windows"]  = VMON_IN_WINDOWS
    records["vmon_in_v"]        = []
    records["vmon_in_t"]        = []

if NEED_HID_V_MON:
    records["vmon_hid_indices"]  = VMON_HID_INDICES
    records["vmon_hid_windows"]  = VMON_HID_WINDOWS
    records["vmon_hid_v"]        = []
    records["vmon_hid_vth"]      = []
    records["vmon_hid_t"]        = []

if RECORD_SPIKE_RASTER_INPUT:
    records["raster_in_i"]  = []
    records["raster_in_t"]  = []

if RECORD_SPIKE_RASTER_HIDDEN:
    records["raster_hid_i"] = []
    records["raster_hid_t"] = []

if RECORD_MEAN_RATE_INPUT:
    records["mfr_in_counts"]        = np.zeros(N_IN, dtype=np.int64)  # global
    records["mfr_in_dur_s"]         = 0.0
    records["mfr_in_sample_counts"] = []   # list[sample] of (N_IN,) int arrays
    records["mfr_in_sample_dur_s"]  = []   # list[sample] of float

if RECORD_MEAN_RATE_HIDDEN:
    records["mfr_hid_counts"]        = np.zeros(N_H,  dtype=np.int64)  # global
    records["mfr_hid_dur_s"]         = 0.0
    records["mfr_hid_sample_counts"] = []
    records["mfr_hid_sample_dur_s"]  = []

# Per-sample weight matrix snapshots (taken at end of each sample, after normalisation)
if RECORD_FINAL_WEIGHTS:
    records["weight_matrix_per_sample"] = []   # list[sample] of (N_IN, N_H) float32


# ============================================================
# Init weight matrix recording
# ============================================================

if RECORD_FINAL_WEIGHTS:
    records["init_weight_matrix"] = w_matrix.astype(np.float32)


# ============================================================
# Training Loop
# ============================================================

for epoch_idx in range(EPOCHS):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch_idx}/{EPOCHS - 1}")
    print(f"{'='*60}")

    # Reset per-epoch records
    epoch_records = {}

    if RECORD_WEIGHTS_EVOLUTION:
        epoch_records["we_pairs"]          = [list(p) for p in RECORD_WEIGHTS_EVOLUTION]
        epoch_records["we_values"]         = [[] for _ in RECORD_WEIGHTS_EVOLUTION]
        epoch_records["we_times_ms"]       = []
        epoch_records["we_sample_values"]  = []
        epoch_records["we_sample_times"]   = []

    if NEED_IN_V_MON:
        epoch_records["vmon_in_indices"]  = VMON_IN_INDICES
        epoch_records["vmon_in_windows"]  = VMON_IN_WINDOWS
        epoch_records["vmon_in_v"]        = []
        epoch_records["vmon_in_t"]        = []

    if NEED_HID_V_MON:
        epoch_records["vmon_hid_indices"]  = VMON_HID_INDICES
        epoch_records["vmon_hid_windows"]  = VMON_HID_WINDOWS
        epoch_records["vmon_hid_v"]        = []
        epoch_records["vmon_hid_vth"]      = []
        epoch_records["vmon_hid_t"]        = []

    if RECORD_SPIKE_RASTER_INPUT:
        epoch_records["raster_in_i"]  = []
        epoch_records["raster_in_t"]  = []

    if RECORD_SPIKE_RASTER_HIDDEN:
        epoch_records["raster_hid_i"] = []
        epoch_records["raster_hid_t"] = []

    if RECORD_MEAN_RATE_INPUT:
        epoch_records["mfr_in_counts"]        = np.zeros(N_IN, dtype=np.int64)
        epoch_records["mfr_in_dur_s"]         = 0.0
        epoch_records["mfr_in_sample_counts"] = []
        epoch_records["mfr_in_sample_dur_s"]  = []

    if RECORD_MEAN_RATE_HIDDEN:
        epoch_records["mfr_hid_counts"]        = np.zeros(N_H, dtype=np.int64)
        epoch_records["mfr_hid_dur_s"]         = 0.0
        epoch_records["mfr_hid_sample_counts"] = []
        epoch_records["mfr_hid_sample_dur_s"]  = []

    if RECORD_FINAL_WEIGHTS:
        epoch_records["weight_matrix_per_sample"] = []

    for sample_idx, audio_path in enumerate(wav_files):
        print(f"[epoch {epoch_idx}/{EPOCHS - 1}, sample {sample_idx}/{len(wav_files)-1}] {os.path.relpath(audio_path)}")

        try:
            I, T = compute_spike_input_current(
                audio_path,
                scale=1,
                sustained_per_band=4,
                onset_per_band=2,
                phase_per_band=1,
                sust_spread_min=0.7,
                sust_spread_max=1.3
                )
        except Exception as e:
            print("Error:", e)
            continue

        duration_s = float(T) * float(DT_SIM)

        # ── Brian2 setup ───────────────────────────────────────────────────────────

        start_scope()
        defaultclock.dt = DT_SIM

        I_timed = TimedArray(I.T.astype(float), dt=DT_SIM)

        # Input neurons
        eqs_in = """
        dv/dt = (-v - a) / tau_m + I_timed(t,i) / tau_current : 1
        da/dt = -a / tau_a : 1
        """
        G_in = NeuronGroup(
            N_IN, eqs_in,
            threshold="v > v_th_in",
            reset="v=0; a+=beta",
            refractory=2*ms,
            method="euler"
        )

        # Hidden neurons
        eqs_h = f"""
        dv/dt           = -v / tau_h                          : 1
        dvth/dt         = -(vth - {vth_rest}) / tau_vth       : 1
        deligibility/dt = -eligibility / tau_elig             : 1
        """
        G_h = NeuronGroup(
            N_H, eqs_h,
            threshold="v > vth",
            reset=f"v=0; vth=vth+{vth_jump}; eligibility=eligibility+1.0",
            refractory=2*ms,
            method="euler"
        )
        G_h.vth         = vth_init
        G_h.eligibility = 0.0

        # STDP Synapses
        stdp_model = """
        w : 1
        dapre/dt  = -apre/taupre   : 1 (event-driven)
        dapost/dt = -apost/taupost : 1 (event-driven)
        """
        on_pre  = "v_post += w\napre += Apre_delta\nw = clip(w + apost, wmin, wmax)"
        on_post = "apost += Apost_delta\nw = clip(w + apre, wmin, wmax)"

        S = Synapses(G_in, G_h, model=stdp_model, on_pre=on_pre, on_post=on_post)
        S.connect()
        src = np.array(S.i)
        tgt = np.array(S.j)
        S.w = w_matrix[src, tgt]

        # Lateral inhibition
        lat = Synapses(G_h, G_h, on_pre='v_post = clip(v_post * 0.8, 0, inf)')
        lat.connect(condition='i != j')

        # ── Monitors ───────────────────────────────────────────────────────────────

        in_spike_mon  = SpikeMonitor(G_in) if NEED_IN_SPIKE_MON else None
        hid_spike_mon = SpikeMonitor(G_h)   # always created

        in_v_mon  = StateMonitor(G_in, "v",          record=VMON_IN_INDICES)  if NEED_IN_V_MON  else None
        hid_v_mon = StateMonitor(G_h,  ["v", "vth"], record=VMON_HID_INDICES) if NEED_HID_V_MON else None

        # ── Weight snapshot ────────────────────────────────────────────────────────

        SNAPSHOT_INTERVAL = 500 * ms

        if NEED_W_SNAPSHOT:
            we_pairs = RECORD_WEIGHTS_EVOLUTION
            pair_flat_idx = []
            for (pi, pj) in we_pairs:
                mask = (src == pi) & (tgt == pj)
                idx  = np.where(mask)[0]
                pair_flat_idx.append(int(idx[0]) if len(idx) > 0 else None)

            # Per-sample accumulators (reset each sample)
            _sample_snap_times  = []
            _sample_snap_values = [[] for _ in we_pairs]   # list per pair

            @network_operation(dt=SNAPSHOT_INTERVAL)
            def record_weights(t):
                t_ms = float(t / ms)
                _sample_snap_times.append(t_ms)
                epoch_records["we_times_ms"].append(t_ms)
                w_arr = np.array(S.w)
                for k, fidx in enumerate(pair_flat_idx):
                    val = float(w_arr[fidx]) if fidx is not None else float("nan")
                    epoch_records["we_values"][k].append(val)
                    _sample_snap_values[k].append(val)

        # ── Run ────────────────────────────────────────────────────────────────────

        run(T * DT_SIM)

        # ── Extract weights ────────────────────────────────────────────────────────

        w_new = np.zeros((N_IN, N_H))
        w_new[src, tgt] = np.array(S.w)
        vth_final = np.array(G_h.vth)

        # Homeostatic normalisation
        # (old adaptive threshold version:)
        # limit = vth_final[nrn] * NORM_MARGIN
        spiked_neurons = np.unique(np.array(hid_spike_mon.i))
        for nrn in spiked_neurons:
            limit = NORM_LIMIT
            wsum  = w_new[:, nrn].sum()
            if wsum > limit > 0:
                w_new[:, nrn] *= limit / wsum

        w_new    = np.clip(w_new, wmin, wmax)
        w_matrix = w_new

        # ── Accumulate records ─────────────────────────────────────────────────────

        if NEED_W_SNAPSHOT:
            # Store per-sample snapshot arrays
            epoch_records["we_sample_times"].append(
                np.array(_sample_snap_times, dtype=np.float32)
            )
            epoch_records["we_sample_values"].append(
                np.array(_sample_snap_values, dtype=np.float32)   # (n_pairs, n_snaps)
            )

        if NEED_IN_V_MON and in_v_mon is not None:
            epoch_records["vmon_in_v"].append(np.array(in_v_mon.v,      dtype=np.float32))
            epoch_records["vmon_in_t"].append(np.array(in_v_mon.t / ms, dtype=np.float32))

        if NEED_HID_V_MON and hid_v_mon is not None:
            epoch_records["vmon_hid_v"].append(np.array(hid_v_mon.v,     dtype=np.float32))
            epoch_records["vmon_hid_vth"].append(np.array(hid_v_mon.vth, dtype=np.float32))
            epoch_records["vmon_hid_t"].append(np.array(hid_v_mon.t / ms, dtype=np.float32))

        if RECORD_SPIKE_RASTER_INPUT and in_spike_mon is not None:
            epoch_records["raster_in_i"].append(np.array(in_spike_mon.i,      dtype=np.int32))
            epoch_records["raster_in_t"].append(np.array(in_spike_mon.t / ms, dtype=np.float32))

        if RECORD_SPIKE_RASTER_HIDDEN:
            epoch_records["raster_hid_i"].append(np.array(hid_spike_mon.i,      dtype=np.int32))
            epoch_records["raster_hid_t"].append(np.array(hid_spike_mon.t / ms, dtype=np.float32))

        if RECORD_MEAN_RATE_INPUT and in_spike_mon is not None:
            spike_i = np.array(in_spike_mon.i, dtype=np.int32)
            sample_counts = np.bincount(spike_i, minlength=N_IN) if len(spike_i) > 0 else np.zeros(N_IN, dtype=np.int64)
            epoch_records["mfr_in_counts"] += sample_counts
            epoch_records["mfr_in_dur_s"]  += duration_s
            epoch_records["mfr_in_sample_counts"].append(sample_counts.astype(np.int32))
            epoch_records["mfr_in_sample_dur_s"].append(duration_s)

        if RECORD_MEAN_RATE_HIDDEN:
            spike_i = np.array(hid_spike_mon.i, dtype=np.int32)
            sample_counts = np.bincount(spike_i, minlength=N_H) if len(spike_i) > 0 else np.zeros(N_H, dtype=np.int64)
            epoch_records["mfr_hid_counts"] += sample_counts
            epoch_records["mfr_hid_dur_s"]  += duration_s
            epoch_records["mfr_hid_sample_counts"].append(sample_counts.astype(np.int32))
            epoch_records["mfr_hid_sample_dur_s"].append(duration_s)

        # Per-sample weight matrix snapshot (taken after normalisation, so it's the
        # state the network carries into the *next* sample — i.e. what was learned so far)
        if RECORD_FINAL_WEIGHTS:
            epoch_records["weight_matrix_per_sample"].append(w_matrix.astype(np.float32))

    # ── End of sample loop for this epoch ──────────────────────────────────────────

    # Save epoch records
    print(f"\nSaving epoch {epoch_idx} records...")
    epoch_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"history_epoch_{epoch_idx}.npz")
    epoch_arrays = {}

    def _pack_ragged(list_of_arrays):
        """Pack a list of arrays with potentially different lengths into an object array."""
        out = np.empty(len(list_of_arrays), dtype=object)
        for i, a in enumerate(list_of_arrays):
            out[i] = a
        return out

    if RECORD_WEIGHTS_EVOLUTION:
        epoch_arrays["we_pairs"]          = np.array(epoch_records["we_pairs"],    dtype=np.int32)
        epoch_arrays["we_values"]         = np.array(epoch_records["we_values"],   dtype=np.float32)
        epoch_arrays["we_times_ms"]       = np.array(epoch_records["we_times_ms"], dtype=np.float32)
        epoch_arrays["we_sample_values"]  = _pack_ragged(epoch_records["we_sample_values"])
        epoch_arrays["we_sample_times"]   = _pack_ragged(epoch_records["we_sample_times"])
        epoch_arrays["we_n_samples"]      = np.int32(len(epoch_records["we_sample_values"]))

    if NEED_IN_V_MON and epoch_records["vmon_in_v"]:
        epoch_arrays["vmon_in_indices"]   = np.array(VMON_IN_INDICES,  dtype=np.int32)
        epoch_arrays["vmon_in_v_all"]     = _pack_ragged(epoch_records["vmon_in_v"])
        epoch_arrays["vmon_in_t_all"]     = _pack_ragged(epoch_records["vmon_in_t"])
        epoch_arrays["vmon_in_n_samples"] = np.int32(len(epoch_records["vmon_in_v"]))
        win_rows = []
        for nid in VMON_IN_INDICES:
            w = VMON_IN_WINDOWS.get(nid)
            win_rows.append([nid, float(w[0]), float(w[1])] if w else [nid, -1.0, -1.0])
        epoch_arrays["vmon_in_windows"] = np.array(win_rows, dtype=np.float32)

    if NEED_HID_V_MON and epoch_records["vmon_hid_v"]:
        epoch_arrays["vmon_hid_indices"]   = np.array(VMON_HID_INDICES,  dtype=np.int32)
        epoch_arrays["vmon_hid_v_all"]     = _pack_ragged(epoch_records["vmon_hid_v"])
        epoch_arrays["vmon_hid_vth_all"]   = _pack_ragged(epoch_records["vmon_hid_vth"])
        epoch_arrays["vmon_hid_t_all"]     = _pack_ragged(epoch_records["vmon_hid_t"])
        epoch_arrays["vmon_hid_n_samples"] = np.int32(len(epoch_records["vmon_hid_v"]))
        win_rows = []
        for nid in VMON_HID_INDICES:
            w = VMON_HID_WINDOWS.get(nid)
            win_rows.append([nid, float(w[0]), float(w[1])] if w else [nid, -1.0, -1.0])
        epoch_arrays["vmon_hid_windows"] = np.array(win_rows, dtype=np.float32)

    if RECORD_SPIKE_RASTER_INPUT and epoch_records["raster_in_i"]:
        epoch_arrays["raster_in_i"]         = _pack_ragged(epoch_records["raster_in_i"])
        epoch_arrays["raster_in_t"]         = _pack_ragged(epoch_records["raster_in_t"])
        epoch_arrays["raster_in_n_samples"] = np.int32(len(epoch_records["raster_in_i"]))
        epoch_arrays["raster_in_n_neurons"] = np.int32(N_IN)

    if RECORD_SPIKE_RASTER_HIDDEN and epoch_records["raster_hid_i"]:
        epoch_arrays["raster_hid_i"]         = _pack_ragged(epoch_records["raster_hid_i"])
        epoch_arrays["raster_hid_t"]         = _pack_ragged(epoch_records["raster_hid_t"])
        epoch_arrays["raster_hid_n_samples"] = np.int32(len(epoch_records["raster_hid_i"]))
        epoch_arrays["raster_hid_n_neurons"] = np.int32(N_H)

    if RECORD_FINAL_WEIGHTS:
        epoch_arrays["final_weights_matrix"] = w_matrix.astype(np.float32)
        epoch_arrays["weight_matrix_per_sample"]   = _pack_ragged(epoch_records["weight_matrix_per_sample"])
        epoch_arrays["weight_matrix_n_samples"]    = np.int32(len(epoch_records["weight_matrix_per_sample"]))

    if RECORD_MEAN_RATE_INPUT and epoch_records.get("mfr_in_counts") is not None:
        dur = epoch_records["mfr_in_dur_s"]
        epoch_arrays["mean_firing_rate_input"] = (
            epoch_records["mfr_in_counts"] / dur if dur > 0 else np.zeros(N_IN)
        ).astype(np.float32)
        epoch_arrays["mfr_in_sample_counts"] = _pack_ragged(epoch_records["mfr_in_sample_counts"])
        epoch_arrays["mfr_in_sample_dur_s"]  = np.array(epoch_records["mfr_in_sample_dur_s"], dtype=np.float32)
        epoch_arrays["mfr_in_n_samples"]     = np.int32(len(epoch_records["mfr_in_sample_dur_s"]))

    if RECORD_MEAN_RATE_HIDDEN and epoch_records.get("mfr_hid_counts") is not None:
        dur = epoch_records["mfr_hid_dur_s"]
        epoch_arrays["mean_firing_rate_hidden"] = (
            epoch_records["mfr_hid_counts"] / dur if dur > 0 else np.zeros(N_H)
        ).astype(np.float32)
        epoch_arrays["mfr_hid_sample_counts"] = _pack_ragged(epoch_records["mfr_hid_sample_counts"])
        epoch_arrays["mfr_hid_sample_dur_s"]  = np.array(epoch_records["mfr_hid_sample_dur_s"], dtype=np.float32)
        epoch_arrays["mfr_hid_n_samples"]     = np.int32(len(epoch_records["mfr_hid_sample_dur_s"]))

    epoch_arrays["v_th_in"] = np.float32(v_th_in)
    np.savez_compressed(epoch_save_path, **epoch_arrays)
    print(f"Saved epoch {epoch_idx} to: {epoch_save_path}")


# ============================================================
# Save initial weights
# ============================================================

init_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history_init.npz")
if RECORD_FINAL_WEIGHTS and "init_weight_matrix" in records:
    np.savez_compressed(init_save_path, init_weight_matrix=records["init_weight_matrix"])
    print(f"Saved init weights to: {init_save_path}")


# ============================================================
# Post-training: Log & visualize top-k weights
# ============================================================

print(f"\n{'='*60}")
print(f"Training Complete!")
print(f"{'='*60}")

if RECORD_FINAL_WEIGHTS:
    W_final = w_matrix
    w_flat  = W_final.flatten()
    top_k   = TOP_K_WEIGHTS_VIZ

    # Sort by descending weight value
    top_k_indices = np.argsort(-w_flat)[:top_k]
    top_k_values  = w_flat[top_k_indices]

    viz_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vizs")
    os.makedirs(viz_dir, exist_ok=True)
    log_path = os.path.join(viz_dir, "final_top_k_weights.log")

    with open(log_path, "w") as flog:
        flog.write(f"Top {top_k} weights after training (sorted descending)\n")
        flog.write(f"{'Rank':>5}  {'src (in)':>10}  {'dst (hid)':>10}  {'weight':>12}\n")
        flog.write("-" * 45 + "\n")
        for rank, (flat_idx, val) in enumerate(zip(top_k_indices, top_k_values), 1):
            i_idx = int(flat_idx) // N_H
            j_idx = int(flat_idx) % N_H
            flog.write(f"{rank:>5}  {i_idx:>10}  {j_idx:>10}  {val:>12.6f}\n")

    print(f"\nTop {top_k} Weights (sorted descending):")
    print(f"{'Rank':>5}  {'src (in)':>10}  {'dst (hid)':>10}  {'weight':>12}")
    print("-" * 45)
    for rank, (flat_idx, val) in enumerate(zip(top_k_indices, top_k_values), 1):
        i_idx = int(flat_idx) // N_H
        j_idx = int(flat_idx) % N_H
        print(f"{rank:>5}  {i_idx:>10}  {j_idx:>10}  {val:>12.6f}")

    print(f"\nSaved top-k log to: {log_path}")


print(f"\nRuntime: {time.time() - start:.2f} seconds")