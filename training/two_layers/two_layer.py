import numpy as np
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

MAX_SAMPLES                  = cfg.get("max_samples", None)
RECORD_WEIGHTS_EVOLUTION     = cfg.get("weights_evolution", [])
RECORD_VMON_INPUT_RAW        = cfg.get("membrane_potential_input", [])
RECORD_VMON_HIDDEN_RAW       = cfg.get("membrane_potential_hidden", [])
RECORD_SPIKE_RASTER_INPUT    = cfg.get("spike_raster_input", False)
RECORD_SPIKE_RASTER_HIDDEN   = cfg.get("spike_raster_hidden", False)
RECORD_FINAL_WEIGHTS         = cfg.get("final_weights_matrix", False)
RECORD_MEAN_RATE_INPUT       = cfg.get("mean_firing_rate_input", False)
RECORD_MEAN_RATE_HIDDEN      = cfg.get("mean_firing_rate_hidden", False)


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

wav_files = sorted(glob.glob("datasets/vox1_small/**/*.wav", recursive=True))
print("Total files:", len(wav_files))

if MAX_SAMPLES is not None:
    wav_files = wav_files[:MAX_SAMPLES]
print("Training on:", len(wav_files))


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
tau_h        = 25 * ms
tau_vth      = 50 * ms
tau_elig     = 20 * ms
vth_rest     = 0.05
vth_init     = 0.05
vth_jump     = 2

# -- STDP --
taupre      = 20 * ms
taupost     = 20 * ms
Apre_delta  =  0.01
Apost_delta = -0.0105

# -- Synaptic weight bounds --
wmax       = 1.0
wmin       = 0.0
W_INIT_SUM = 7

# -- Homeostatic normalisation --
NORM_MARGIN = 1.15

# -- Lateral inhibition --
lat_inh = 1


# ============================================================
# Initialize weights
# ============================================================

w_matrix = np.random.uniform(0, 1, size=(N_IN, N_H))
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
# Training Loop
# ============================================================

for sample_idx, audio_path in enumerate(wav_files):

    print(f"[{sample_idx}/{len(wav_files)}] {os.path.relpath(audio_path)}")

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
    lat = Synapses(G_h, G_h, on_pre='v_post = clip(v_post - lat_inh, 0, inf)')
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
            records["we_times_ms"].append(t_ms)
            w_arr = np.array(S.w)
            for k, fidx in enumerate(pair_flat_idx):
                val = float(w_arr[fidx]) if fidx is not None else float("nan")
                records["we_values"][k].append(val)
                _sample_snap_values[k].append(val)

    # ── Run ────────────────────────────────────────────────────────────────────

    run(T * DT_SIM)

    # ── Extract weights ────────────────────────────────────────────────────────

    w_new = np.zeros((N_IN, N_H))
    w_new[src, tgt] = np.array(S.w)
    vth_final = np.array(G_h.vth)

    # Homeostatic normalisation
    spiked_neurons = np.unique(np.array(hid_spike_mon.i))
    for nrn in spiked_neurons:
        # limit = vth_final[nrn] * NORM_MARGIN
        limit = 7 #TODO: reconsider this
        wsum  = w_new[:, nrn].sum()
        if wsum > limit > 0:
            w_new[:, nrn] *= limit / wsum

    w_new    = np.clip(w_new, wmin, wmax)
    w_matrix = w_new

    # ── Accumulate records ─────────────────────────────────────────────────────

    if NEED_W_SNAPSHOT:
        # Store per-sample snapshot arrays
        records["we_sample_times"].append(
            np.array(_sample_snap_times, dtype=np.float32)
        )
        records["we_sample_values"].append(
            np.array(_sample_snap_values, dtype=np.float32)   # (n_pairs, n_snaps)
        )

    if NEED_IN_V_MON and in_v_mon is not None:
        records["vmon_in_v"].append(np.array(in_v_mon.v,      dtype=np.float32))
        records["vmon_in_t"].append(np.array(in_v_mon.t / ms, dtype=np.float32))

    if NEED_HID_V_MON and hid_v_mon is not None:
        records["vmon_hid_v"].append(np.array(hid_v_mon.v,     dtype=np.float32))
        records["vmon_hid_vth"].append(np.array(hid_v_mon.vth, dtype=np.float32))
        records["vmon_hid_t"].append(np.array(hid_v_mon.t / ms, dtype=np.float32))

    if RECORD_SPIKE_RASTER_INPUT and in_spike_mon is not None:
        records["raster_in_i"].append(np.array(in_spike_mon.i,      dtype=np.int32))
        records["raster_in_t"].append(np.array(in_spike_mon.t / ms, dtype=np.float32))

    if RECORD_SPIKE_RASTER_HIDDEN:
        records["raster_hid_i"].append(np.array(hid_spike_mon.i,      dtype=np.int32))
        records["raster_hid_t"].append(np.array(hid_spike_mon.t / ms, dtype=np.float32))

    if RECORD_MEAN_RATE_INPUT and in_spike_mon is not None:
        spike_i = np.array(in_spike_mon.i, dtype=np.int32)
        sample_counts = np.bincount(spike_i, minlength=N_IN) if len(spike_i) > 0 else np.zeros(N_IN, dtype=np.int64)
        records["mfr_in_counts"] += sample_counts
        records["mfr_in_dur_s"]  += duration_s
        records["mfr_in_sample_counts"].append(sample_counts.astype(np.int32))
        records["mfr_in_sample_dur_s"].append(duration_s)

    if RECORD_MEAN_RATE_HIDDEN:
        spike_i = np.array(hid_spike_mon.i, dtype=np.int32)
        sample_counts = np.bincount(spike_i, minlength=N_H) if len(spike_i) > 0 else np.zeros(N_H, dtype=np.int64)
        records["mfr_hid_counts"] += sample_counts
        records["mfr_hid_dur_s"]  += duration_s
        records["mfr_hid_sample_counts"].append(sample_counts.astype(np.int32))
        records["mfr_hid_sample_dur_s"].append(duration_s)

    # Per-sample weight matrix snapshot (taken after normalisation, so it's the
    # state the network carries into the *next* sample — i.e. what was learned so far)
    if RECORD_FINAL_WEIGHTS:
        records["weight_matrix_per_sample"].append(w_matrix.astype(np.float32))


# ============================================================
# Save records
# ============================================================

save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.npz")
arrays    = {}

def _pack_ragged(list_of_arrays):
    """Pack a list of arrays with potentially different lengths into an object array."""
    out = np.empty(len(list_of_arrays), dtype=object)
    for i, a in enumerate(list_of_arrays):
        out[i] = a
    return out

if RECORD_WEIGHTS_EVOLUTION:
    arrays["we_pairs"]          = np.array(records["we_pairs"],    dtype=np.int32)
    arrays["we_values"]         = np.array(records["we_values"],   dtype=np.float32)
    arrays["we_times_ms"]       = np.array(records["we_times_ms"], dtype=np.float32)
    arrays["we_sample_values"]  = _pack_ragged(records["we_sample_values"])  # obj[(n_pairs, n_snaps)]
    arrays["we_sample_times"]   = _pack_ragged(records["we_sample_times"])   # obj[(n_snaps,)]
    arrays["we_n_samples"]      = np.int32(len(records["we_sample_values"]))

if NEED_IN_V_MON and records["vmon_in_v"]:
    arrays["vmon_in_indices"]   = np.array(VMON_IN_INDICES,  dtype=np.int32)
    arrays["vmon_in_v_all"]     = _pack_ragged(records["vmon_in_v"])
    arrays["vmon_in_t_all"]     = _pack_ragged(records["vmon_in_t"])
    arrays["vmon_in_n_samples"] = np.int32(len(records["vmon_in_v"]))
    win_rows = []
    for nid in VMON_IN_INDICES:
        w = VMON_IN_WINDOWS.get(nid)
        win_rows.append([nid, float(w[0]), float(w[1])] if w else [nid, -1.0, -1.0])
    arrays["vmon_in_windows"] = np.array(win_rows, dtype=np.float32)

if NEED_HID_V_MON and records["vmon_hid_v"]:
    arrays["vmon_hid_indices"]   = np.array(VMON_HID_INDICES,  dtype=np.int32)
    arrays["vmon_hid_v_all"]     = _pack_ragged(records["vmon_hid_v"])
    arrays["vmon_hid_vth_all"]   = _pack_ragged(records["vmon_hid_vth"])
    arrays["vmon_hid_t_all"]     = _pack_ragged(records["vmon_hid_t"])
    arrays["vmon_hid_n_samples"] = np.int32(len(records["vmon_hid_v"]))
    win_rows = []
    for nid in VMON_HID_INDICES:
        w = VMON_HID_WINDOWS.get(nid)
        win_rows.append([nid, float(w[0]), float(w[1])] if w else [nid, -1.0, -1.0])
    arrays["vmon_hid_windows"] = np.array(win_rows, dtype=np.float32)

if RECORD_SPIKE_RASTER_INPUT and records["raster_in_i"]:
    arrays["raster_in_i"]         = _pack_ragged(records["raster_in_i"])
    arrays["raster_in_t"]         = _pack_ragged(records["raster_in_t"])
    arrays["raster_in_n_samples"] = np.int32(len(records["raster_in_i"]))
    arrays["raster_in_n_neurons"] = np.int32(N_IN)

if RECORD_SPIKE_RASTER_HIDDEN and records["raster_hid_i"]:
    arrays["raster_hid_i"]         = _pack_ragged(records["raster_hid_i"])
    arrays["raster_hid_t"]         = _pack_ragged(records["raster_hid_t"])
    arrays["raster_hid_n_samples"] = np.int32(len(records["raster_hid_i"]))
    arrays["raster_hid_n_neurons"] = np.int32(N_H)

if RECORD_FINAL_WEIGHTS:
    # Whole-dataset: final state after all training
    arrays["final_weights_matrix"] = w_matrix.astype(np.float32)
    # Per-sample: weight state after each sample (shape: n_samples x N_IN x N_H)
    # Stored as object array of 2-D matrices to keep ragged-array packing consistent
    arrays["weight_matrix_per_sample"]   = _pack_ragged(records["weight_matrix_per_sample"])
    arrays["weight_matrix_n_samples"]    = np.int32(len(records["weight_matrix_per_sample"]))

if RECORD_MEAN_RATE_INPUT and records.get("mfr_in_counts") is not None:
    # Whole-dataset aggregate
    dur = records["mfr_in_dur_s"]
    arrays["mean_firing_rate_input"] = (
        records["mfr_in_counts"] / dur if dur > 0 else np.zeros(N_IN)
    ).astype(np.float32)
    # Per-sample rates
    arrays["mfr_in_sample_counts"] = _pack_ragged(records["mfr_in_sample_counts"])
    arrays["mfr_in_sample_dur_s"]  = np.array(records["mfr_in_sample_dur_s"], dtype=np.float32)
    arrays["mfr_in_n_samples"]     = np.int32(len(records["mfr_in_sample_dur_s"]))

if RECORD_MEAN_RATE_HIDDEN and records.get("mfr_hid_counts") is not None:
    dur = records["mfr_hid_dur_s"]
    arrays["mean_firing_rate_hidden"] = (
        records["mfr_hid_counts"] / dur if dur > 0 else np.zeros(N_H)
    ).astype(np.float32)
    arrays["mfr_hid_sample_counts"] = _pack_ragged(records["mfr_hid_sample_counts"])
    arrays["mfr_hid_sample_dur_s"]  = np.array(records["mfr_hid_sample_dur_s"], dtype=np.float32)
    arrays["mfr_hid_n_samples"]     = np.int32(len(records["mfr_hid_sample_dur_s"]))

arrays["v_th_in"] = np.float32(v_th_in)

np.savez_compressed(save_path, **arrays)

print(f"\nTraining complete")
print(f"Runtime: {time.time() - start:.2f} seconds")
print(f"Saved records to: {save_path}")
print(f"Keys saved: {list(arrays.keys())}")