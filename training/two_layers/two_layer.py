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

MAX_SAMPLES                = cfg.get("max_samples", None)            # None = all files
RECORD_WEIGHTS_EVOLUTION   = cfg.get("weights_evolution", [])        # list of [i, j] pairs
RECORD_MEMBRANE_POTENTIAL  = cfg.get("membrane_potential", [])        # list of hidden neuron indices
RECORD_FINAL_WEIGHTS       = cfg.get("final_weights_matrix", False)
RECORD_MEAN_RATE_INPUT     = cfg.get("mean_firing_rate_input", False)
RECORD_MEAN_RATE_HIDDEN    = cfg.get("mean_firing_rate_hidden", False)

# Derived flags
NEED_IN_SPIKES  = RECORD_MEAN_RATE_INPUT
NEED_H_SPIKES   = RECORD_MEAN_RATE_HIDDEN or bool(RECORD_WEIGHTS_EVOLUTION)
NEED_V_MONITOR  = bool(RECORD_MEMBRANE_POTENTIAL)
NEED_W_SNAPSHOT = bool(RECORD_WEIGHTS_EVOLUTION)

# Input spike monitor always needs full neuron-level detail for per-neuron rates

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

N_IN = 900   # input layer size: 100 channels × 9 neurons each
N_H  = 900   # hidden (STDP) layer size

DT_SIM = 1 * ms

# -- Input layer (adaptive LIF) --
tau_m       = 40 * ms
tau_a       = 20 * ms
tau_current = 1 * ms
beta        = 0.25
v_th_in     = 1.0

# -- Hidden layer (adaptive-threshold LIF) --
tau_h        = 10 * ms
tau_vth      = 10 * ms
tau_elig     = 20 * ms
vth_rest     = 1.0
vth_init     = 1.0
vth_jump     = 0.25

# -- STDP --
taupre      = 20 * ms
taupost     = 20 * ms
Apre_delta  =  0.01
Apost_delta = -0.0105

# -- Synaptic weight bounds --
wmax        = 1.0
wmin        = 0.0
W_INIT_SUM  = 90

# -- Homeostatic normalisation --
NORM_MARGIN = 1.15

# -- Lateral inhibition --
lat_inh = 0.5


# ============================================================
# Initialize weights
# ============================================================

np.random.seed(42)

w_matrix = np.random.uniform(0, 1, size=(N_IN, N_H))
w_matrix = w_matrix / w_matrix.sum(axis=0, keepdims=True) * W_INIT_SUM
w_matrix = np.clip(w_matrix, wmin, wmax)

# Storage for recorded data
records = {}

if RECORD_WEIGHTS_EVOLUTION:
    # Shape: (num_pairs, num_snapshots_total)
    records["weights_evolution_pairs"] = [list(p) for p in RECORD_WEIGHTS_EVOLUTION]
    records["weights_evolution_values"] = [[] for _ in RECORD_WEIGHTS_EVOLUTION]
    records["weights_evolution_times_ms"] = []

if RECORD_MEMBRANE_POTENTIAL:
    records["membrane_potential_neurons"] = RECORD_MEMBRANE_POTENTIAL
    records["membrane_potential_v"] = []    # list of arrays, one per sample
    records["membrane_potential_t"] = []

if RECORD_FINAL_WEIGHTS:
    records["final_weights_matrix"] = None

if RECORD_MEAN_RATE_INPUT:
    records["mean_firing_rate_input_counts"] = np.zeros(N_IN, dtype=np.int64)
    records["mean_firing_rate_input_duration_s"] = 0.0

if RECORD_MEAN_RATE_HIDDEN:
    records["mean_firing_rate_hidden_counts"] = np.zeros(N_H, dtype=np.int64)
    records["mean_firing_rate_hidden_duration_s"] = 0.0


# ============================================================
# Training Loop
# ============================================================

for sample_idx, audio_path in enumerate(wav_files):

    print(f"[{sample_idx}/{len(wav_files)}] {os.path.relpath(audio_path)}")

    # ======================================================
    # Audio → input current
    # ======================================================

    try:
        I, T = compute_spike_input_current(audio_path, scale=1)
    except Exception as e:
        print("Error:", e)
        continue

    duration_s = float(T) * float(DT_SIM)

    # ======================================================
    # Brian2 simulation
    # ======================================================

    start_scope()
    defaultclock.dt = DT_SIM

    I_timed = TimedArray(I.T.astype(float), dt=DT_SIM)

    # ------------------------------------------------------
    # Input neurons
    # ------------------------------------------------------

    eqs_in = """
    dv/dt = (-v - a) / tau_m + I_timed(t,i) / tau_current : 1
    da/dt = -a / tau_a : 1
    """

    G_in = NeuronGroup(
        N_IN,
        eqs_in,
        threshold="v > v_th_in",
        reset="v=0; a+=beta",
        refractory=2*ms,
        method="euler"
    )

    # ------------------------------------------------------
    # Hidden neurons
    # ------------------------------------------------------

    eqs_h = f"""
    dv/dt          = -v / tau_h                           : 1
    dvth/dt        = -(vth - {vth_rest}) / tau_vth        : 1
    deligibility/dt = -eligibility / tau_elig             : 1
    """

    G_h = NeuronGroup(
        N_H,
        eqs_h,
        threshold="v > vth",
        reset=f"v=0; vth=vth+{vth_jump}; eligibility=eligibility+1.0",
        refractory=2*ms,
        method="euler"
    )
    G_h.vth         = vth_init
    G_h.eligibility = 0.0

    # ------------------------------------------------------
    # STDP Synapses
    # ------------------------------------------------------

    stdp_model = """
    w : 1
    dapre/dt  = -apre/taupre  : 1 (event-driven)
    dapost/dt = -apost/taupost : 1 (event-driven)
    """

    on_pre = """
    v_post += w
    apre += Apre_delta
    w = clip(w + apost, wmin, wmax)
    """

    on_post = """
    apost += Apost_delta
    w = clip(w + apre, wmin, wmax)
    """

    S = Synapses(
        G_in,
        G_h,
        model=stdp_model,
        on_pre=on_pre,
        on_post=on_post
    )
    S.connect()

    src = np.array(S.i)
    tgt = np.array(S.j)
    S.w = w_matrix[src, tgt]

    # ------------------------------------------------------
    # Lateral inhibition
    # ------------------------------------------------------

    lat = Synapses(G_h, G_h, on_pre='v_post = clip(v_post - lat_inh, 0, inf)')
    lat.connect(condition='i != j')

    # ------------------------------------------------------
    # Monitors — only instantiate what config requires
    # ------------------------------------------------------

    in_mon = SpikeMonitor(G_in) if NEED_IN_SPIKES else None
    h_mon  = SpikeMonitor(G_h)  if NEED_H_SPIKES  else None

    # Always need spike monitor on hidden layer for homeostatic norm
    if h_mon is None:
        h_mon = SpikeMonitor(G_h)

    v_mon = None
    if NEED_V_MONITOR:
        v_mon = StateMonitor(G_h, "v", record=RECORD_MEMBRANE_POTENTIAL)

    # ------------------------------------------------------
    # Weight snapshot NetworkOperation
    # ------------------------------------------------------

    SNAPSHOT_INTERVAL = 500 * ms

    if NEED_W_SNAPSHOT:
        pairs = RECORD_WEIGHTS_EVOLUTION
        # precompute flat indices into S.w for each pair
        pair_flat_indices = []
        for (pi, pj) in pairs:
            # S.w is indexed by synapse; find the synapse for (pi, pj)
            mask = (src == pi) & (tgt == pj)
            idx  = np.where(mask)[0]
            pair_flat_indices.append(int(idx[0]) if len(idx) > 0 else None)

        snap_times = []

        @network_operation(dt=SNAPSHOT_INTERVAL)
        def record_weights(t):
            snap_times.append(float(t / ms))
            w_arr = np.array(S.w)
            for k, fidx in enumerate(pair_flat_indices):
                val = float(w_arr[fidx]) if fidx is not None else float("nan")
                records["weights_evolution_values"][k].append(val)

    # ------------------------------------------------------
    # Run
    # ------------------------------------------------------

    run(T * DT_SIM)

    # ======================================================
    # Extract updated weights
    # ======================================================

    w_prev = w_matrix.copy()

    w_new = np.zeros((N_IN, N_H))
    w_new[src, tgt] = np.array(S.w)

    vth_final = np.array(G_h.vth)

    # ======================================================
    # Homeostatic normalisation
    # ======================================================

    spiked_neurons = np.unique(np.array(h_mon.i))

    for nrn in spiked_neurons:
        limit = vth_final[nrn] * NORM_MARGIN
        wsum  = w_new[:, nrn].sum()
        if wsum > limit > 0:
            w_new[:, nrn] *= limit / wsum

    w_new    = np.clip(w_new, wmin, wmax)
    w_matrix = w_new

    # ======================================================
    # Accumulate records
    # ======================================================

    if NEED_W_SNAPSHOT:
        # snap_times are per-sample; offset by sample so they stay meaningful
        records["weights_evolution_times_ms"].extend(snap_times)

    if NEED_V_MONITOR and v_mon is not None:
        records["membrane_potential_v"].append(np.array(v_mon.v, dtype=np.float32))
        records["membrane_potential_t"].append(np.array(v_mon.t / ms, dtype=np.float32))

    if RECORD_MEAN_RATE_INPUT and in_mon is not None:
        spike_i = np.array(in_mon.i, dtype=np.int32)
        if len(spike_i) > 0:
            counts = np.bincount(spike_i, minlength=N_IN)
            records["mean_firing_rate_input_counts"] += counts
        records["mean_firing_rate_input_duration_s"] += duration_s

    if RECORD_MEAN_RATE_HIDDEN:
        spike_i = np.array(h_mon.i, dtype=np.int32)
        if len(spike_i) > 0:
            counts = np.bincount(spike_i, minlength=N_H)
            records["mean_firing_rate_hidden_counts"] += counts
        records["mean_firing_rate_hidden_duration_s"] += duration_s


# ============================================================
# Save records
# ============================================================

save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.npz")

arrays = {}

if RECORD_WEIGHTS_EVOLUTION:
    arrays["we_pairs"]    = np.array(records["weights_evolution_pairs"],  dtype=np.int32)
    arrays["we_values"]   = np.array(records["weights_evolution_values"], dtype=np.float32)  # (n_pairs, n_snaps)
    arrays["we_times_ms"] = np.array(records["weights_evolution_times_ms"], dtype=np.float32)

if RECORD_MEMBRANE_POTENTIAL:
    arrays["vmon_neurons"] = np.array(records["membrane_potential_neurons"], dtype=np.int32)
    if records["membrane_potential_v"]:
        # Save all samples; each may have different length T so store as object array
        v_all = np.empty(len(records["membrane_potential_v"]), dtype=object)
        t_all = np.empty(len(records["membrane_potential_t"]), dtype=object)
        for s, (vv, tt) in enumerate(zip(records["membrane_potential_v"], records["membrane_potential_t"])):
            v_all[s] = vv   # (n_neurons, T_s)
            t_all[s] = tt   # (T_s,)
        arrays["vmon_v_all"] = v_all
        arrays["vmon_t_all"] = t_all
        arrays["vmon_n_samples"] = np.int32(len(records["membrane_potential_v"]))

if RECORD_FINAL_WEIGHTS:
    arrays["final_weights_matrix"] = w_matrix.astype(np.float32)

if RECORD_MEAN_RATE_INPUT and records.get("mean_firing_rate_input_counts") is not None:
    dur = records["mean_firing_rate_input_duration_s"]
    rates = records["mean_firing_rate_input_counts"] / dur if dur > 0 else np.zeros(N_IN)
    arrays["mean_firing_rate_input"] = rates.astype(np.float32)  # (N_IN,)

if RECORD_MEAN_RATE_HIDDEN and records.get("mean_firing_rate_hidden_counts") is not None:
    dur = records["mean_firing_rate_hidden_duration_s"]
    rates = records["mean_firing_rate_hidden_counts"] / dur if dur > 0 else np.zeros(N_H)
    arrays["mean_firing_rate_hidden"] = rates.astype(np.float32)  # (N_H,)

np.savez_compressed(save_path, **arrays)

print(f"\nTraining complete")
print(f"Runtime: {time.time() - start:.2f} seconds")
print(f"Saved records to: {save_path}")
print(f"Keys saved: {list(arrays.keys())}")