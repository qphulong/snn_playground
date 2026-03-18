import numpy as np
from brian2 import *
import glob
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.spike_encoding import compute_spike_input_current

import time
start = time.time()
# ============================================================
# Find dataset
# ============================================================
wav_files = sorted(glob.glob("datasets/vox1_small/**/*.wav", recursive=True))

print("Total files:", len(wav_files))

MAX_SAMPLES = 1
wav_files = wav_files[:MAX_SAMPLES]

print("Training on:", len(wav_files))


# ============================================================
# Network Hyperparameters
# ============================================================

N_IN = 900   # input layer size: 100 channels × 9 neurons each
N_H  = 900   # hidden (STDP) layer size

DT_SIM = 1 * ms  # simulation time step

# -- Input layer (adaptive LIF) --
tau_m       = 40 * ms   # membrane time constant
tau_a       = 20 * ms   # adaptation time constant
tau_current = 1 * ms    # unit-conversion factor for input current
beta        = 0.25      # adaptation increment on spike
v_th_in     = 1.0       # spike threshold

# -- Hidden layer (adaptive-threshold LIF) --
tau_h        = 10 * ms   # membrane time constant
tau_vth      = 10 * ms   # adaptive threshold decay
tau_elig     = 20 * ms   # post-synaptic eligibility trace decay
vth_rest     = 1.0       # resting threshold
vth_init     = 1.0       # initial threshold (== resting)
vth_jump     = 0.25      # threshold increment on spike

# -- STDP --
taupre      = 20 * ms    # pre-synaptic trace decay
taupost     = 20 * ms    # post-synaptic trace decay
Apre_delta  =  0.01      # LTP increment
Apost_delta = -0.0105    # LTD increment (slightly larger for weight drift control)

# -- Synaptic weight bounds --
wmax        = 1.0
wmin        = 0.0
w_init_mean = 0.3   # initial weight mean (uniform in [wmin, 2*w_init_mean])

# -- Weight normalisation (reference-style: threshold-relative, spike-gated) --
NORM_MARGIN = 1.15

# -- Winner-suppress-all (lateral inhibition in hidden layer) --
lat_inh = 0.5   # inhibitory voltage kick applied to all non-winning hidden neurons


# ============================================================
# Initialize weights
# ============================================================

np.random.seed(42)

w_matrix = np.random.uniform(wmin, 2*w_init_mean, size=(N_IN, N_H))

history = {
    "mean_w": [],
    "std_w": [],
    "hidden_spikes": [],
    "delta_w": []
}


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


    # ======================================================
    # Brian2 simulation
    # ======================================================

    start_scope()
    defaultclock.dt = DT_SIM

    I_timed = TimedArray(I.T.astype(float), dt=DT_SIM)

    # ------------------------------------------------------
    # Input encoding neurons
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
    # Hidden neurons (adaptive-threshold LIF)
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
    # Winner-suppress-all lateral inhibition (hidden layer)
    # ------------------------------------------------------

    lat = Synapses(G_h, G_h, on_pre='v_post = clip(v_post - lat_inh, 0, inf)')
    lat.connect(condition='i != j')


    # ------------------------------------------------------
    # Spike monitors (full raster recording)
    # ------------------------------------------------------

    in_mon = SpikeMonitor(G_in)
    h_mon  = SpikeMonitor(G_h)

    # ------------------------------------------------------
    # Weight snapshot via NetworkOperation
    # ------------------------------------------------------

    SNAPSHOT_INTERVAL = 500 * ms
    w_snapshots_list = []
    w_snapshot_times_list = []

    @network_operation(dt=SNAPSHOT_INTERVAL)
    def record_weights(t):
        w_snap = np.zeros((N_IN, N_H), dtype=np.float32)
        w_snap[src, tgt] = np.array(S.w, dtype=np.float32)
        w_snapshots_list.append(w_snap.copy())
        w_snapshot_times_list.append(float(t / ms))

    run(T * DT_SIM)


    # ======================================================
    # Extract updated weights and thresholds
    # ======================================================

    w_prev = w_matrix.copy()

    w_new = np.zeros((N_IN, N_H))
    w_new[src, tgt] = np.array(S.w)

    vth_final = np.array(G_h.vth)   # per-neuron adaptive threshold after sample


    # ======================================================
    # Homeostatic normalisation (threshold-relative, spike-gated)
    #
    # Only normalise hidden neurons that fired at least once.
    # Each such neuron's total incoming weight is capped at
    # vth_final[nrn] * NORM_MARGIN, matching the reference script.
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
    # Diagnostics
    # ======================================================

    delta_w = np.mean(np.abs(w_matrix - w_prev))

    history["mean_w"].append(float(w_matrix.mean()))
    history["std_w"].append(float(w_matrix.std()))
    history["hidden_spikes"].append(int(h_mon.num_spikes))
    history["delta_w"].append(float(delta_w))


print("\nTraining complete")

print("Final mean weight:", w_matrix.mean())
print("Final std weight :", w_matrix.std())
print("Mean hidden spikes/sample:", np.mean(history["hidden_spikes"]))

print(f"Runtime: {time.time() - start:.2f} seconds")


# ============================================================
# Save history for visualization
# ============================================================

save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.npz")

np.savez_compressed(
    save_path,
    # Spike rasters
    in_spike_i=np.array(in_mon.i, dtype=np.int32),
    in_spike_t=np.array(in_mon.t / ms, dtype=np.float32),
    h_spike_i=np.array(h_mon.i, dtype=np.int32),
    h_spike_t=np.array(h_mon.t / ms, dtype=np.float32),
    # Final weight matrix
    w_final=w_matrix.astype(np.float32),
    # Weight snapshots over time
    w_snapshots=np.array(w_snapshots_list, dtype=np.float32),
    w_snapshot_times_ms=np.array(w_snapshot_times_list, dtype=np.float32),
    # Metadata
    T_ms=np.float32(float(T)),
    N_IN=np.int32(N_IN),
    N_H=np.int32(N_H),
    # Scalar diagnostics
    mean_w=np.array(history["mean_w"], dtype=np.float32),
    std_w=np.array(history["std_w"], dtype=np.float32),
    hidden_spikes=np.array(history["hidden_spikes"], dtype=np.int32),
    delta_w=np.array(history["delta_w"], dtype=np.float32),
)

print(f"History saved to: {save_path}")