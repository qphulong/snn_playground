import numpy as np
np.random.seed(42)
from brian2 import *
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.spike_encoding import compute_spike_input_current
# from src.utils.weights_utils import gaussian_weight_matrix, l1_normalise_weights
from src.recorder import Recorder

import time
start = time.time()

# ============================================================
# Dataset
# ============================================================
# TODO: set the path
wav_files = [
    "datasets/vox1_cleaned/wav_dev/id10022/id10022_00002/00002.wav",
    "datasets/vox1_cleaned/wav_dev/id10022/id10022_00006/00004.wav",
    "datasets/vox1_cleaned/wav_dev/id10022/id10022_00009/00005.wav",
    "datasets/vox1_cleaned/wav_dev/id10022/id10022_00017/00003.wav"
    
]
print(f"Found {len(wav_files)} wav files")

EPOCHS   = 4
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Hyperparameters
# ============================================================

N_IN = 672   # 96 channels × 7 neurons/channel
# N_H  = 672

DT_SIM = 1 * ms

# -- Input layer (adaptive LIF) --
tau_m       = 40 * ms
tau_a       = 100 * ms
tau_current = 1 * ms
beta        = 1
v_th_in     = 1.0

# -- Hidden layer (adaptive-threshold LIF) --
# tau_h    = 50 * ms
# tau_vth  = 100 * ms
# vth_rest = 0.8
# vth_init = 0.8
# vth_jump = 0.3

# -- STDP --
# taupre      = 20 * ms
# taupost     = 20 * ms
# Apre_delta  =  0.004
# Apost_delta = -0.0048

# -- Synaptic weight bounds --
# wmax = 1.0
# wmin = 0.0

# -- Weight initialisation (Gaussian, toroidal topology) --
# W_INIT_SIGMA     = N_IN / 5
# W_INIT_NOISE_STD = 0.005
# W_INIT_SUM       = 2

# -- Homeostatic normalisation --
# NORM_LIMIT = 2



# ============================================================
# Initialise weight matrices
#
# TO ADD A NEW SYNAPSE GROUP: add a weight matrix here following
# the same pattern and give it a descriptive name.
# ============================================================

# w_ih = gaussian_weight_matrix(N_IN, N_H, W_INIT_SIGMA, W_INIT_NOISE_STD,
#                                W_INIT_SUM, wmin, wmax)
# example for a second synapse group:
# w_ho = np.random.uniform(wmin, wmax, size=(N_H, N_OUT))

# init_weights = {"in->hid": w_ih.copy()}


# ============================================================
# Build Brian2 network  (once — never rebuilt between samples)
# ============================================================

defaultclock.dt = DT_SIM

# ── Input neurons ──────────────────────────────────────────────────────────────
# net.store/restore resets the Brian2 clock to 0 before every sample, so
# I_timed(t, i) always indexes from the beginning of the array.
# Do not call start_scope() or rebuild the network inside the sample loop.
eqs_in = """
dv/dt = (-v - a) / tau_m + I_timed(t, i) / tau_current : 1
da/dt = -a / tau_a : 1
"""
G_in = NeuronGroup(
    N_IN, eqs_in,
    threshold="v > v_th_in",
    reset="v=0; a+=beta",
    refractory=2 * ms,
    method="euler"
)

_dummy_I = np.zeros((1, N_IN), dtype=float)
G_in.namespace["I_timed"] = TimedArray(_dummy_I, dt=DT_SIM)

# ── Hidden neurons ────────────────────────────────────────────────────────────
#
# TO ADD A NEW NEURON GROUP: copy this block, change the name and equations.
# Then register it with the recorder below via recorder.track_group().
#
# eqs_h = f"""
# dv/dt   = -v / tau_h                    : 1
# dvth/dt = -(vth - {vth_rest}) / tau_vth : 1
# """
# G_h = NeuronGroup(
#     N_H, eqs_h,
#     threshold="v > vth",
#     reset=f"v=0; vth=vth+{vth_jump};",
#     refractory=2 * ms,
#     method="euler"
# )

# ── STDP synapses: input → hidden ─────────────────────────────────────────────
#
# TO ADD A NEW SYNAPSE GROUP: copy this block, change the name and connect
# to the right groups. Then register it with the recorder below.
#
# stdp_model = """
# w          : 1
# dapre/dt   = -apre  / taupre  : 1 (event-driven)
# dapost/dt  = -apost / taupost : 1 (event-driven)
# """
# on_pre  = "v_post += w\napre += Apre_delta\nw = clip(w + apost*(w-wmin), wmin, wmax)"
# on_post = "apost += Apost_delta\nw = clip(w + apre*(wmax-w), wmin, wmax)"
#
# S_ih = Synapses(G_in, G_h, model=stdp_model, on_pre=on_pre, on_post=on_post)
# S_ih.connect()
# src_ih = np.array(S_ih.i)
# tgt_ih = np.array(S_ih.j)

net = Network(G_in)


# ============================================================
# Recorder setup
#
# Register every group and synapse you want to be recordable.
# The recorder reads record_config.yaml to decide what to actually record —
# registering a group here that has no config entry is harmless (nothing recorded).
#
# TO ADD A NEW GROUP:    recorder.track_group("mygroup", G_new)
# TO ADD A NEW SYNAPSE:  recorder.track_synapses("pre->post", S_new, src_new, tgt_new)
#
# That is the only change needed in this file when extending the architecture.
# ============================================================

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "record_and_visualize_config.yaml")
recorder = Recorder(CONFIG_PATH, net)

recorder.track_group("input",  G_in)
# recorder.track_group("hidden", G_h)
# recorder.track_group("output", G_out)   # <-- add new groups here

# recorder.track_synapses("in->hid", S_ih, src_ih, tgt_ih)
# recorder.track_synapses("hid->out", S_ho, src_ho, tgt_ho)   # <-- add new synapses here

recorder.build()   # attaches all Brian2 monitors — call once, after all registrations

# Snapshot the clean initial state (clock=0, v=0, a=0, vth=vth_init,
# not_refractory=True, empty monitors).  net.restore('init') before every
# sample brings the network back to this state, which is the only correct
# way to reset all Brian2 internal state (including refractory bookkeeping)
# without rebuilding the network.
# G_h.vth = vth_init   # default Brian2 init is 0; set before store
net.store('init')


# ============================================================
# Training loop
# ============================================================

for epoch_idx in range(EPOCHS):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch_idx}/{EPOCHS - 1}")
    print(f"{'='*60}")

    recorder.reset_epoch()

    for sample_idx, audio_path in enumerate(wav_files):
        print(f"  [epoch {epoch_idx}, sample {sample_idx}/{len(wav_files)-1}] "
              f"{os.path.relpath(audio_path)}")

        # ── Encode audio ───────────────────────────────────────────────────────
        try:
            I, T = compute_spike_input_current(
                audio_path,
                scale=0.8,
                num_filters=96,
                sustained_per_band=4,
                onset_per_band=2,
                phase_per_band=1,
                sust_spread_min=0.7,
                sust_spread_max=1.3,
            )
        except Exception as e:
            print(f"    Error encoding audio: {e}")
            continue

        duration_s = float(T) * float(DT_SIM)

        # ── Reset to clean state, then inject this sample's data ──────────────
        # restore resets: clock→0, v, a, not_refractory, and all monitor buffers.
        net.restore('init')

        G_in.namespace["I_timed"] = TimedArray(I.T.astype(float), dt=DT_SIM)

        # Weights survive across samples — override the restored initial weights.
        # S_ih.w    = w_ih[src_ih, tgt_ih]
        # S_ih.apre  = 0
        # S_ih.apost = 0
        # TO ADD A NEW SYNAPSE GROUP reset:
        # S_ho.w     = w_ho[src_ho, tgt_ho]
        # S_ho.apre  = 0
        # S_ho.apost = 0

        # Recorder tracks elapsed time in Python; since restore resets the
        # Brian2 clock to 0, spike times in the monitor are always [0, T] ms.
        # Reset _elapsed_ms so start_ms = 0 and the slice arithmetic is correct.
        recorder._elapsed_ms = 0.0

        # ── Record: before ─────────────────────────────────────────────────────
        recorder.before_sample(sample_idx)

        # ── Simulate ───────────────────────────────────────────────────────────
        net.run(T * DT_SIM)

        # ── Extract updated weights ────────────────────────────────────────────
        # w_ih_new = np.zeros((N_IN, N_H))
        # w_ih_new[src_ih, tgt_ih] = np.array(S_ih.w)
        # TO ADD A NEW SYNAPSE GROUP extraction:
        # w_ho_new = np.zeros((N_H, N_OUT))
        # w_ho_new[src_ho, tgt_ho] = np.array(S_ho.w)

        # ── Custom normalisation ───────────────────────────────────────────────
        # spiked_hidden = np.unique(recorder.spikes_this_sample("hidden"))
        # w_ih = l1_normalise_weights(w_ih_new, spiked_hidden, NORM_LIMIT, wmin, wmax)
        # spiked_output = np.unique(recorder.spikes_this_sample("output"))
        # w_ho = normalise_weights(w_ho_new, spiked_output, NORM_LIMIT, wmin, wmax)

        # ── Record: after ──────────────────────────────────────────────────────
        recorder.after_sample(sample_idx, duration_s)

    # ── End of epoch ──────────────────────────────────────────────────────────
    recorder.save_epoch(epoch_idx, save_dir=SAVE_DIR)


# ============================================================
# Done
# ============================================================

print(f"\n{'='*60}")
print(f"Training complete — {time.time() - start:.2f}s")
print(f"{'='*60}")