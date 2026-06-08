import numpy as np
np.random.seed(42)
from brian2 import *
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.spike_encoding import compute_spike_input_current
from src.utils.calibration import calibrate_current_to_rate
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
N_H  = 672

DT_SIM = 1 * ms

# -- Input encoding (compute_spike_input_current) --
ENC_NUM_FILTERS      = 96
ENC_SUSTAINED        = 4     # sustained neurons / band
ENC_ONSET            = 2     # onset neurons / band
ENC_PHASE            = 1     # phase neurons / band
ENC_SUST_SPREAD_MIN  = 0.7
ENC_SUST_SPREAD_MAX  = 1.3
ENC_SUST_GAIN        = 1.0
ENC_ONSET_GAIN       = 2.0
ENC_PHASE_GAIN       = 1.0
ENC_SCALE_SEED       = 1.0   # seed for auto-calibration (final level set by target rate)
ENC_CHANNEL_FLOOR    = 0.95   # across-channel "activity distance": 1.0 = equal, <1 = louder channels dominate

# Auto-calibration: hold input firing RATE constant across files (Hz / neuron).
# This is the main STDP-health knob — tune it.
TARGET_INPUT_RATE_HZ = 10.0

# -- Input layer (adaptive LIF) --
tau_m       = 40 * ms
tau_a       = 100 * ms
tau_current = 1 * ms
beta        = 1
v_th_in     = 1.0

# -- Hidden layer (adaptive-threshold LIF + slow AHP) --
tau_h    = 50  * ms
tau_vth  = 100 * ms
vth_rest = 0.8
vth_init = 0.8
vth_jump = 0.3
tau_a_h  = 500 * ms
beta_h   = 0.1

EXC_SUM_LIMIT = 2.0

# -- STDP: excitatory (in→hid) --
taupre_exc      = 20  * ms
taupost_exc     = 20  * ms
Apre_delta_exc  =  0.004
Apost_delta_exc = -0.0048
wmax_exc        = 1.0
wmin_exc        = 0.0

# -- STDP: inhibitory (hid→hid) --
taupre_inh      = 20  * ms
taupost_inh     = 20  * ms
Apre_delta_inh  =  0.001
Apost_delta_inh = -0.0012
wmax_inh        = 0.2
wmin_inh        = 0.0


# ============================================================
# Initialise weight matrices
# ============================================================

# Excitatory: input → hidden, shape (N_IN, N_H)
# Uniform init, column-sum normalised to 2
w_ih = np.random.uniform(0, 1, (N_IN, N_H)).astype(float)
col_sums = w_ih.sum(axis=0)
col_sums[col_sums == 0] = 1.0
w_ih = w_ih / col_sums * EXC_SUM_LIMIT
w_ih = np.clip(w_ih, wmin_exc, wmax_exc)

# Inhibitory: hidden → hidden, shape (N_H, N_H), no self-connections
# Zero init — STDP grows weights from scratch
w_hh = np.zeros((N_H, N_H), dtype=float)


# ============================================================
# Build Brian2 network  (once — never rebuilt between samples)
# ============================================================

defaultclock.dt = DT_SIM

# ── Input neurons ──────────────────────────────────────────────────────────────
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

# ── Hidden neurons (adaptive-threshold LIF + slow AHP) ───────────────────────
eqs_h = f"""
dv/dt   = (-v - a_h) / tau_h              : 1
dvth/dt = -(vth - {vth_rest}) / tau_vth   : 1
da_h/dt = -a_h / tau_a_h                  : 1
"""
G_h = NeuronGroup(
    N_H, eqs_h,
    threshold="v > vth",
    reset=f"v=0; vth=vth+{vth_jump}; a_h+={beta_h}",
    refractory=2 * ms,
    method="euler"
)
G_h.vth = vth_init

# ── STDP synapse model (shared) ───────────────────────────────────────────────
stdp_model = """
w         : 1
dapre/dt  = -apre  / taupre  : 1 (event-driven)
dapost/dt = -apost / taupost : 1 (event-driven)
"""

# ── Excitatory synapses: input → hidden ───────────────────────────────────────
on_pre_exc  = "v_post += w\napre += Apre_delta\nw = clip(w + apost*(w - wmin), wmin, wmax)"
on_post_exc = "apost += Apost_delta\nw = clip(w + apre*(wmax - w), wmin, wmax)"

S_ih = Synapses(G_in, G_h, model=stdp_model, on_pre=on_pre_exc, on_post=on_post_exc,
                namespace={'taupre': taupre_exc, 'taupost': taupost_exc,
                           'Apre_delta': Apre_delta_exc, 'Apost_delta': Apost_delta_exc,
                           'wmin': wmin_exc, 'wmax': wmax_exc})
S_ih.connect()
src_ih = np.array(S_ih.i)
tgt_ih = np.array(S_ih.j)
S_ih.w = w_ih[src_ih, tgt_ih]

# ── Inhibitory synapses: hidden → hidden (i≠j) ───────────────────────────────
on_pre_inh  = "v_post -= w\napre += Apre_delta\nw = clip(w + apost*(w - wmin), wmin, wmax)"
on_post_inh = "apost += Apost_delta\nw = clip(w + apre*(wmax - w), wmin, wmax)"

S_hh = Synapses(G_h, G_h, model=stdp_model, on_pre=on_pre_inh, on_post=on_post_inh,
                namespace={'taupre': taupre_inh, 'taupost': taupost_inh,
                           'Apre_delta': Apre_delta_inh, 'Apost_delta': Apost_delta_inh,
                           'wmin': wmin_inh, 'wmax': wmax_inh})
S_hh.connect(condition="i != j")
src_hh = np.array(S_hh.i)
tgt_hh = np.array(S_hh.j)
S_hh.w = w_hh[src_hh, tgt_hh]

@network_operation(dt=500 * ms)
def normalise_exc_weights(t):
    if float(t / ms) == 0:
        return
    w_flat = np.array(S_ih.w)
    W = np.zeros((N_IN, N_H))
    W[src_ih, tgt_ih] = w_flat
    col_sums = W.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    W = W / col_sums * EXC_SUM_LIMIT
    W = np.clip(W, wmin_exc, wmax_exc)
    S_ih.w = W[src_ih, tgt_ih]

net = Network(G_in, G_h, S_ih, S_hh, normalise_exc_weights)


# ============================================================
# Recorder setup
# ============================================================

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "record_and_visualize_config.yaml")
recorder = Recorder(CONFIG_PATH, net)

recorder.track_group("input",  G_in)
recorder.track_group("hidden", G_h)
recorder.track_synapses("in->hid",  S_ih, src_ih, tgt_ih)
recorder.track_synapses("hid->hid", S_hh, src_hh, tgt_hh)

recorder.build()

net.store('init')


# ============================================================
# Pre-encode + auto-calibrate inputs  (once — inputs are identical every epoch)
# ============================================================
# Encoding/calibration are deterministic per file and weight-independent, so we
# do them once up front and cache the calibrated current. Calibration scales each
# file so the input layer fires at ~TARGET_INPUT_RATE_HZ regardless of the file or
# its length — the property that makes a tuned STDP setup transfer across Vox1/Vox2.

INPUT_NEURON_PARAMS = {
    "tau_m": tau_m, "tau_a": tau_a, "tau_current": tau_current,
    "beta": beta, "v_th_in": v_th_in, "refractory": 2 * ms,
}

samples = []  # (audio_path, I_calibrated, T, duration_s)
print("\nPre-encoding + calibrating inputs ...")
for sample_idx, audio_path in enumerate(wav_files):
    try:
        I, T = compute_spike_input_current(
            audio_path,
            scale=ENC_SCALE_SEED,
            num_filters=ENC_NUM_FILTERS,
            sustained_per_band=ENC_SUSTAINED,
            onset_per_band=ENC_ONSET,
            phase_per_band=ENC_PHASE,
            sust_spread_min=ENC_SUST_SPREAD_MIN,
            sust_spread_max=ENC_SUST_SPREAD_MAX,
            sust_gain=ENC_SUST_GAIN,
            onset_gain=ENC_ONSET_GAIN,
            phase_gain=ENC_PHASE_GAIN,
            normalization="contrast",
            channel_floor=ENC_CHANNEL_FLOOR,
        )
    except Exception as e:
        print(f"  [sample {sample_idx}] error encoding {audio_path}: {e}")
        continue

    I, gain, rate = calibrate_current_to_rate(
        I, dt=DT_SIM, target_rate_hz=TARGET_INPUT_RATE_HZ,
        neuron_params=INPUT_NEURON_PARAMS,
    )
    duration_s = float(T) * float(DT_SIM)
    print(f"  [sample {sample_idx}] {os.path.relpath(audio_path)} "
          f"T={T} gain={gain:.4g} -> {rate:.2f} Hz")
    samples.append((audio_path, I, T, duration_s))


# ============================================================
# Training loop
# ============================================================

for epoch_idx in range(EPOCHS):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch_idx}/{EPOCHS - 1}")
    print(f"{'='*60}")

    recorder.reset_epoch()

    for sample_idx, (audio_path, I, T, duration_s) in enumerate(samples):
        print(f"  [epoch {epoch_idx}, sample {sample_idx}/{len(samples)-1}] "
              f"{os.path.relpath(audio_path)}")

        # ── Reset to clean state, then inject this sample's data ──────────────
        net.restore('init')

        G_in.namespace["I_timed"] = TimedArray(I.T.astype(float), dt=DT_SIM)

        # Restore persistent weights and zero STDP traces
        S_ih.w = w_ih[src_ih, tgt_ih];  S_ih.apre = 0;  S_ih.apost = 0
        S_hh.w = w_hh[src_hh, tgt_hh]; S_hh.apre = 0;  S_hh.apost = 0

        recorder._elapsed_ms = 0.0

        # ── Record: before ─────────────────────────────────────────────────────
        recorder.before_sample(sample_idx)

        # ── Simulate ───────────────────────────────────────────────────────────
        net.run(T * DT_SIM)

        # ── Extract updated weights ────────────────────────────────────────────
        w_ih_new = np.zeros((N_IN, N_H))
        w_ih_new[src_ih, tgt_ih] = np.array(S_ih.w)
        w_ih = w_ih_new

        w_hh_new = np.zeros((N_H, N_H))
        w_hh_new[src_hh, tgt_hh] = np.array(S_hh.w)
        w_hh = w_hh_new

        # ── Record: after ──────────────────────────────────────────────────────
        recorder.after_sample(sample_idx, duration_s,
                              w_matrices={"in->hid": w_ih, "hid->hid": w_hh})

    # ── End of epoch ──────────────────────────────────────────────────────────
    recorder.save_epoch(epoch_idx, save_dir=SAVE_DIR,
                        final_weights={"in->hid": w_ih, "hid->hid": w_hh})


# ============================================================
# Done
# ============================================================

print(f"\n{'='*60}")
print(f"Training complete — {time.time() - start:.2f}s")
print(f"{'='*60}")
