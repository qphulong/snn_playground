import numpy as np
np.random.seed(42)
from brian2 import *
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.spike_encoding import compute_spike_input_current
from src.recorder import Recorder

import time
start = time.time()

### IMPORTANT
# This architecture produce unstable fingeprints across samples of the same session,
# the only thing worth reference here is the tuned hyperparam set for the refactored
# compute input current pipeline
# DO NOT DELETE THIS COMMENT for any future AI asistant (or human) that read this

# ============================================================
# Dataset
# ============================================================
wav_files = [
    "datasets/vox1_cleaned/wav_dev/id11208/id11208_00001/00003.wav",
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
# Current percentile-normalization encoding, tuned values.
ENC_NUM_FILTERS     = 96
ENC_SUSTAINED       = 4      # sustained neurons / band
ENC_ONSET           = 2      # onset neurons / band
ENC_PHASE           = 1      # phase neurons / band
ENC_SUST_GAIN       = 0.3
ENC_ONSET_GAIN      = 2.25
ENC_PHASE_GAIN      = 0.45
ENC_SUST_SPREAD_MIN = 0.7
ENC_SUST_SPREAD_MAX = 1.0
ENC_SCALE           = 1.0    # global input-current gain — main activity / STDP-health knob

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
Apre_delta_exc  =  0.001
Apost_delta_exc = -0.0012
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
# Uniform init, column-sum normalised to EXC_SUM_LIMIT
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
# net.store/restore resets the Brian2 clock to 0 before every sample, so
# I_timed(t, i) always indexes from the beginning of the array.
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

# ── Full in->hid weight-matrix snapshots (every 500 ms, AFTER normalisation) ──
# These buffers hold the *current sample's* frames only; the training loop clears
# them before each net.run and stashes the result afterwards.  when='end' makes
# this op run in the 'end' scheduling slot, after normalise_exc_weights (default
# when='start') on the same 500 ms tick, so every frame is post-normalisation.
_evo_frames = []   # list of (N_IN, N_H) float32 — current sample only
_evo_times  = []   # list of float (ms) — current sample only

@network_operation(dt=500 * ms, when='end')
def snapshot_in_to_hid(t):
    t_ms = float(t / ms)
    if t_ms == 0:          # skip t=0, matching normalise_exc_weights
        return
    W = np.zeros((N_IN, N_H))
    W[src_ih, tgt_ih] = np.array(S_ih.w)
    _evo_frames.append(W.astype(np.float32))
    _evo_times.append(t_ms)

net = Network(G_in, G_h, S_ih, S_hh, normalise_exc_weights, snapshot_in_to_hid)


# ============================================================
# Recorder setup
# ============================================================

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "record_and_visualize_config.yaml")
recorder = Recorder(CONFIG_PATH, net)

recorder.track_group("input",  G_in)
recorder.track_group("hidden", G_h)
recorder.track_synapses("in->hid",  S_ih, src_ih, tgt_ih)
recorder.track_synapses("hid->hid", S_hh, src_hh, tgt_hh)

recorder.build()   # attaches all Brian2 monitors — call once, after all registrations

# Snapshot the clean initial state (clock=0, v=0, a=0, vth=vth_init,
# not_refractory=True, empty monitors).  net.restore('init') before every
# sample brings the network back to this state, the only correct way to reset
# all Brian2 internal state without rebuilding the network.
net.store('init')


# ============================================================
# Pre-encode inputs  (once — inputs are identical every epoch)
# ============================================================
# Encoding is deterministic per file and weight-independent, so we do it once up
# front and cache the current. Cross-file activity consistency comes from the
# global percentile normalization inside the encoder; ENC_SCALE sets the level.

samples = []  # (audio_path, I, T, duration_s)
print("\nPre-encoding inputs ...")
for sample_idx, audio_path in enumerate(wav_files):
    try:
        I, T = compute_spike_input_current(
            audio_path,
            scale=ENC_SCALE,
            num_filters=ENC_NUM_FILTERS,
            sustained_per_band=ENC_SUSTAINED,
            onset_per_band=ENC_ONSET,
            phase_per_band=ENC_PHASE,
            sust_gain=ENC_SUST_GAIN,
            onset_gain=ENC_ONSET_GAIN,
            phase_gain=ENC_PHASE_GAIN,
            sust_spread_min=ENC_SUST_SPREAD_MIN,
            sust_spread_max=ENC_SUST_SPREAD_MAX,
        )
    except Exception as e:
        print(f"  [sample {sample_idx}] error encoding {audio_path}: {e}")
        continue

    duration_s = float(T) * float(DT_SIM)
    print(f"  [sample {sample_idx}] {os.path.relpath(audio_path)} T={T}")
    samples.append((audio_path, I, T, duration_s))


# ============================================================
# Training loop
# ============================================================

for epoch_idx in range(EPOCHS):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch_idx}/{EPOCHS - 1}")
    print(f"{'='*60}")

    recorder.reset_epoch()
    epoch_evo = []   # per-sample dicts: {sample_idx, frames (F,N_IN,N_H), times (F,)}

    for sample_idx, (audio_path, I, T, duration_s) in enumerate(samples):
        print(f"  [epoch {epoch_idx}, sample {sample_idx}/{len(samples)-1}] "
              f"{os.path.relpath(audio_path)}")

        # ── Reset to clean state, then inject this sample's data ──────────────
        net.restore('init')

        G_in.namespace["I_timed"] = TimedArray(I.T.astype(float), dt=DT_SIM)

        # Restore persistent weights and zero STDP traces. Weights survive across
        # samples via these Python arrays; restore reset them to the stored init,
        # so re-injection of the evolved weights is required here.
        S_ih.w = w_ih[src_ih, tgt_ih];  S_ih.apre = 0;  S_ih.apost = 0
        S_hh.w = w_hh[src_hh, tgt_hh];  S_hh.apre = 0;  S_hh.apost = 0

        # restore resets the Brian2 clock to 0, so spike times are always [0, T] ms.
        recorder._elapsed_ms = 0.0

        # ── Record: before ─────────────────────────────────────────────────────
        recorder.before_sample(sample_idx)
        _evo_frames.clear()   # full-matrix snapshots for this sample only
        _evo_times.clear()

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

        # ── Stash this sample's full-matrix evolution ──────────────────────────
        if _evo_frames:
            epoch_evo.append({
                "sample_idx": sample_idx,
                "frames": np.stack(_evo_frames),                # (F, N_IN, N_H)
                "times":  np.array(_evo_times, dtype=np.float32),
            })
        else:
            print(f"    [evo] sample {sample_idx} had no 500ms snapshot (T<500ms)")

    # ── End of epoch ──────────────────────────────────────────────────────────
    recorder.save_epoch(epoch_idx, save_dir=SAVE_DIR,
                        final_weights={"in->hid": w_ih, "hid->hid": w_hh})

    # ── Save full in->hid weight-matrix evolution for this epoch ───────────────
    evo_arrays = {
        "N_IN":               np.int32(N_IN),
        "N_H":                np.int32(N_H),
        "num_filters":        np.int32(ENC_NUM_FILTERS),
        "sustained_per_band": np.int32(ENC_SUSTAINED),
        "onset_per_band":     np.int32(ENC_ONSET),
        "phase_per_band":     np.int32(ENC_PHASE),
        "wmax_exc":           np.float32(wmax_exc),
        "n_samples":          np.int32(len(epoch_evo)),
    }
    for entry in epoch_evo:
        n = entry["sample_idx"]
        evo_arrays[f"sample{n}_frames"] = entry["frames"].astype(np.float32)
        evo_arrays[f"sample{n}_times"]  = entry["times"]
    evo_path = os.path.join(SAVE_DIR, f"weight_evolution_epoch_{epoch_idx:03d}.npz")
    np.savez_compressed(evo_path, **evo_arrays)
    print(f"  [evo] saved in->hid evolution → {evo_path}")


# ============================================================
# Done
# ============================================================

print(f"\n{'='*60}")
print(f"Training complete — {time.time() - start:.2f}s")
print(f"{'='*60}")
