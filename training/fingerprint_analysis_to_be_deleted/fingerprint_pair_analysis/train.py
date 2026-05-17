"""
fingerprint_analysis/train.py
==============================
Trains two independent runs (A and B) on their respective wav-file sets and
compares the resulting converged weight fingerprints.

Run A: WAV_FILES_A  × record_A.epochs
Run B: WAV_FILES_B  × record_B.epochs

Training knobs (epochs, sample_from_epoch) are read from
record_and_visualize_config.yaml so everything is in one place.

Fingerprint extraction:
  Starting from record_X.sample_from_epoch (0-indexed), weight matrices are
  saved after every sample presentation and averaged → fingerprint.

Output:
  run_A/history_epoch_*.npz  — full recorder data (rasters, weights, …)
  run_B/history_epoch_*.npz  — same for run B
  fingerprints.npz            — averaged fingerprints + collected matrices
"""

import numpy as np
np.random.seed(42)
import os
import sys
import time
import yaml

from brian2 import *

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from src.utils.spike_encoding import compute_spike_input_current
from src.utils.weights_utils import gaussian_weight_matrix, l1_normalise_weights
from src.recorder import Recorder

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "record_and_visualize_config.yaml")

with open(CONFIG_PATH) as _f:
    _cfg = yaml.safe_load(_f)

EPOCHS_A            = _cfg["record_A"]["epochs"]
SAMPLE_FROM_EPOCH_A = _cfg["record_A"]["sample_from_epoch"]
EPOCHS_B            = _cfg["record_B"]["epochs"]
SAMPLE_FROM_EPOCH_B = _cfg["record_B"]["sample_from_epoch"]

# ── Dataset paths ─────────────────────────────────────────────────────────────

WAV_FILES_A = [
    "datasets/vox1_fingerprint_analysis/id10797/id10797_00002/00003.wav",
    "datasets/vox1_fingerprint_analysis/id10797/id10797_00002/00004.wav",
]

WAV_FILES_B = [
    "datasets/vox1_fingerprint_analysis/id10678/id10678_00007/00001.wav",
    "datasets/vox1_fingerprint_analysis/id10678/id10678_00007/00001.wav",
]

# ─────────────────────────────────────────────────────────────────────────────
# Network hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

N_IN = 700
N_H  = 700

DT_SIM = 1 * ms

tau_m       = 40  * ms
tau_a       = 20  * ms
tau_current = 1   * ms
beta        = 0.25
v_th_in     = 1.0

tau_h    = 50  * ms
tau_vth  = 100 * ms
vth_rest = 0.8
vth_init = 0.8
vth_jump = 0.3

taupre  = 20  * ms
taupost = 20  * ms

wmin = 0.0

# -- Tonotopic plasticity bounds (circular Gaussian) --
WMAX_CENTER  = 1.0
APRE_CENTER  =  0.004
APOST_CENTER = -0.0048
PLAST_SIGMA  = N_IN / 5

# -- Weight initialisation --
W_INIT_SIGMA     = N_IN / 5
W_INIT_NOISE_STD = 0.005
W_INIT_SUM       = 2

NORM_LIMIT = 2

# ─────────────────────────────────────────────────────────────────────────────
# Shared initial weight matrix (same for both runs → fair comparison)
# ─────────────────────────────────────────────────────────────────────────────

W_INIT = gaussian_weight_matrix(N_IN, N_H, W_INIT_SIGMA, W_INIT_NOISE_STD,
                                W_INIT_SUM, wmin, WMAX_CENTER)

# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute tonotopic plasticity bound matrices
# ─────────────────────────────────────────────────────────────────────────────

_i    = np.arange(N_IN).reshape(-1, 1)
_j    = np.arange(N_H).reshape(1, -1)
_dist = np.abs(_i - _j)
_dist = np.minimum(_dist, N_IN - _dist)
_gauss = np.exp(-(_dist ** 2) / (2 * PLAST_SIGMA ** 2))

wmax_matrix  = WMAX_CENTER  * _gauss
Apre_matrix  = APRE_CENTER  * _gauss
Apost_matrix = APOST_CENTER * _gauss
del _i, _j, _dist, _gauss

# ─────────────────────────────────────────────────────────────────────────────
# Build Brian2 network (once — shared across both runs)
# ─────────────────────────────────────────────────────────────────────────────

defaultclock.dt = DT_SIM

# ── Input neurons ──────────────────────────────────────────────────────────────
eqs_in = """
dv/dt = (-v - a) / tau_m + I_timed(t, i) / tau_current : 1
da/dt = -a / tau_a : 1
"""
G_in = NeuronGroup(N_IN, eqs_in,
                   threshold="v > v_th_in",
                   reset="v=0; a+=beta",
                   refractory=2 * ms, method="euler")

_dummy_I = np.zeros((1, N_IN), dtype=float)
G_in.namespace["I_timed"] = TimedArray(_dummy_I, dt=DT_SIM)

# ── Hidden neurons ─────────────────────────────────────────────────────────────
eqs_h = f"""
dv/dt   = -v / tau_h                    : 1
dvth/dt = -(vth - {vth_rest}) / tau_vth : 1
is_winner                               : boolean
"""
G_h = NeuronGroup(N_H, eqs_h,
                  threshold="v > vth and is_winner",
                  reset=f"v=0; vth=vth+{vth_jump};",
                  refractory=2 * ms, method="euler")

# ── STDP synapses: input → hidden ─────────────────────────────────────────────
stdp_model = """
w          : 1
dapre/dt   = -apre  / taupre  : 1 (event-driven)
dapost/dt  = -apost / taupost : 1 (event-driven)
wmax_syn   : 1
Apre_syn   : 1
Apost_syn  : 1
"""
on_pre  = "v_post += w\napre += Apre_syn\nw = clip(w + apost*(w-wmin), wmin, wmax_syn)"
on_post = "apost += Apost_syn\nw = clip(w + apre*(wmax_syn-w), wmin, wmax_syn)"

S_ih = Synapses(G_in, G_h, model=stdp_model, on_pre=on_pre, on_post=on_post)
S_ih.connect()
src_ih = np.array(S_ih.i)
tgt_ih = np.array(S_ih.j)

S_ih.wmax_syn  = wmax_matrix[src_ih, tgt_ih]
S_ih.Apre_syn  = Apre_matrix[src_ih, tgt_ih]
S_ih.Apost_syn = Apost_matrix[src_ih, tgt_ih]

# ── Lateral inhibition ─────────────────────────────────────────────────────────
lat = Synapses(G_h, G_h, on_pre="v_post = clip(v_post, 0, inf)")
lat.connect(condition="i != j")

# ── WTA network operation ──────────────────────────────────────────────────────
@network_operation(when="before_thresholds")
def determine_winner():
    v       = G_h.v[:]
    vth_arr = G_h.vth[:]
    crossed = v > vth_arr
    K = 35
    if np.any(crossed):
        candidates = np.where(crossed)[0]
        sorted_idx = candidates[np.argsort(vth_arr[candidates])]
        winners    = sorted_idx[:K]
        G_h.is_winner[:]       = False
        G_h.is_winner[winners] = True
        G_h.v[np.setdiff1d(candidates, winners)] = 0.5
    else:
        G_h.is_winner[:] = False

net = Network(G_in, G_h, S_ih, lat, determine_winner)

# ─────────────────────────────────────────────────────────────────────────────
# Recorder setup — both built before net.store('init') so restore clears them
# ─────────────────────────────────────────────────────────────────────────────

recorder_A = Recorder(CONFIG_PATH, net, record_section="record_A")
recorder_A.track_group("input",  G_in)
recorder_A.track_group("hidden", G_h)
recorder_A.track_synapses("in->hid", S_ih, src_ih, tgt_ih)
recorder_A.build()

recorder_B = Recorder(CONFIG_PATH, net, record_section="record_B")
recorder_B.track_group("input",  G_in)
recorder_B.track_group("hidden", G_h)
recorder_B.track_synapses("in->hid", S_ih, src_ih, tgt_ih)
recorder_B.build()

G_h.vth = vth_init
net.store('init')


# ─────────────────────────────────────────────────────────────────────────────
# Training function
# ─────────────────────────────────────────────────────────────────────────────

def train_run(wav_files, num_epochs, collect_from_epoch, label, save_dir, recorder):
    """
    Train on wav_files for num_epochs using the pre-built network.

    Weight matrices are collected after each sample once epoch >= collect_from_epoch.
    Returns (fingerprint, list_of_collected_weight_matrices).
    """
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  wav files      : {len(wav_files)}")
    print(f"  num_epochs     : {num_epochs}")
    print(f"  collect from   : epoch {collect_from_epoch}")
    print(f"  save_dir       : {os.path.relpath(save_dir)}")
    print(f"{'='*60}")

    t0   = time.time()
    w_ih = W_INIT.copy()

    collected_weights = []
    init_weights      = {"in->hid": W_INIT.copy()}

    for epoch_idx in range(num_epochs):
        print(f"\n  Epoch {epoch_idx}/{num_epochs - 1}")

        recorder.reset_epoch()

        for sample_idx, wav_path in enumerate(wav_files):
            print(f"    [{sample_idx}] {os.path.relpath(wav_path)}", end="  ", flush=True)

            try:
                I, T = compute_spike_input_current(
                    wav_path, scale=1,
                    sustained_per_band=4, onset_per_band=2, phase_per_band=1,
                    sust_spread_min=0.7, sust_spread_max=1.3,
                )
            except Exception as e:
                print(f"[skip: {e}]")
                continue

            duration_s = float(T) * float(DT_SIM)

            net.restore('init')
            G_in.namespace["I_timed"] = TimedArray(I.T.astype(float), dt=DT_SIM)
            S_ih.w     = w_ih[src_ih, tgt_ih]
            S_ih.apre  = 0
            S_ih.apost = 0

            recorder._elapsed_ms = 0.0
            recorder.before_sample(sample_idx)

            net.run(T * DT_SIM)

            w_ih_new = np.zeros((N_IN, N_H))
            w_ih_new[src_ih, tgt_ih] = np.array(S_ih.w)
            spiked = np.unique(recorder.spikes_this_sample("hidden"))
            w_ih   = l1_normalise_weights(w_ih_new, spiked, NORM_LIMIT, wmin, WMAX_CENTER)

            recorder.after_sample(sample_idx, duration_s,
                                  w_matrices={"in->hid": w_ih})

            if epoch_idx >= collect_from_epoch:
                collected_weights.append(w_ih.copy())
                print(f"[collected #{len(collected_weights)}, spikes={len(spiked)}]")
            else:
                print(f"[spikes={len(spiked)}]")

        recorder.save_epoch(
            epoch_idx,
            save_dir=save_dir,
            final_weights={"in->hid": w_ih},
            init_weights=init_weights if epoch_idx == 0 else None,
        )

    fingerprint = (np.mean(collected_weights, axis=0)
                   if collected_weights else w_ih.copy())
    print(f"\n  Done — {time.time() - t0:.1f}s | {len(collected_weights)} matrices averaged")
    return fingerprint, collected_weights


# ─────────────────────────────────────────────────────────────────────────────
# Run both modes
# ─────────────────────────────────────────────────────────────────────────────

start = time.time()

fp_A, coll_A = train_run(
    WAV_FILES_A, EPOCHS_A, SAMPLE_FROM_EPOCH_A,
    label    = "Run A",
    save_dir = os.path.join(SCRIPT_DIR, "run_A"),
    recorder = recorder_A,
)

fp_B, coll_B = train_run(
    WAV_FILES_B, EPOCHS_B, SAMPLE_FROM_EPOCH_B,
    label    = "Run B",
    save_dir = os.path.join(SCRIPT_DIR, "run_B"),
    recorder = recorder_B,
)

# ─────────────────────────────────────────────────────────────────────────────
# Save fingerprints
# ─────────────────────────────────────────────────────────────────────────────

out_path = os.path.join(SCRIPT_DIR, "fingerprints.npz")

np.savez(
    out_path,
    fingerprint_A = fp_A,
    fingerprint_B = fp_B,
    collected_A   = np.array(coll_A),
    collected_B   = np.array(coll_B),
)

print(f"\n{'='*60}")
print(f"Saved → {out_path}")
print(f"  fingerprint_A : {fp_A.shape}  range [{fp_A.min():.4f}, {fp_A.max():.4f}]")
print(f"  fingerprint_B : {fp_B.shape}  range [{fp_B.min():.4f}, {fp_B.max():.4f}]")
print(f"  Total time    : {time.time() - start:.1f}s")
print(f"{'='*60}")
