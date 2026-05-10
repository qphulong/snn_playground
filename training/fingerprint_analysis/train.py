"""
convergence_proof/train.py
==========================
Proves that training on 1 sample repeated N times produces the same learned
fingerprint as training on 3 distinct samples for EPOCHS epochs (same total).

Mode A (3-sample): WAV_FILES_3  × EPOCHS      epochs
Mode B (1-sample): WAV_FILES_1  × EPOCHS * 3  epochs

Spike rasters and firing rates are recorded by the Recorder class, driven by
record_and_visualize_config.yaml.  Each run saves its epoch history to its own
subdirectory (run_3sample/ and run_1sample/).

Fingerprint extraction:
  Starting from collect_from_epoch (0-indexed), weight matrices are saved after
  every sample presentation and averaged → fingerprint.
  For 3-sample: collect_from = SAMPLE_FROM_EPOCH
  For 1-sample: collect_from = SAMPLE_FROM_EPOCH * 3  (equivalent training point)

Output:
  run_3sample/history_epoch_*.npz  — full recorder data (rasters, weights, …)
  run_1sample/history_epoch_*.npz  — same for 1-sample run
  fingerprints.npz                 — averaged fingerprints + collected matrices
"""

import numpy as np
import os
import sys
import time

from brian2 import *

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.spike_encoding import compute_spike_input_current
from src.utils.weights_utils import gaussian_weight_matrix, l1_normalise_weights
from src.recorder import Recorder

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "record_and_visualize_config.yaml")

EPOCHS            = 4   # epochs for the 3-sample version; 1-sample runs EPOCHS*3
SAMPLE_FROM_EPOCH = 2   # 0-indexed; start collecting weights from this epoch

# ── Dataset paths ─────────────────────────────────────────────────────────────
WAV_FILES_3 = [
    "datasets/vox1_fingerprint_analysis/id10340/id10340_00004/00002.wav",
    "datasets/vox1_fingerprint_analysis/id10340/id10340_00004/00003.wav",
    "datasets/vox1_fingerprint_analysis/id10340/id10340_00004/00004.wav",
]

WAV_FILES_1 = [
    "datasets/vox1_fingerprint_analysis/id10340/id10340_00004/00001.wav",
]

# ─────────────────────────────────────────────────────────────────────────────
# Network hyperparameters  (must match training/fingerprint_analysis/train.py)
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

taupre      = 20  * ms
taupost     = 20  * ms
Apre_delta  =  0.0001
Apost_delta = -0.00012

wmax = 1.0
wmin = 0.0

W_INIT_SIGMA     = N_IN / 5
W_INIT_NOISE_STD = 0.005
W_INIT_SUM       = 2

NORM_LIMIT = 2

# ─────────────────────────────────────────────────────────────────────────────
# Shared initial weight matrix (same for both runs → fair comparison)
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
W_INIT = gaussian_weight_matrix(N_IN, N_H, W_INIT_SIGMA, W_INIT_NOISE_STD,
                                W_INIT_SUM, wmin, wmax)


# ─────────────────────────────────────────────────────────────────────────────
# Training function
# ─────────────────────────────────────────────────────────────────────────────

def train_run(wav_files, num_epochs, collect_from_epoch, label, save_dir):
    """
    Build a fresh Brian2 SNN, train on wav_files for num_epochs.

    The Recorder handles spike rasters and firing rates, driven by
    record_and_visualize_config.yaml.  Epoch data is saved to save_dir.

    Weight matrices are collected after each sample once epoch >= collect_from_epoch.
    Returns (fingerprint, list_of_collected_weight_matrices).
    """
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  wav files    : {len(wav_files)}")
    print(f"  num_epochs   : {num_epochs}")
    print(f"  collect from : epoch {collect_from_epoch}")
    print(f"  save_dir     : {os.path.relpath(save_dir)}")
    print(f"{'='*60}")

    t0   = time.time()
    w_ih = W_INIT.copy()

    # ── Build Brian2 network ──────────────────────────────────────────────────
    start_scope()
    defaultclock.dt = DT_SIM

    _dummy_I = np.zeros((1, N_IN), dtype=float)
    I_timed  = TimedArray(_dummy_I, dt=DT_SIM)

    eqs_in = """
    dv/dt = (-v - a) / tau_m + I_timed(t, i) / tau_current : 1
    da/dt = -a / tau_a : 1
    """
    G_in = NeuronGroup(N_IN, eqs_in,
                       threshold="v > v_th_in",
                       reset="v=0; a+=beta",
                       refractory=2 * ms, method="euler")

    eqs_h = f"""
    dv/dt   = -v / tau_h                    : 1
    dvth/dt = -(vth - {vth_rest}) / tau_vth : 1
    is_winner                               : boolean
    """
    G_h = NeuronGroup(N_H, eqs_h,
                      threshold="v > vth and is_winner",
                      reset=f"v=0; vth=vth+{vth_jump};",
                      refractory=2 * ms, method="euler")

    stdp_model = """
    w         : 1
    dapre/dt  = -apre  / taupre  : 1 (event-driven)
    dapost/dt = -apost / taupost : 1 (event-driven)
    """
    on_pre  = "v_post += w\napre += Apre_delta\nw = clip(w + apost*(w-wmin), wmin, wmax)"
    on_post = "apost += Apost_delta\nw = clip(w + apre*(wmax-w), wmin, wmax)"

    S_ih = Synapses(G_in, G_h, model=stdp_model, on_pre=on_pre, on_post=on_post)
    S_ih.connect()
    src_ih = np.array(S_ih.i)
    tgt_ih = np.array(S_ih.j)

    lat = Synapses(G_h, G_h, on_pre="v_post = clip(v_post, 0, inf)")
    lat.connect(condition="i != j")

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
    G_h.vth = vth_init

    # ── Recorder setup ────────────────────────────────────────────────────────
    recorder = Recorder(CONFIG_PATH, net)
    recorder.track_group("input",  G_in)
    recorder.track_group("hidden", G_h)
    recorder.track_synapses("in->hid", S_ih, src_ih, tgt_ih)
    recorder.build()

    net.store('init')

    # ── Training loop ─────────────────────────────────────────────────────────
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
            w_ih   = l1_normalise_weights(w_ih_new, spiked, NORM_LIMIT, wmin, wmax)

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

fp3, coll3 = train_run(
    WAV_FILES_3, EPOCHS, SAMPLE_FROM_EPOCH,
    label    = "Mode A — 3-sample",
    save_dir = os.path.join(SCRIPT_DIR, "run_3sample"),
)

fp1, coll1 = train_run(
    WAV_FILES_1, EPOCHS * 3, SAMPLE_FROM_EPOCH * 3,
    label    = "Mode B — 1-sample",
    save_dir = os.path.join(SCRIPT_DIR, "run_1sample"),
)

# ─────────────────────────────────────────────────────────────────────────────
# Save fingerprints
# ─────────────────────────────────────────────────────────────────────────────

out_path = os.path.join(SCRIPT_DIR, "fingerprints.npz")

np.savez(
    out_path,
    fingerprint_3 = fp3,
    fingerprint_1 = fp1,
    collected_3   = np.array(coll3),
    collected_1   = np.array(coll1),
)

print(f"\n{'='*60}")
print(f"Saved → {out_path}")
print(f"  fingerprint_3 : {fp3.shape}  range [{fp3.min():.4f}, {fp3.max():.4f}]")
print(f"  fingerprint_1 : {fp1.shape}  range [{fp1.min():.4f}, {fp1.max():.4f}]")
print(f"  Total time    : {time.time() - start:.1f}s")
print(f"{'='*60}")
