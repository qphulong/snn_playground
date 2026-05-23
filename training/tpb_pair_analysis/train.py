import numpy as np
np.random.seed(42)
from brian2 import *
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.spike_encoding import compute_spike_input_current
from src.utils.weights_utils import gaussian_weight_matrix
from src.recorder import Recorder

import time
start = time.time()

# ============================================================
# Dataset  — edit these two lists
# ============================================================

WAV_FILES_A = [
    "datasets/vox1_fingerprint_analysis/id10545/id10545_00010/00001.wav",
    "datasets/vox1_fingerprint_analysis/id10545/id10545_00010/00002.wav",
]

WAV_FILES_B = [
    "datasets/vox1_fingerprint_analysis/id10545/id10545_00010/00003.wav",
    "datasets/vox1_fingerprint_analysis/id10545/id10545_00010/00004.wav",
]

EPOCHS   = 2
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Hyperparameters  (identical to tonotopic_plasticity_bound)
# ============================================================

N_IN = 700
N_H  = 700

DT_SIM = 1 * ms

# -- Input layer (adaptive LIF) --
tau_m       = 40 * ms
tau_a       = 100 * ms
tau_current = 1 * ms
beta        = 1
v_th_in     = 1.0

# -- Hidden layer (adaptive-threshold LIF) --
tau_h    = 50 * ms
tau_vth  = 100 * ms
vth_rest = 0.8
vth_init = 0.8
vth_jump = 0.3

# -- Soft refractory (hidden only) --
tau_r = 10 * ms

# -- Membrane noise (hidden only) --
sigma_noise = 0.03 * second**(-0.5)

# -- STDP (excitatory) --
taupre  = 20 * ms
taupost = 20 * ms

# -- Excitatory weight bounds --
wmin = 0.0

# -- Tonotopic plasticity bounds — excitatory (circular Gaussian) --
WMAX_CENTER  = 1.0
WMAX_MIN     = 0.3
APRE_CENTER  =  0.002
APOST_CENTER = -0.0024
PLAST_SIGMA  = N_IN / 5

# -- Inhibitory lateral synapse (distance-limited STDP) --
SIGMA_INH       = N_H / 5
W_INH_CENTER    = 0.9
W_INH_MIN       = 0.0
APRE_INH        = 0.002
APOST_INH       = -0.0024
INH_CUTOFF_MULT = 3

# -- Weight initialisation (Gaussian, toroidal topology) --
W_INIT_SIGMA     = N_IN / 5
W_INIT_NOISE_STD = 0.005
W_INIT_SUM       = 2

# -- Homeostatic normalisation --
NORM_LIMIT = 2

# ============================================================
# Shared initial weight matrices (computed once; seed already set above)
# ============================================================

W_INIT_IH = gaussian_weight_matrix(N_IN, N_H, W_INIT_SIGMA, W_INIT_NOISE_STD,
                                    W_INIT_SUM, wmin, WMAX_CENTER)

# ============================================================
# Pre-compute tonotopic plasticity bound matrices — excitatory
# ============================================================

_i    = np.arange(N_IN).reshape(-1, 1)
_j    = np.arange(N_H).reshape(1, -1)
_dist = np.abs(_i - _j)
_dist = np.minimum(_dist, N_IN - _dist)
_gauss = np.exp(-(_dist ** 2) / (2 * PLAST_SIGMA ** 2))

wmax_matrix  = WMAX_MIN + (WMAX_CENTER - WMAX_MIN) * _gauss
Apre_matrix  = APRE_CENTER  * _gauss
Apost_matrix = APOST_CENTER * _gauss
del _i, _j, _dist, _gauss

# ============================================================
# Pre-compute tonotopic plasticity bound matrices — inhibitory
# ============================================================

_i_hh    = np.arange(N_H).reshape(-1, 1)
_j_hh    = np.arange(N_H).reshape(1, -1)
_dist_hh = np.abs(_i_hh - _j_hh)
_dist_hh = np.minimum(_dist_hh, N_H - _dist_hh)
_gauss_hh = np.exp(-(_dist_hh ** 2) / (2 * SIGMA_INH ** 2))

wmax_inh_matrix  = W_INH_CENTER * _gauss_hh
Apre_inh_matrix  = APRE_INH     * _gauss_hh
Apost_inh_matrix = APOST_INH    * _gauss_hh

_cutoff_hh       = INH_CUTOFF_MULT * SIGMA_INH
_mask_hh         = (_dist_hh > 0) & (_dist_hh <= _cutoff_hh)
_src_hh, _tgt_hh = np.where(_mask_hh)
del _i_hh, _j_hh, _dist_hh, _gauss_hh

# ============================================================
# Build Brian2 network (once — shared across both runs)
# ============================================================

defaultclock.dt = DT_SIM

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

eqs_h = f"""
dv/dt       = -v / tau_h + sigma_noise * xi                       : 1
dvth/dt     = -(vth - {vth_rest}) / tau_vth                       : 1
dtrace_r/dt = -trace_r / tau_r                                    : 1
is_winner                                                         : boolean
"""
G_h = NeuronGroup(
    N_H, eqs_h,
    threshold="v > vth and is_winner",
    reset=f"v=0; vth=vth+{vth_jump}; trace_r=1;",
    method="euler"
)

stdp_model = """
w          : 1
dapre/dt   = -apre  / taupre  : 1 (event-driven)
dapost/dt  = -apost / taupost : 1 (event-driven)
wmax_syn   : 1
Apre_syn   : 1
Apost_syn  : 1
"""
on_pre  = "v_post += w * (1 - trace_r_post)\napre += Apre_syn\nw = clip(w + apost*(w-wmin), wmin, wmax_syn)"
on_post = "apost += Apost_syn\nw = clip(w + apre*(wmax_syn-w), wmin, wmax_syn)"

S_ih = Synapses(G_in, G_h, model=stdp_model, on_pre=on_pre, on_post=on_post)
S_ih.connect()
src_ih = np.array(S_ih.i)
tgt_ih = np.array(S_ih.j)

S_ih.wmax_syn  = wmax_matrix[src_ih, tgt_ih]
S_ih.Apre_syn  = Apre_matrix[src_ih, tgt_ih]
S_ih.Apost_syn = Apost_matrix[src_ih, tgt_ih]

stdp_inh_model = """
w_inh          : 1
dapre_inh/dt   = -apre_inh  / taupre  : 1 (event-driven)
dapost_inh/dt  = -apost_inh / taupost : 1 (event-driven)
wmax_inh_syn   : 1
Apre_inh_syn   : 1
Apost_inh_syn  : 1
"""
on_pre_inh  = (f"v_post -= w_inh\n"
               f"apre_inh += Apre_inh_syn\n"
               f"w_inh = clip(w_inh + apost_inh*(w_inh-{W_INH_MIN}), {W_INH_MIN}, wmax_inh_syn)")
on_post_inh = (f"apost_inh += Apost_inh_syn\n"
               f"w_inh = clip(w_inh + apre_inh*(wmax_inh_syn-w_inh), {W_INH_MIN}, wmax_inh_syn)")

S_hh = Synapses(G_h, G_h, model=stdp_inh_model, on_pre=on_pre_inh, on_post=on_post_inh)
S_hh.connect(i=_src_hh, j=_tgt_hh)

S_hh.wmax_inh_syn  = wmax_inh_matrix[_src_hh, _tgt_hh]
S_hh.Apre_inh_syn  = Apre_inh_matrix[_src_hh, _tgt_hh]
S_hh.Apost_inh_syn = Apost_inh_matrix[_src_hh, _tgt_hh]
S_hh.w_inh         = 0.0

@network_operation(when="before_thresholds")
def determine_winner():
    v   = G_h.v[:]
    vth = G_h.vth[:]
    crossed = v > vth
    K = 35
    if np.any(crossed):
        candidates = np.where(crossed)[0]
        sorted_idx = candidates[np.argsort(vth[candidates])]
        winners    = sorted_idx[:K]
        G_h.is_winner[:]       = False
        G_h.is_winner[winners] = True
        losers = np.setdiff1d(candidates, winners)
        G_h.v[losers] = 0.5
    else:
        G_h.is_winner[:] = False

net = Network(G_in, G_h, S_ih, S_hh, determine_winner)

# ============================================================
# Recorder setup
# ============================================================

CONFIG_PATH = os.path.join(SAVE_DIR, "record_and_visualize_config.yaml")
recorder = Recorder(CONFIG_PATH, net)

recorder.track_group("input",  G_in)
recorder.track_group("hidden", G_h)
recorder.track_synapses("in->hid",  S_ih, src_ih, tgt_ih)
recorder.track_synapses("hid->hid", S_hh, _src_hh, _tgt_hh)

recorder.build()

# ── Periodic L1 normalisation every 500 ms ───────────────────────────────────
tgt_masks_ih = [np.where(tgt_ih == j)[0] for j in range(N_H)]
wmax_syn_arr  = np.array(S_ih.wmax_syn)

@network_operation(dt=500*ms, when='end')
def normalize_weights():
    for j in range(N_H):
        idx   = tgt_masks_ih[j]
        w_col = np.array(S_ih.w[idx])
        wsum  = w_col.sum()
        if wsum > NORM_LIMIT and NORM_LIMIT > 0:
            S_ih.w[idx] = np.clip(w_col * NORM_LIMIT / wsum, wmin, wmax_syn_arr[idx])

net.add(normalize_weights)

G_h.vth = vth_init
net.store('init')

# ============================================================
# Training helper
# ============================================================

def train_run(wav_files, save_subdir, label):
    """Train one independent SNN run. Returns (w_ih, w_hh) after final epoch."""
    save_dir     = os.path.join(SAVE_DIR, save_subdir)
    w_ih         = W_INIT_IH.copy()
    w_hh         = np.zeros((N_H, N_H))
    init_weights = {"in->hid": w_ih.copy(), "hid->hid": w_hh.copy()}

    print(f"\n{'='*60}")
    print(f"Run {label}  ({len(wav_files)} file(s), {EPOCHS} epoch(s))")
    print(f"{'='*60}")

    for epoch_idx in range(EPOCHS):
        print(f"\n  Epoch {epoch_idx}/{EPOCHS - 1}")
        recorder.reset_epoch()

        for sample_idx, audio_path in enumerate(wav_files):
            print(f"    [epoch {epoch_idx}, sample {sample_idx}/{len(wav_files)-1}] "
                  f"{os.path.relpath(audio_path)}")

            try:
                I, T = compute_spike_input_current(
                    audio_path,
                    scale=0.8,
                    sustained_per_band=4,
                    onset_per_band=2,
                    phase_per_band=1,
                    sust_spread_min=0.7,
                    sust_spread_max=1.3,
                )
            except Exception as e:
                print(f"      Error encoding audio: {e}")
                continue

            duration_s = float(T) * float(DT_SIM)

            net.restore('init')
            G_in.namespace["I_timed"] = TimedArray(I.T.astype(float), dt=DT_SIM)

            S_ih.w    = w_ih[src_ih, tgt_ih]
            S_ih.apre  = 0
            S_ih.apost = 0

            S_hh.w_inh     = w_hh[_src_hh, _tgt_hh]
            S_hh.apre_inh  = 0
            S_hh.apost_inh = 0

            recorder._elapsed_ms = 0.0
            recorder.before_sample(sample_idx)

            net.run(T * DT_SIM)

            w_ih_new = np.zeros((N_IN, N_H))
            w_ih_new[src_ih, tgt_ih] = np.array(S_ih.w)
            w_ih = w_ih_new

            w_hh_new = np.zeros((N_H, N_H))
            w_hh_new[_src_hh, _tgt_hh] = np.array(S_hh.w_inh)
            w_hh = w_hh_new

            recorder.after_sample(
                sample_idx,
                duration_s,
                w_matrices={
                    "in->hid":  w_ih,
                    "hid->hid": w_hh,
                }
            )

        recorder.save_epoch(
            epoch_idx,
            save_dir=save_dir,
            final_weights={
                "in->hid":  w_ih,
                "hid->hid": w_hh,
            },
            init_weights=init_weights if epoch_idx == 0 else None,
        )

    return w_ih, w_hh

# ============================================================
# Main — run A then B, then save fingerprints
# ============================================================

w_ih_A, w_hh_A = train_run(WAV_FILES_A, "run_A", "A")
w_ih_B, w_hh_B = train_run(WAV_FILES_B, "run_B", "B")

fp_path = os.path.join(SAVE_DIR, "fingerprints.npz")
np.savez(
    fp_path,
    fingerprints = np.stack([w_ih_A, w_ih_B], axis=0).astype(np.float32),
    labels       = np.array(["A", "B"]),
    wav_files_A  = np.array(WAV_FILES_A),
    wav_files_B  = np.array(WAV_FILES_B),
)
print(f"\nFingerprints saved → {os.path.relpath(fp_path)}")

print(f"\n{'='*60}")
print(f"Done — {time.time() - start:.2f}s")
print(f"{'='*60}")
