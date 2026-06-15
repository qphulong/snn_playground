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

# ============================================================
# Dataset
# ============================================================
# wav_files = [
#     # "datasets/vox1_fingerprint_analysis/id10797/id10797_00007/00003.wav",
#     "datasets/vox1_fingerprint_analysis/id10797/id10797_00007/00004.wav"
# ]

wav_files = [
    # "datasets/vox1_cleaned/wav_dev/id10007/id10007_00002/00001.wav",
    "datasets/vox1_cleaned/wav_dev/id10007/id10007_00002/00002.wav"
]

print(f"Found {len(wav_files)} wav files")

EPOCHS   = 4
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Hyperparameters
# ============================================================

N_IN = 384   # 96 channels × 4 neurons/channel
N_H  = 384

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

# -- Soft refractory (hidden only; replaces hard refractory) --
tau_r = 10 * ms

# -- Membrane noise (hidden only) --
# sigma_noise has units second**(-0.5) so sigma_noise*xi has units 1/second,
# matching dv/dt for a dimensionless variable.
sigma_noise = 0.03 * second**(-0.5)

# -- STDP (excitatory) --
taupre  = 20 * ms
taupost = 20 * ms

# -- Excitatory weight bounds --
wmin = 0.0

# -- Excitatory synapse --
WMAX_CENTER  = 1.0
APRE_CENTER  =  0.01
APOST_CENTER = -0.012

# -- Inhibitory lateral synapse --
W_INH_CENTER = 1.0
W_INH_MIN    = 0.0
APRE_INH     = 0.0005
APOST_INH    = -0.0006

# -- Channel layout --
N_CHANNELS    = 96
N_PER_CHANNEL = N_IN // N_CHANNELS   # 4

# -- Tonotopic plasticity (polynomial decay: max(0, 1-(d_channel/R)^p)) --
R_EXC_CHANNEL = 16
p_EXC         = 3

# -- Homeostatic normalisation --
NORM_LIMIT_EXC = 2
NORM_LIMIT_INH = 0.9


# ============================================================
# Pre-compute tonotopic plasticity matrices — excitatory
# Distance is measured in channels: d_ch = |ch_i - ch_j| (circular on N_CHANNELS).
# topo(d_ch) = max(0, 1-(d_ch/R_EXC_CHANNEL)^p); hard cutoff at d_ch=R_EXC_CHANNEL.
# ============================================================

_ch_i    = (np.arange(N_IN) // N_PER_CHANNEL).reshape(-1, 1)   # (N_IN, 1)
_ch_j    = (np.arange(N_H)  // N_PER_CHANNEL).reshape(1, -1)   # (1, N_H)
_dist_ch = np.abs(_ch_i - _ch_j)
_dist_ch = np.minimum(_dist_ch, N_CHANNELS - _dist_ch)          # circular
_topo_exc = np.maximum(0.0, 1.0 - (_dist_ch / R_EXC_CHANNEL) ** p_EXC)
_mask_ih  = _dist_ch <= R_EXC_CHANNEL
_src_ih, _tgt_ih = np.where(_mask_ih)

wmax_matrix  = WMAX_CENTER  * _topo_exc
Apre_matrix  = APRE_CENTER  * _topo_exc
Apost_matrix = APOST_CENTER * _topo_exc
del _ch_i, _ch_j, _dist_ch, _topo_exc

# ============================================================
# Pre-compute inhibitory plasticity matrices — Jaccard similarity.
# Two hidden neurons inhibit each other iff their excitatory input sets overlap
# (i.e., they share at least one input neuron). wmax/Apre/Apost scale by Jaccard.
#
# Since every hidden neuron j receives from a window of (2*R_EXC_CHANNEL+1) channels
# centred on j//N_PER_CHANNEL (circular), the Jaccard of neurons h1, h2 simplifies to:
#   jaccard = overlap_channels / (window_size + d_ch_hh)
# where overlap_channels = max(0, window_size - d_ch_hh).
# No self-connections.
# ============================================================

_ch_h       = np.arange(N_H) // N_PER_CHANNEL
_dist_ch_hh = np.abs(_ch_h.reshape(-1, 1) - _ch_h.reshape(1, -1))
_dist_ch_hh = np.minimum(_dist_ch_hh, N_CHANNELS - _dist_ch_hh)   # circular

_window_size = 2 * R_EXC_CHANNEL + 1                               # 17
_overlap_ch  = np.maximum(0, _window_size - _dist_ch_hh)
_jaccard     = np.where(
    _overlap_ch > 0,
    _overlap_ch / (_window_size + _dist_ch_hh),
    0.0,
)
_mask_hh  = (_overlap_ch > 0) & (~np.eye(N_H, dtype=bool))
_src_hh, _tgt_hh = np.where(_mask_hh)

wmax_inh_matrix  = W_INH_CENTER * _jaccard
Apre_inh_matrix  = APRE_INH     * _jaccard
Apost_inh_matrix = APOST_INH    * _jaccard
del _ch_h, _dist_ch_hh, _overlap_ch, _jaccard

# ============================================================
# Initialise weight matrices
# ============================================================

# Excitatory: formula-shaped, column-normalised to NORM_LIMIT_EXC
w_ih = np.zeros((N_IN, N_H))
w_ih[_src_ih, _tgt_ih] = wmax_matrix[_src_ih, _tgt_ih]
for _j in range(N_H):
    _col_mask = (_tgt_ih == _j)
    _rows = _src_ih[_col_mask]
    _wsum = w_ih[_rows, _j].sum()
    if _wsum > 0:
        w_ih[_rows, _j] *= NORM_LIMIT_EXC / _wsum

# Inhibitory: formula-shaped, column-normalised to NORM_LIMIT_INH
w_hh = np.zeros((N_H, N_H))
w_hh[_src_hh, _tgt_hh] = wmax_inh_matrix[_src_hh, _tgt_hh]
for _j in range(N_H):
    _col_mask = (_tgt_hh == _j)
    _rows = _src_hh[_col_mask]
    _wsum = w_hh[_rows, _j].sum()
    if _wsum > 0:
        w_hh[_rows, _j] *= NORM_LIMIT_INH / _wsum

init_weights = {"in->hid": w_ih.copy(), "hid->hid": w_hh.copy()}

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
# Put I_timed only in the group namespace to avoid a run-namespace conflict
# that Brian2 warns about when the same name appears in both.
_dummy_I = np.zeros((1, N_IN), dtype=float)
G_in.namespace["I_timed"] = TimedArray(_dummy_I, dt=DT_SIM)

# ── Hidden neurons ────────────────────────────────────────────────────────────
#
# Changes vs. original:
#   - Added sigma_noise*xi noise term on dv/dt
#   - Added trace_r state variable (soft refractory): resets to 1 on spike,
#     decays with tau_r; gates incoming excitatory current in S_ih.on_pre
#   - Removed hard refractory (trace_r replaces it)
#
# TO ADD A NEW NEURON GROUP: copy this block, change the name and equations.
# Then register it with the recorder below via recorder.track_group().
#
eqs_h = f"""
dv/dt       = -v / tau_h + sigma_noise * xi                       : 1
dvth/dt     = -(vth - {vth_rest}) / tau_vth                       : 1
dtrace_r/dt = -trace_r / tau_r                                    : 1
"""
G_h = NeuronGroup(
    N_H, eqs_h,
    threshold="v > vth",
    reset=f"v=0; vth=vth+{vth_jump}; trace_r=1;",
    method="euler"   # no refractory — trace_r provides soft refractoriness
)

# ── STDP synapses: input → hidden ─────────────────────────────────────────────
#
# Excitatory current is gated by trace_r_post: v_post += w * (1 - trace_r_post)
# This reduces drive to recently-spiked neurons (soft refractory on the input side).
# STDP weight update is unchanged (not gated).
#
# TO ADD A NEW SYNAPSE GROUP: copy this block, change the name and connect
# to the right groups. Then register it with the recorder below.
#
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
S_ih.connect(i=_src_ih, j=_tgt_ih)
src_ih = np.array(S_ih.i)
tgt_ih = np.array(S_ih.j)

S_ih.wmax_syn  = wmax_matrix[src_ih, tgt_ih]
S_ih.Apre_syn  = Apre_matrix[src_ih, tgt_ih]
S_ih.Apost_syn = Apost_matrix[src_ih, tgt_ih]

# ── Inhibitory lateral synapse: hidden → hidden (tonotopic, STDP) ─────────────
#
# Replaces the former global voltage-clip lat synapse.
# Connectivity is distance-limited (polynomial decay, hard cutoff at R_INH).
# Standard Hebbian STDP: pre-before-post strengthens inhibition,
#                        post-before-pre weakens it.
# Inhibitory current is NOT gated by trace_r (arrives unconditionally).
#
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
S_hh.w_inh         = w_hh[_src_hh, _tgt_hh]   # start at 0

net = Network(G_in, G_h, S_ih, S_hh)


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
recorder.track_group("hidden", G_h)
# recorder.track_group("output", G_out)   # <-- add new groups here

recorder.track_synapses("in->hid",  S_ih, src_ih, tgt_ih)
recorder.track_synapses("hid->hid", S_hh, _src_hh, _tgt_hh)
# recorder.track_synapses("hid->out", S_ho, src_ho, tgt_ho)   # <-- add new synapses here

recorder.build()   # attaches all Brian2 monitors — call once, after all registrations

# ── Periodic L1 normalisation every 50 ms — excitatory and inhibitory ────────
tgt_masks_ih     = [np.where(tgt_ih == j)[0] for j in range(N_H)]
tgt_masks_hh     = [np.where(_tgt_hh == j)[0] for j in range(N_H)]
wmax_syn_arr     = np.array(S_ih.wmax_syn)
wmax_inh_syn_arr = np.array(S_hh.wmax_inh_syn)

@network_operation(dt=500*ms, when='end')
def normalize_weights():
    # exc_sums = np.array([np.array(S_ih.w[tgt_masks_ih[j]]).sum() for j in range(N_H)])
    # inh_sums = np.array([np.array(S_hh.w_inh[tgt_masks_hh[j]]).sum() for j in range(N_H)])
    # print(f"  [norm t={defaultclock.t/ms:.0f}ms] exc col-sum: min={exc_sums.min():.4f} mean={exc_sums.mean():.4f} max={exc_sums.max():.4f} | "
    #       f"inh col-sum: min={inh_sums.min():.4f} mean={inh_sums.mean():.4f} max={inh_sums.max():.4f}")
    for j in range(N_H):
        idx   = tgt_masks_ih[j]
        w_col = np.array(S_ih.w[idx])
        wsum  = w_col.sum()
        if wsum > NORM_LIMIT_EXC:
            S_ih.w[idx] = np.clip(w_col * NORM_LIMIT_EXC / wsum, wmin, wmax_syn_arr[idx])
    for j in range(N_H):
        idx   = tgt_masks_hh[j]
        w_col = np.array(S_hh.w_inh[idx])
        wsum  = w_col.sum()
        if wsum > NORM_LIMIT_INH:
            S_hh.w_inh[idx] = np.clip(w_col * NORM_LIMIT_INH / wsum, W_INH_MIN, wmax_inh_syn_arr[idx])

net.add(normalize_weights)

# Snapshot the clean initial state (clock=0, v=0, a=0, vth=vth_init,
# trace_r=0, apre/apost=0, empty monitors).
# net.restore('init') before every sample brings the network back to this state.
G_h.vth = vth_init   # default Brian2 init is 0; set before store
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
                scale=1.0,
                num_filters=96,
                sustained_per_band=2,
                onset_per_band=1,
                phase_per_band=1,
                sust_gain=0.3,
                onset_gain=2.25,
                phase_gain=0.45,
                sust_spread_min=0.8,
                sust_spread_max=1.0,
            )
        except Exception as e:
            print(f"    Error encoding audio: {e}")
            continue

        duration_s = float(T) * float(DT_SIM)

        # ── Reset to clean state, then inject this sample's data ──────────────
        # restore resets: clock→0, v, a, vth, trace_r, apre, apost,
        #                 and all monitor buffers.
        net.restore('init')

        G_in.namespace["I_timed"] = TimedArray(I.T.astype(float), dt=DT_SIM)

        # Weights survive across samples — override the restored initial weights.
        S_ih.w    = w_ih[src_ih, tgt_ih]
        S_ih.apre  = 0
        S_ih.apost = 0

        # Restore learned inhibitory weights (start at 0 for epoch 0).
        S_hh.w_inh     = w_hh[_src_hh, _tgt_hh]
        S_hh.apre_inh  = 0
        S_hh.apost_inh = 0

        # Recorder tracks elapsed time in Python; since restore resets the
        # Brian2 clock to 0, spike times in the monitor are always [0, T] ms.
        # Reset _elapsed_ms so start_ms = 0 and the slice arithmetic is correct.
        recorder._elapsed_ms = 0.0

        # ── Record: before ─────────────────────────────────────────────────────
        recorder.before_sample(sample_idx)

        # ── Simulate ───────────────────────────────────────────────────────────
        net.run(T * DT_SIM)

        # ── Extract updated weights ────────────────────────────────────────────
        # Both excitatory and inhibitory are L1-normalised every 50 ms.
        w_ih_new = np.zeros((N_IN, N_H))
        w_ih_new[src_ih, tgt_ih] = np.array(S_ih.w)
        w_ih = w_ih_new

        w_hh_new = np.zeros((N_H, N_H))
        w_hh_new[_src_hh, _tgt_hh] = np.array(S_hh.w_inh)
        w_hh = w_hh_new

        # ── Record: after ──────────────────────────────────────────────────────
        # Pass all post-simulation weight matrices you want recorded per-sample.
        recorder.after_sample(
            sample_idx,
            duration_s,
            w_matrices={
                "in->hid":  w_ih,
                "hid->hid": w_hh,
            }
        )

    # ── End of epoch ──────────────────────────────────────────────────────────
    recorder.save_epoch(
        epoch_idx,
        save_dir=SAVE_DIR,
        final_weights={
            "in->hid":  w_ih,
            "hid->hid": w_hh,
        },
        init_weights=init_weights if epoch_idx == 0 else None,
    )


# ============================================================
# Done
# ============================================================

print(f"\n{'='*60}")
print(f"Training complete — {time.time() - start:.2f}s")
print(f"{'='*60}")
