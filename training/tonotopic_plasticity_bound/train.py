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
# Dataset — one clip, sliced into overlapping pieces (see "Piece framing"
# below); each piece trains independently from the SAME shared init weights.
# ============================================================

AUDIO_PATH = "datasets/vox1/dev_01/id10002/C7k7C-PDvAA/00002.m4a"
print(f"Audio: {AUDIO_PATH}")

# -- Piece framing --
WINDOW_MS = 200   # piece length
HOP_MS    = 100   # slide between consecutive pieces (50% overlap)
N_REPEATS = 8    # repeated exposures per piece == "samples" within its epoch

# -- Input encoding gains (passed to compute_spike_input_current) --
SUST_GAIN  = 0.3    # sustained-energy neurons
ONSET_GAIN = 3.0    # onset (transient) neurons — bumped up from 2.25 to fire more;
                     # check vizs/global/raster_input_full_clip.png (crimson) and retune
PHASE_GAIN = 1.0    # phase-locking neurons (unused: phase_per_band=0 below)

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Hyperparameters
# ============================================================

N_IN = 128   # 64 channels × 2 neurons/channel (sustained + onset, no phase)
N_H  = 128

DT_SIM = 1 * ms

# -- Input layer (adaptive LIF) --
tau_m       = 40 * ms
tau_a       = 40 * ms    # was 200ms (which was itself bumped from 100ms) — 200ms barely
                          # decayed between ANY spikes, even 20Hz-spaced ones, collapsing
                          # them too. 40ms sits between the 20Hz (~50ms ISI, mostly decays)
                          # and 50Hz (~20ms ISI, compounds) regimes for rate-selective adaptation.
beta        = 3.5        # was 1.75 — tau_a=40ms fixed selectivity (20Hz no longer
                          # collapses) but the absolute drag at beta=1.75 was too small to
                          # slow 50Hz down; doubling preserves the tau-driven ratio while
                          # raising the absolute suppression on the high-rate regime
tau_current = 1 * ms
v_th_in     = 1.0

# -- Hidden layer (adaptive-threshold LIF) --
tau_h    = 50 * ms
tau_vth  = 60 * ms    # was 180ms (itself bumped from 100ms) — same over-persistence issue
                       # as tau_a; shortened so it stays rate-selective rather than a flat
                       # per-piece spike-count penalty. Given a bit more slack than tau_a
                       # since hidden output is naturally sparser than input drive.
vth_rest = 0.8
vth_init = 0.8
vth_jump = 1.0        # was 0.5 — same reasoning as beta above: tau_vth=60ms fixed
                       # selectivity, doubling the amplitude raises absolute suppression
                       # on the high-rate regime without changing the selectivity ratio

# -- Soft refractory (hidden only; replaces hard refractory) --
tau_r = 10 * ms

# -- Membrane noise (hidden only) --
# sigma_noise has units second**(-0.5) so sigma_noise*xi has units 1/second,
# matching dv/dt for a dimensionless variable.
sigma_noise = 0.03 * second**(-0.5)

# -- STDP (excitatory) --
taupre  = 20 * ms
taupost = 20 * ms

# -- Triplet STDP (excitatory; Pfister-Gerstner, added on top of the pair rule) --
# Slow detector traces r2 (pre) and o2 (post) make LTP/LTD depend on a recent
# *previous* spike of the same kind (frequency-dependent, BCM-like).
# Set both A3 centres to 0 to recover the plain pair-STDP behaviour exactly.
tau_x         = 100 * ms   # slow presynaptic detector r2
tau_y         = 125 * ms   # slow postsynaptic detector o2
A3PRE_CENTER  =  0.004     # triplet LTP amplitude (pre-post-post), > 0
A3POST_CENTER = -0.002     # triplet LTD amplitude (post-pre-pre), < 0

# -- Excitatory weight bounds --
wmin = 0.0

# -- Excitatory synapse --
WMAX_CENTER  = 1.0
APRE_CENTER  =  0.008    # was 0.006 (before that 0.004) — raised further now that both
APOST_CENTER = -0.0096   # rate regimes have comparably-bounded, non-collapsing spike
                          # budgets (tau_a/tau_vth fix above), so a bump applies evenly
                          # instead of favoring one regime (ratio kept ~1.2)

# -- Inhibitory lateral synapse --
W_INH_CENTER = 1.0
W_INH_MIN    = 0.0
APRE_INH     = 0.004
APOST_INH    = -0.0048

# -- Channel layout --
N_CHANNELS    = 64
N_PER_CHANNEL = N_IN // N_CHANNELS   # 2

# -- Tonotopic plasticity (polynomial decay: max(0, 1-(d_channel/R)^p)) --
R_EXC_CHANNEL = 11
p_EXC         = 3

# -- Homeostatic normalisation --
NORM_LIMIT_EXC = 1.0455
NORM_LIMIT_INH = 0.40


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

wmax_matrix   = WMAX_CENTER   * _topo_exc
Apre_matrix   = APRE_CENTER   * _topo_exc
Apost_matrix  = APOST_CENTER  * _topo_exc
A3pre_matrix  = A3PRE_CENTER  * _topo_exc   # triplet LTP amplitude (topo-scaled)
A3post_matrix = A3POST_CENTER * _topo_exc   # triplet LTD amplitude (topo-scaled)
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
# Initialise weight matrices (shared init — every piece restores to this)
# ============================================================

# Excitatory: formula-shaped, column-normalised to NORM_LIMIT_EXC
w_ih_init = np.zeros((N_IN, N_H))
w_ih_init[_src_ih, _tgt_ih] = wmax_matrix[_src_ih, _tgt_ih]
for _j in range(N_H):
    _col_mask = (_tgt_ih == _j)
    _rows = _src_ih[_col_mask]
    _wsum = w_ih_init[_rows, _j].sum()
    if _wsum > 0:
        w_ih_init[_rows, _j] *= NORM_LIMIT_EXC / _wsum

# Inhibitory: uniform random init in [0.01, 0.02] on connected positions
w_hh_init = np.zeros((N_H, N_H))
w_hh_init[_src_hh, _tgt_hh] = np.random.uniform(0.01, 0.02, size=_src_hh.shape[0])

init_weights = {"in->hid": w_ih_init.copy(), "hid->hid": w_hh_init.copy()}

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
dr1/dt     = -r1 / taupre      : 1 (event-driven)
dr2/dt     = -r2 / tau_x       : 1 (event-driven)
do1/dt     = -o1 / taupost     : 1 (event-driven)
do2/dt     = -o2 / tau_y       : 1 (event-driven)
wmax_syn   : 1
Apre_syn   : 1
Apost_syn  : 1
A3pre_syn  : 1
A3post_syn : 1
"""
# Pair terms kept verbatim; triplet term appended. Slow traces (r2, o2) are read
# BEFORE being incremented, so they carry the value from the *previous* spike.
#   on_pre  LTD: pair apost*(w-wmin)     + triplet A3post_syn*o1*r2*(w-wmin)
#   on_post LTP: pair apre*(wmax_syn-w)  + triplet A3pre_syn *r1*o2*(wmax_syn-w)
on_pre  = ("v_post += w * (1 - trace_r_post)\n"
           "apre += Apre_syn\n"
           "w = clip(w + apost*(w-wmin) + A3post_syn*o1*r2*(w-wmin), wmin, wmax_syn)\n"
           "r1 += 1\n"
           "r2 += 1")
on_post = ("apost += Apost_syn\n"
           "w = clip(w + apre*(wmax_syn-w) + A3pre_syn*r1*o2*(wmax_syn-w), wmin, wmax_syn)\n"
           "o1 += 1\n"
           "o2 += 1")

S_ih = Synapses(G_in, G_h, model=stdp_model, on_pre=on_pre, on_post=on_post)
S_ih.connect(i=_src_ih, j=_tgt_ih)
src_ih = np.array(S_ih.i)
tgt_ih = np.array(S_ih.j)

S_ih.wmax_syn   = wmax_matrix[src_ih, tgt_ih]
S_ih.Apre_syn   = Apre_matrix[src_ih, tgt_ih]
S_ih.Apost_syn  = Apost_matrix[src_ih, tgt_ih]
S_ih.A3pre_syn  = A3pre_matrix[src_ih, tgt_ih]
S_ih.A3post_syn = A3post_matrix[src_ih, tgt_ih]

# ── Inhibitory lateral synapse: hidden → hidden (tonotopic, STDP) ─────────────
#
# Replaces the former global voltage-clip lat synapse.
# Connectivity is distance-limited (polynomial decay, hard cutoff at R_INH).
# Standard Hebbian STDP: pre-before-post strengthens inhibition,
#                        post-before-pre weakens it.
# Inhibitory current is gated by trace_r_post: v_post -= w_inh * (1 - trace_r_post),
# matching the excitatory soft-refractory gating (less drive to recently-spiked neurons).
#
stdp_inh_model = """
w_inh          : 1
dapre_inh/dt   = -apre_inh  / taupre  : 1 (event-driven)
dapost_inh/dt  = -apost_inh / taupost : 1 (event-driven)
wmax_inh_syn   : 1
Apre_inh_syn   : 1
Apost_inh_syn  : 1
"""
on_pre_inh  = (f"v_post -= w_inh * (1 - trace_r_post)\n"
               f"apre_inh += Apre_inh_syn\n"
               f"w_inh = clip(w_inh + apost_inh*(w_inh-{W_INH_MIN}), {W_INH_MIN}, wmax_inh_syn)")
on_post_inh = (f"apost_inh += Apost_inh_syn\n"
               f"w_inh = clip(w_inh + apre_inh*(wmax_inh_syn-w_inh), {W_INH_MIN}, wmax_inh_syn)")

S_hh = Synapses(G_h, G_h, model=stdp_inh_model, on_pre=on_pre_inh, on_post=on_post_inh)
S_hh.connect(i=_src_hh, j=_tgt_hh)

S_hh.wmax_inh_syn  = wmax_inh_matrix[_src_hh, _tgt_hh]
S_hh.Apre_inh_syn  = Apre_inh_matrix[_src_hh, _tgt_hh]
S_hh.Apost_inh_syn = Apost_inh_matrix[_src_hh, _tgt_hh]
S_hh.w_inh         = w_hh_init[_src_hh, _tgt_hh]   # uniform [0.01, 0.02]

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

@network_operation(dt=100*ms, when='end')
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
# Encode the whole clip once, then slice into WINDOW_MS/HOP_MS pieces.
# ============================================================

try:
    I_sim, T_sim = compute_spike_input_current(
        AUDIO_PATH,
        scale=1.0,
        num_filters=64,
        sustained_per_band=1,
        onset_per_band=1,
        phase_per_band=0,
        sust_gain=SUST_GAIN,
        onset_gain=ONSET_GAIN,
        sust_spread_min=1,
        sust_spread_max=1,
    )
except Exception as e:
    raise RuntimeError(f"Error encoding audio {AUDIO_PATH}: {e}") from e

print(f"Clip length: {T_sim} ms")

N_PIECES = (T_sim - WINDOW_MS) // HOP_MS + 1
if N_PIECES < 1:
    raise ValueError(f"Clip ({T_sim} ms) shorter than one window ({WINDOW_MS} ms)")
print(f"{N_PIECES} pieces of {WINDOW_MS}ms (hop {HOP_MS}ms), {N_REPEATS} repeats each")


# ============================================================
# Global input-only preview pass (no learning, no hidden layer).
#
# Runs the WHOLE clip once through an isolated probe copy of the input
# layer, before any piece-wise training starts. This never touches G_in,
# G_h, S_ih, S_hh, or the 'init' snapshot used by training — it is a
# separate NeuronGroup/Network built solely to produce a global raster.
# ============================================================

G_in_probe = NeuronGroup(
    N_IN, eqs_in,
    threshold="v > v_th_in",
    reset="v=0; a+=beta",
    refractory=2 * ms,
    method="euler"
)
G_in_probe.namespace["I_timed"] = TimedArray(I_sim.T.astype(float), dt=DT_SIM)
mon_probe = SpikeMonitor(G_in_probe)
Network(G_in_probe, mon_probe).run(T_sim * DT_SIM)

np.savez_compressed(
    os.path.join(SAVE_DIR, "global_input_raster.npz"),
    raster_i=np.array(mon_probe.i, dtype=np.int32),
    raster_t=np.array(mon_probe.t / ms, dtype=np.float32),
    n_neurons=np.int32(N_IN),
    duration_ms=np.float32(T_sim),
)
print(f"Global input raster: {len(mon_probe.i)} spikes over {T_sim}ms "
      f"-> global_input_raster.npz")


# ============================================================
# Training loop — pieces, then each piece's own epochs.
#
# Every piece restores to the SAME shared init weights (never carried over
# from the previous piece). Within a piece, each repeated exposure is its
# own recorder epoch — its own history_epoch_NNN.npz, with NNN numbered
# from 0 independently for every piece (piece_0000/epoch_000..031,
# piece_0001/epoch_000..031, ...). The underlying Brian2 clock is NOT reset
# between repeats within a piece (no net.restore there), so neuron/synapse
# state carries over continuously — only the recorder's per-epoch
# bookkeeping resets, so each repeat's plots read as [0, WINDOW_MS) local time.
# ============================================================

for piece_idx in range(N_PIECES):
    t_start = piece_idx * HOP_MS
    t_end   = t_start + WINDOW_MS

    print(f"\n{'='*60}")
    print(f"Piece {piece_idx}/{N_PIECES - 1}  t=[{t_start},{t_end})ms")
    print(f"{'='*60}")

    piece_save_dir = os.path.join(SAVE_DIR, f"piece_{piece_idx:04d}")

    I_piece = I_sim[:, t_start:t_end]                 # (N_IN, WINDOW_MS)
    I_tiled = np.tile(I_piece, (1, N_REPEATS))         # (N_IN, WINDOW_MS * N_REPEATS)

    # ── Reset to the shared clean init for this piece ─────────────────────────
    # restore resets: clock→0, v, a, vth, trace_r, apre, apost,
    #                 and all monitor buffers.
    net.restore('init')

    G_in.namespace["I_timed"] = TimedArray(I_tiled.T.astype(float), dt=DT_SIM)

    w_ih = w_ih_init.copy()
    w_hh = w_hh_init.copy()
    S_ih.w    = w_ih[src_ih, tgt_ih]
    S_ih.apre  = 0
    S_ih.apost = 0
    S_ih.r1 = 0; S_ih.r2 = 0; S_ih.o1 = 0; S_ih.o2 = 0   # triplet detectors

    S_hh.w_inh     = w_hh[_src_hh, _tgt_hh]
    S_hh.apre_inh  = 0
    S_hh.apost_inh = 0

    for rep in range(N_REPEATS):
        print(f"  [piece {piece_idx}, epoch {rep}/{N_REPEATS - 1}]")

        # Each repeat is its own recorder epoch -> its own npz file. The
        # Brian2 clock keeps advancing through I_tiled (no restore here), so
        # we anchor _elapsed_ms to the clock's actual position instead of
        # letting reset_epoch() zero it, keeping this epoch's recorded
        # times relative to ITS OWN start rather than the piece's start.
        recorder.reset_epoch()
        recorder._elapsed_ms = float(rep * WINDOW_MS)

        # ── Record: before ─────────────────────────────────────────────────────
        recorder.before_sample(0)

        # ── Simulate this repeat. No restore between repeats — state carries
        #    over, giving continuous repeated exposure rather than N resets.
        net.run(WINDOW_MS * DT_SIM)

        # ── Extract updated weights ────────────────────────────────────────────
        # Both excitatory and inhibitory are L1-normalised every 100 ms.
        w_ih_new = np.zeros((N_IN, N_H))
        w_ih_new[src_ih, tgt_ih] = np.array(S_ih.w)
        w_ih = w_ih_new

        w_hh_new = np.zeros((N_H, N_H))
        w_hh_new[_src_hh, _tgt_hh] = np.array(S_hh.w_inh)
        w_hh = w_hh_new

        # ── Record: after ──────────────────────────────────────────────────────
        recorder.after_sample(
            0,
            WINDOW_MS / 1000.0,
            w_matrices={
                "in->hid":  w_ih,
                "hid->hid": w_hh,
            }
        )

        # ── Save this repeat as its own epoch, in this piece's own folder ──────
        recorder.save_epoch(
            rep,
            save_dir=piece_save_dir,
            final_weights={
                "in->hid":  w_ih,
                "hid->hid": w_hh,
            },
            init_weights=init_weights if rep == 0 else None,
        )

    # ── Save this piece's continuous membrane-potential trace (spans all
    #    N_REPEATS repeats) once, rather than duplicating it into every
    #    repeat's history_epoch_NNN.npz ────────────────────────────────────────
    recorder.save_piece_membrane(piece_save_dir)


# ============================================================
# Done
# ============================================================

print(f"\n{'='*60}")
print(f"Training complete — {time.time() - start:.2f}s")
print(f"{'='*60}")
