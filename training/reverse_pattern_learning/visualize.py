"""
visualize.py  -  Load snn_history.pkl and produce plots
========================================================
Reads:   training/reverse_pattern_learning/snn_history.pkl
Saves:   training/reverse_pattern_learning/*.png
         training/reverse_pattern_learning/neurons_sample/sampleX_neuronY.png
"""

import os
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SAMPLE_DUR = 10.0   # ms – must match train.py

DATA_DIR    = os.path.join('training', 'reverse_pattern_learning')
NEURON_DIR  = os.path.join(DATA_DIR, 'neurons_sample')
PKL_PATH    = os.path.join(DATA_DIR, 'snn_history.pkl')

os.makedirs(NEURON_DIR, exist_ok=True)

def save(fig, path):
    fig.savefig(path, dpi=150)
    print(f'Saved -> {path}')

# ── load ────────────────────────────────────────────────────
with open(PKL_PATH, 'rb') as f:
    H = pickle.load(f)

sequence   = H['sequence']
weight_arr = np.array(H['weights'])   # (n_samples + 1, 4)
n_samples  = len(sequence)
x_axis     = np.arange(n_samples + 1)

print(f'Loaded {n_samples} samples from {PKL_PATH}')

# ── colours ─────────────────────────────────────────────────
C_wCA = '#e63946'
C_wCB = '#457b9d'
C_wDA = '#e9c46a'
C_wDB = '#2a9d8f'
C_AB  = 'steelblue'
C_BA  = 'tomato'
C_vm  = '#2196F3'
C_thr = '#FF5722'
C_spk = '#4CAF50'
C_A   = '#9C27B0'
C_B   = '#FF9800'

# ============================================================
# PLOT 1 – Weight evolution
# ============================================================

fig_w, ax_w = plt.subplots(figsize=(13, 5))

labels_w = [r'$w_{CA}$', r'$w_{CB}$', r'$w_{DA}$', r'$w_{DB}$']
colors_w = [C_wCA, C_wCB, C_wDA, C_wDB]

for i, (col, lbl) in enumerate(zip(colors_w, labels_w)):
    ax_w.plot(x_axis, weight_arr[:, i], color=col, lw=1.8, label=lbl)

for s, pat in enumerate(sequence):
    ax_w.axvspan(s + 0.5, s + 1.5, alpha=0.07,
                 color=C_AB if pat == 'AB' else C_BA)

handles, _ = ax_w.get_legend_handles_labels()
handles += [Patch(color=C_AB, alpha=0.35, label='AB sample'),
            Patch(color=C_BA, alpha=0.35, label='BA sample')]
ax_w.legend(handles=handles, loc='upper right', fontsize=9)
ax_w.set_xlabel('Sample index  (0 = initial weights)')
ax_w.set_ylabel('Weight value')
ax_w.set_title('Weight evolution across training')
ax_w.set_xlim(0, n_samples)
ax_w.set_ylim(-0.05, 1.10)
ax_w.grid(alpha=0.3)
fig_w.tight_layout()
save(fig_w, os.path.join(DATA_DIR, 'weight_evolution.png'))

# ============================================================
# PLOT 2 – Final weight bar chart
# ============================================================

fig_b, ax_b = plt.subplots(figsize=(6, 4))
final_w = weight_arr[-1]
bars = ax_b.bar(labels_w, final_w, color=colors_w,
                width=0.5, edgecolor='k', linewidth=0.8)
for bar, val in zip(bars, final_w):
    ax_b.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
              f'{val:.3f}', ha='center', va='bottom', fontsize=9)
ax_b.set_ylim(0, 1.15)
ax_b.set_ylabel('Weight value')
ax_b.set_title('Final weights after training')
ax_b.grid(axis='y', alpha=0.3)
fig_b.tight_layout()
save(fig_b, os.path.join(DATA_DIR, 'final_weights.png'))

# ============================================================
# PLOT 3 – Membrane potential for ALL samples × ALL neurons
#           saved as neurons_sample/sampleX_neuronY.png
# ============================================================

NEURON_MAP = {0: 'C', 1: 'D'}

for sample_idx in range(n_samples):
    pat_lbl = sequence[sample_idx]
    t_rel   = H['t_trace'][sample_idx]
    spk_A   = H['spikes_A'][sample_idx]
    spk_B   = H['spikes_B'][sample_idx]

    for neuron_idx, nname in NEURON_MAP.items():
        vm   = H[f'vm_{nname}'][sample_idx]
        thr  = H[f'theta_{nname}'][sample_idx]
        spks = H[f'spikes_{nname}'][sample_idx]

        fig_v, (ax_v, ax_inp) = plt.subplots(
            2, 1, figsize=(10, 6), sharex=True,
            gridspec_kw={'height_ratios': [3, 1]},
        )

        ax_v.plot(t_rel, vm,  color=C_vm,  lw=1.8, label=f'$V_m$ (neuron {nname})')
        ax_v.plot(t_rel, thr, color=C_thr, lw=1.5, ls='--',
                  label=r'Adaptive threshold $\vartheta$')

        if len(spks):
            for st in spks:
                ax_v.axvline(st, color=C_spk, lw=1.2, alpha=0.8)
            ax_v.scatter(spks, np.full(len(spks), float(thr.max()) * 1.08),
                         marker='v', color=C_spk, s=70, zorder=5, label='Output spike')

        ax_v.axhline(1.0, color='grey', lw=0.8, ls=':', alpha=0.6,
                     label='Init. threshold (1.0)')
        ax_v.set_ylabel('Potential / Threshold (a.u.)')
        ax_v.set_title(
            f'Neuron {nname}  -  Sample {sample_idx + 1} / {n_samples}  '
            f'(pattern: {pat_lbl})'
        )
        ax_v.legend(fontsize=8, loc='upper right')
        ax_v.grid(alpha=0.3)

        for st in spk_A:
            ax_inp.axvline(st, color=C_A, lw=2.5,
                           label='A spike' if st == spk_A[0] else '')
        for st in spk_B:
            ax_inp.axvline(st, color=C_B, lw=2.5,
                           label='B spike' if st == spk_B[0] else '')
        ax_inp.set_yticks([])
        ax_inp.set_ylabel('Inputs')
        ax_inp.set_xlabel('Time within sample (ms)')
        ax_inp.legend(fontsize=8, loc='upper right')
        ax_inp.set_xlim(0, SAMPLE_DUR)
        ax_inp.grid(alpha=0.2)

        fig_v.tight_layout()
        fname = f'sample{sample_idx + 1}_neuron{nname}.png'
        save(fig_v, os.path.join(NEURON_DIR, fname))
        plt.close(fig_v)

# ============================================================
# PLOT 4 – Per-pattern spike count summary  (improved x-axis)
# ============================================================

cnt_C   = np.array([len(s) for s in H['spikes_C']])
cnt_D   = np.array([len(s) for s in H['spikes_D']])
ab_mask = np.array(sequence) == 'AB'
ba_mask = ~ab_mask

fig_s, axes_s = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, mask, title in zip(
    axes_s, [ab_mask, ba_mask], ['AB samples', 'BA samples']
):
    xs = np.where(mask)[0] + 1          # 1-based sample indices

    # One bar per sample, side-by-side C / D
    ax.bar(xs - 0.2, cnt_C[mask], width=0.4, color=C_wCA,
           label='C spikes', edgecolor='k', linewidth=0.5)
    ax.bar(xs + 0.2, cnt_D[mask], width=0.4, color=C_wDB,
           label='D spikes', edgecolor='k', linewidth=0.5)

    # Draw a thin vertical separator between every sample
    for x in xs:
        ax.axvline(x + 0.5, color='grey', lw=0.5, alpha=0.4, zorder=0)

    # X-axis: tick at every sample, label every 5th
    ax.set_xticks(xs)
    tick_labels = [str(x) if x % 5 == 0 else '' for x in xs]
    ax.set_xticklabels(tick_labels, fontsize=8)

    # Minor ticks (unlabelled) at every sample for the grid lines
    ax.tick_params(axis='x', which='major', length=4)

    # Annotate every 5th sample with its index above the taller bar
    for x, c, d in zip(xs, cnt_C[mask], cnt_D[mask]):
        if x % 5 == 0:
            top = max(c, d)
            ax.text(x, top + 0.05, str(x),
                    ha='center', va='bottom', fontsize=7,
                    color='dimgrey', fontweight='bold')

    ax.set_xlim(xs[0] - 0.8, xs[-1] + 0.8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Sample index', fontsize=10)
    ax.set_ylabel('Spike count per sample', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

fig_s.suptitle('Output neuron spike counts  (C should win AB,  D should win BA)',
               fontsize=12, fontweight='bold')
fig_s.tight_layout()
save(fig_s, os.path.join(DATA_DIR, 'spike_counts.png'))

plt.close('all')
print(f'\nDone.')
print(f'  - Membrane plots: {NEURON_DIR}/sampleX_neuronY.png  ({n_samples * 2} files)')
print(f'  - Summary plots:  {DATA_DIR}/')