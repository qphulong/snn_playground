# Improving the SNN fingerprint — plan and test

**Date:** 2026-09-02
**Companion documents:** `../ecapa_film_snn/RESULTS.md` (what was measured and why this plan exists),
`fingerprint_sweep.ipynb` (the experiment that tests this plan).

---

## 1. Where we are

From `../ecapa_film_snn/RESULTS.md`:

| system | EER |
|---|---|
| ECAPA from audio alone | 0.133 |
| SNN fingerprint alone (best learned readout) | 0.345 |
| ECAPA + fingerprint (best of two fusion methods) | 0.1335 |

The fingerprint **does** carry information the audio lacks — the two systems' pair scores
correlate only **+0.22**. But the gain from combining them is ~1%, and two completely
independent fusion mechanisms (FiLM inside the network, score fusion outside it) agree on that
number. That agreement means the ~1% is a property of **the fingerprint itself**, not of how it
is wired in.

**So the fusion is not the problem, and no further fusion work is worth doing.** The fingerprint
has to get stronger. This document says how, based on what the 16-epoch run in
`training/tonotopic_plasticity_bound/` actually shows.

---

## 2. Three measured defects

All three come from analysing `history_epoch_0*.npz` (16 epochs, one 2s clip).

### Defect 1 — the hidden layer is more redundant than its own input

`R_EXC_CHANNEL = 11` means every hidden neuron pools **23 of the 64 channels**. Two neighbouring
hidden neurons therefore share **22 of those 23** — 96% overlap. The 128 hidden units are not 128
different views of the sound; they are a handful of genuinely distinct views, smeared across 128
outputs.

The measurement confirms it. Fraction of variance in the top 5 principal components:

| | PC1–5 |
|---|---|
| `input_activity` (the layer's *input*) | 0.684 |
| `hidden_activity` (the layer's *output*) | **0.856** |

**The hidden layer comes out more redundant than what went in.** Pooling is destroying rank, and
low rank is exactly why a fingerprint cannot separate thousands of speakers — there are only a
few independent numbers in it.

This is the most important defect and the one most likely to move the 0.345.

### Defect 2 — 19% of the hidden layer is dead

24 of 128 hidden neurons are silent in **every single epoch** (plus 17/128 input neurons). They
absorb 18.7% of the excitatory weight budget and zero out 18.8% of the `(2, 23, 128)` fingerprint
tensor. Nearly a fifth of the representation is structurally empty before any speaker information
is even considered.

### Defect 3 — twelve of the sixteen epochs do nothing

Tonotopic sharpening, measured as the rms channel-spread of each column's weight mass:

```
init 5.160  →  e0 4.874  →  e3 4.612  →  e7 4.562  →  e15 4.549
```

90% of the sharpening happens in the first three epochs and it is completely flat after epoch 7.
Similarity to the final weights: epoch 4 already reaches cos 0.970, epoch 6 reaches 0.979.

After epoch 8 the weights stop converging and start **oscillating in place**: per-epoch change
plateaus at 0.097 and never decays, and over epochs 8→15 the summed steps (1.64) are far larger
than the net displacement (0.45) — a ratio of 0.28, below even the 0.38 a pure random walk would
give.

This is not itself a quality problem (that oscillation is ~7× smaller than the differences
between clips, so it is not what limits the fingerprint). It is a **cost** problem: 12 of 16
epochs are bought and not used.

### Not a defect: weight saturation

Worth recording so it is not chased. `frac > 90% of wmax` is exactly 0.000 at every epoch, and
the strongest synapse reaches only 0.168 of `wmax = 1.0`. That is not a failure to converge — the
per-synapse ceiling is simply never the binding constraint. The L1 column budget is
(`NORM_LIMIT_EXC = 1.0455` shared across ~46 synapses, mean 0.0227). The classic bimodal 0/wmax
STDP signature **cannot** form in this architecture by construction.

---

## 3. The three changes

| # | change | fixes | rationale |
|---|---|---|---|
| 1 | `R_EXC_CHANNEL` 11 → ~5 | defect 1 | Each hidden neuron pools 11 channels instead of 23; neighbours decorrelate; effective rank rises. The L1 column normalisation keeps the column sum at `NORM_LIMIT_EXC` regardless of fan-in, so total drive is roughly unchanged — fewer inputs, each proportionally stronger. |
| 2 | lower `vth_rest` 0.6 → ~0.45 | defect 2 | Cheapest test of whether the dead units are a threshold problem. If it works, replace it with proper intrinsic homeostatic plasticity (threshold that falls when a neuron's rate is below target) rather than a hand-set constant. |
| 3 | `N_EPOCHS` 16 → 4 | defect 3 | Structural learning is finished by epoch 3. Buys a **4× cheaper pipeline** (32s → 8s of simulated time per wav, across ~167k wavs), which is what makes every experiment after this one affordable. |

Change 1 carries a real risk worth naming: a narrower receptive field means a hidden neuron
hears only 11 channels, so it may fall silent when those channels are quiet — **it could make
defect 2 worse**. That is precisely why changes 1 and 2 are tested together as well as separately.

---

## 4. How this gets tested

`fingerprint_sweep.ipynb` runs four variants over the *same* set of clips, so every comparison is
paired:

| variant | epochs | R | vth_rest | question it answers |
|---|---|---|---|---|
| `baseline` | 16 | 11 | 0.60 | reference — reproduces current behaviour |
| `ep4` | 4 | 11 | 0.60 | is 4 epochs really as good as 16? |
| `ep4_r5` | 4 | 5 | 0.60 | does a narrower receptive field raise the rank? |
| `ep4_r5_vth` | 4 | 5 | 0.45 | does lowering the threshold recover the dead units? |

Metrics reported per variant:

- **cross-session retrieval EER** — the headline; same protocol as everywhere else
- **PC1–5 / PC1–20** — effective rank; the thing defect 1 is about. *Lower is better.*
- **dead hidden fraction** — defect 2
- **Fisher ratio F** — between-speaker vs within-speaker variance (1.0 = no speaker structure)
- **within-session vs between-session similarity** — see below
- wall-clock cost per wav

### The bonus measurement

The sweep keeps **several utterances per session**, which the main corpus protocol does not (it
stores exactly one). That makes it the first opportunity to measure the design's own core claim:
**do utterances from the same session converge to nearly the same fingerprint?** The table reports
mean similarity for same-session pairs, different-session same-speaker pairs, and different-speaker
pairs. If same-session similarity is much higher than different-session similarity, the claim
holds — and session-level averaging becomes available as free denoising.

---

## 5. Decision gate — set this before looking at the numbers

Baseline to beat: **fingerprint-alone EER 0.345** (from `../ecapa_film_snn/RESULTS.md`, learned readout on
held-out speakers). The sweep's unsupervised numbers are not directly comparable to that, so
judge the sweep on its **own `baseline` row**:

- **A variant beats `baseline` EER by a clear margin AND lowers PC1–5** → the diagnosis was right.
  Regenerate a full shard with that setting, re-run `../ecapa_film_snn/ecapa_film_snn.ipynb` cell 14 Part B, and
  check whether fingerprint-alone EER drops below **0.28**. If it does, the direction is alive:
  re-run the fusion test (cell 15), then a short FiLM run.
- **Nothing moves** → the fingerprint's weakness is not fan-in, dead units, or exposure length.
  At that point the honest call is to stop pursuing SNN-fingerprint conditioning for ECAPA, and
  the ~1% stands as the measured answer.

The point of writing the gate down now is that 1% effects are easy to keep chasing.

---

## 6. What not to do

- **No more full ECAPA training runs** until fingerprint-alone EER improves. Cells 14 and 15 are
  the cheap proxies and answer the same question in minutes rather than 14 hours.
- **No multi-seed sweep.** With seed noise plausibly ~0.002 and the effect at 0.0014, resolving it
  would take ~16 paired seeds (~110 GPU-hours) to measure precisely something already known to be
  ~1%.
- **No per-component FiLM ablations**, for the same reason.
- **Don't chase weight saturation.** It cannot happen here (§2), so its absence is not evidence
  of anything.

---

## 7. If the sweep succeeds — what comes after

1. Replace the hand-set `vth_rest` with **intrinsic homeostatic plasticity**: let each neuron's
   threshold fall while its rate is below a target. That fixes dead units in a way that
   generalises across clips instead of being tuned to one.
2. **Session-averaged fingerprints**, if the same-session convergence measurement supports it.
3. Only then revisit fusion — and if the fingerprint is materially stronger, re-run the
   FiLM-vs-no-FiLM comparison properly, at ~15–20 epochs (both current runs overfit from epoch 8,
   see `../ecapa_film_snn/RESULTS.md` §2).
