# Result diary

Running record of the fingerprint improvement loop. Newest round at the bottom.
One entry per sweep run — see `PROTOCOL.md` for how the loop works.

---

## Round 0 — where we start (2026-09-02, no sweep run yet)

**Not a sweep round.** This records the state the loop begins from, so later rounds have
something to be measured against. Full detail in `../ecapa_film_snn/RESULTS.md`.

### The baseline numbers

| system | EER |
|---|---|
| ECAPA from audio alone | 0.1328 |
| SNN fingerprint alone (learned readout, held-out speakers) | 0.345 |
| ECAPA + fingerprint, best of two fusion methods | 0.1335 |

Pair-score correlation between ECAPA and the fingerprint: **+0.22**.

### What that means

The fingerprint carries information the audio does not (correlation is low), but it is far too
weak for that to matter. FiLM conditioning gave +0.0014 EER; score-level fusion gave +0.0018
with the weight tuned on test. **Two independent fusion mechanisms agreeing on ~1% means the
limit is the fingerprint itself, not the wiring.** So the fusion work is closed and the loop
attacks the fingerprint.

### The three defects the loop is targeting

From analysing the 16-epoch run in `../tonotopic_plasticity_bound/`:

1. **The hidden layer is more redundant than its own input.** `R_EXC_CHANNEL=11` makes each
   hidden neuron pool 23 of 64 channels, so neighbours overlap 22/23. PC1-5 is 0.856 for
   `hidden_activity` vs 0.684 for `input_activity`. Low rank is why it cannot separate
   thousands of speakers.
2. **19% of the hidden layer is dead** — 24/128 units silent in every epoch, absorbing 18.7% of
   the weight budget and zeroing 18.8% of the fingerprint tensor.
3. **12 of 16 epochs do nothing.** Tonotopic sharpening is 90% complete by epoch 3 and flat
   after epoch 7; after epoch 8 the weights oscillate in place rather than converging.

### Ruled out, do not re-investigate

- **Weight saturation.** Cannot occur — the L1 column budget binds long before the per-synapse
  ceiling, so the classic bimodal STDP signature is impossible here by construction.
- **Epoch-to-epoch churn as a quality problem.** It is real (~0.097 per epoch, never decays) but
  ~7× smaller than between-clip differences. Not what limits the fingerprint.
- **The shared-attractor hypothesis.** Disproven: clips move far from the shared init (0.659)
  and to *different* places (spread 0.644).

### Open question the loop can finally answer

Do same-session utterances converge to nearly the same fingerprint? This is the design's own
premise and **has never been measurable** — the main corpus protocol stores exactly one
utterance per session. `fingerprint_sweep.ipynb` keeps two, so its `SESSION CONVERGENCE` table
is the first look at it.

---

## Round 1 — awaiting results

**Variants queued in the notebook:**

| variant | epochs | R_EXC | vth_rest | hypothesis |
|---|---|---|---|---|
| `baseline` | 16 | 11 | 0.60 | reference |
| `ep4` | 4 | 11 | 0.60 | learning is done by epoch 3, so 4 epochs should match 16 at 4× less cost |
| `ep4_r5` | 4 | 5 | 0.60 | a narrower receptive field decorrelates neighbouring hidden units and raises effective rank |
| `ep4_r5_vth` | 4 | 5 | 0.45 | a lower threshold revives the dead units — and covers the risk that a narrower field makes them *worse* |

**Primary metric:** `PC1-5` (lower = less redundant). EER should follow if the mechanism is right.
**Also watch:** `dead` for `ep4_r5` — if it rises and `ep4_r5_vth` fixes it, the two changes need
each other.

**Status:** not yet run.

<!-- ─────────────────────────────────────────────────────────────────────────────
TEMPLATE for each new round — copy below and fill in.

## Round N — <one-line summary of what was tested> (date)

**Variants:** <table or list>

**Results:**
<paste MAIN COMPARISON and SESSION CONVERGENCE tables>

**Reading:**
- <what moved, what didn't, and what that implies mechanistically>

**Decision:** <promote / next round / stop> — <why>

**Changed for next round:** <what and why>
────────────────────────────────────────────────────────────────────────────── -->
