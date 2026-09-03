# ECAPA + SNN FiLM — results report

**Date:** 2026-09-02
**Scope:** everything measured in `training/ecapa_film_snn/` to date — two full 80-epoch
training runs plus four diagnostic experiments run from temporary notebook cells.

---

## 1. Summary (read this first)

**The question:** does the STDP fingerprint from the SNN give ECAPA-TDNN speaker information
that the audio alone does not have?

**The answer:** yes, technically — and it does not matter in practice.

Three findings, in order of importance:

1. **The fingerprint genuinely knows something different from the audio.** When we scored the
   same pairs of clips with ECAPA and with the fingerprint, the two systems' scores correlated
   only **+0.22**. They are not looking at the same thing. This part of the hypothesis is
   confirmed, not rejected.

2. **But it buys about 1% and no more.** Two completely different ways of combining the
   fingerprint with ECAPA both landed on the same tiny number:
   - FiLM conditioning inside the network: **0.0014** better EER
   - score-level fusion outside the network: **0.0018** better EER (and that one was tuned on
     the test set, so it is the optimistic version)

   Because two independent mechanisms agree, the ~1% is a property of **the fingerprint
   itself**, not of how it is wired in. There is no cleverer fusion waiting to be found.

3. **The reason is that the fingerprint is a weak speaker detector.** On its own it reaches
   EER 0.345, where ECAPA from raw audio reaches 0.133. Information that is *different but
   weak* contributes very little no matter how you combine it.

**What to do with this:** stop tuning the fusion. If this direction is to work, the fingerprint
itself has to get substantially stronger. See §8 for the two concrete leads.

**One number to remember:** ECAPA alone 0.133 → ECAPA + fingerprint 0.1335. That is the whole
effect.

### Reading the metrics

| metric | meaning | better is |
|---|---|---|
| **EER** | Equal Error Rate. The error level where false accepts equal false rejects. 0.5 = coin flip, 0.0 = perfect. | lower |
| **R@1 / R@5** | Of all query clips, the fraction whose top-1 (top-5) nearest neighbour is the same speaker. | higher |
| **mAP** | Mean average precision over the whole ranked list — rewards putting *all* of a speaker's clips near the top. | higher |

All evaluations use the same protocol: session-free all-pairs cosine/L1 retrieval on VoxCeleb1,
speakers never seen in training.

---

## 2. Experiment 1 — FiLM vs no-FiLM, two full training runs

Both runs on the current fp32-pooling code, so this is the first apples-to-apples comparison.
(An older `USE_FILM=False` number of 0.2635 exists but came from the old fp16-pooling code and
is **not** comparable — ignore it.)

Corpus: 150480 train clips / 6112 speakers (VoxCeleb2), 22496 eval clips / 1251 speakers
(VoxCeleb1). 0 missing files, 0 decode failures, 0 speaker overlap.

| | no-FiLM | FiLM (all 4 components) |
|---|---|---|
| best EER | **0.1328** @ epoch 8 | **0.1314** @ epoch 8 |
| R@1 / R@5 / mAP at best | 0.439 / 0.675 / 0.161 | 0.445 / 0.680 / 0.163 |
| mean EER over epochs 5–8 | 0.1340 | 0.1324 |
| parameters | 15.44M | 16.67M (+1.22M in FiLM) |
| time per epoch | 605s | 622s (+3%) |

**Difference: 0.0014 EER, about 1% relative.** FiLM won 36 of 53 epochs, and won consistently
from epoch 9 onward — but that whole region is inside the overfitting collapse, so the most
likely explanation there is mild regularisation from 1.2M extra parameters, not identity
information.

This single number is not trustworthy on its own: it is one seed, and `best_eer` is selected by
taking the minimum over 80 epochs *on the same set it reports*. Selection noise alone is larger
than 0.0014.

### A serious problem visible in both runs: overfitting

Both runs peak at **epoch 8 of 80**, then get monotonically worse — train loss falls to 0.01 by
epoch 26 while eval EER climbs from 0.133 to 0.20. About 70 of the 80 epochs are wasted, and
harmful.

Cause: there is **no augmentation at all**, and `CROP_SEC == CLIP_SEC` means even random
crop-position jitter is a no-op. Every epoch sees a byte-identical 2-second window of every clip.
The 0.97/epoch × 80 schedule is inherited from the reference recipe, which relies on
augmentation this run does not have.

Also visible: **periodic loss explosions** roughly every 5 epochs once train loss reaches ~0.01
(no-FiLM at e26/27, e31/32, e36/37, e42, e47/48, e53; FiLM at e25/26, e29/30, e34/35, e39/40,
e44/45, e50). Each spike *temporarily improves* EER — e.g. no-FiLM epoch 26 gives 0.1733 against
~0.190 either side. That is the network being kicked out of a memorised minimum, and it is more
evidence that regularisation, not conditioning, is the binding constraint.

**Practical consequence:** cut `EPOCHS` to ~15–20. Each future experiment then costs ~3.5h
instead of 14h, with no loss of information.

---

## 3. Experiment 2 — is there any speaker identity in the fingerprints? (notebook cell 12)

Rather than run more 14-hour trainings, we tested the fingerprints directly: run the same
retrieval protocol on the raw fingerprints with **L1 distance**, no ECAPA, no training at all.

5000 VoxCeleb1 clips, 1218 speakers, ~4 same-speaker targets per query. Chance R@1 = 0.0009.

| component | EER (raw) | EER (L1-normalised) | R@1 (raw) | vs chance |
|---|---|---|---|---|
| `input_activity` | **0.4029** | 0.4097 | 0.0129 | ~14× |
| `in_weights` | 0.4131 | 0.4248 | 0.0108 | ~12× |
| `hidden_activity` | 0.4329 | 0.4416 | 0.0031 | ~3.4× |
| `hid_weights` | 0.4426 | 0.4497 | 0.0047 | ~5× |
| all 4 concatenated | — | 0.4269 | 0.0102 | ~11× |
| **random control** | 0.5043 | — | 0.0010 | 1× |

The random control landing at 0.5043 / 0.0010 confirms the protocol and the chance level are
correct.

**What this says:**

- The signal is **statistically overwhelming but practically negligible**. R@1 at 14× chance is
  ~70 correct hits where 4.4 are expected — that is not noise. But EER 0.40 against ECAPA's 0.13
  is nearly useless.
- **The ordering is backwards for the hypothesis.** The *input* layer carries more than the
  *hidden* layer in both modalities (activity 0.403 vs 0.433; weights 0.413 vs 0.443). And
  `input_activity` is essentially a coarse cochlear energy profile — exactly what a mel
  front-end already captures, better. The STDP-*learned* components carry the least.
- L1-normalising makes every component slightly worse, meaning part of the weak signal lives in
  total mass (loudness), not pattern.

**Important caveat that turned out to matter:** this measures *distance-space* structure, which
is a **lower bound** on extractable information. A learned readout does not need raw distances
to be good. Experiment 3 showed this bound was very loose.

---

## 4. Experiment 3 — why is it so weak? (notebook cell 13)

6231 VoxCeleb2 clips from 300 speakers (20.8 clips/speaker — sampled by whole speakers so
within-speaker variance is measurable).

| component | spread | zero% | PC1 | PC1-20 | \|r\| mass | \|r\| zero | F median |
|---|---|---|---|---|---|---|---|
| `in_weights` | 0.644 | 31.0% | 0.166 | 0.650 | 0.900 | 0.885 | 3.19 |
| `hid_weights` | 0.554 | 24.1% | 0.187 | 0.614 | 0.901 | 0.891 | 3.02 |
| `input_activity` | 0.512 | 16.3% | 0.281 | 0.958 | 0.603 | 0.399 | 5.69 |
| `hidden_activity` | 0.476 | 15.8% | 0.434 | 0.973 | 0.776 | 0.409 | 5.68 |
| **random control** | — | 0% | 0.003 | 0.056 | 0.003 | 0.000 | **0.99** |

- `spread` = how much fingerprints differ between clips (small would mean every clip produces
  nearly the same vector)
- `PC1`, `PC1-20` = fraction of between-clip variance explained by the top 1 / top 20 principal
  components
- `|r| mass`, `|r| zero` = how strongly the dominant direction correlates with total activity
  mass and with the fraction of silence-masked entries
- `F` = between-speaker variance ÷ within-speaker variance, per dimension. **F = 1 means no
  speaker structure at all** — the random row confirms this at 0.99.

**Displacement from the shared starting weights:** median 0.659 (p10 0.599, p90 0.735); corpus
mean fingerprint vs `W_IH_INIT` cosine 0.837.

**What this says:**

1. **Fingerprints vary a lot, and they vary in different directions.** Displacement from the
   shared init (0.659) and between-clip spread (0.644) are the same magnitude, so clips move far
   from the starting point *and* to different places. An earlier hypothesis that homeostatic
   normalisation was collapsing everything to one shared attractor is **disproven**. The
   simulation is doing real per-clip work.
2. **There is substantial speaker-correlated structure** — F of 3.0 to 5.7 against a null of
   0.99. That is far from nothing. (Caveat: F counts *any* speaker-consistent factor — gender,
   recording device, channel, loudness — not identity as such.)
3. **But it is very low-rank**: 20 components explain 96–97% of the activity variance. A 128-d
   vector is effectively ~20-d, and those dimensions are highly correlated, so the number of
   independent discriminative directions is small.
4. **And the dominant factor is energy/silence, not identity**: for both weight components the
   top direction correlates 0.90 with total mass and 0.89 with the per-clip silence fraction.
   The silent-cell zeroing (31% / 24% of entries) injects that energy signal directly into the
   fingerprint.

This explains why Experiment 2 looked near-chance: raw L1 distance over 5888 dimensions is
dominated by the high-variance energy directions, so a handful of correlated discriminative
directions buried in that covariance are invisible to it.

---

## 5. Experiment 4 — how much is actually extractable? (notebook cell 14)

Two parts. All PCA and standardisation statistics fit on VoxCeleb2 and applied to VoxCeleb1 —
never fit on the evaluation set. The eval subsample is identical to Experiment 2's, and the
`raw`/`l1` rows reproduced that table exactly, which validates both cells against each other.

### Part A — remove the energy directions, retry unsupervised retrieval

Best result per component (all cosine):

| component | best variant | EER | (was, raw L1) |
|---|---|---|---|
| `input_activity` | −PC5 | **0.3734** | 0.4029 |
| `input_activity` | whiten64 | 0.3795 (R@1 **0.0213** ≈ 23× chance) | |
| `in_weights` | −PC1 | **0.3800** | 0.4131 |
| `hidden_activity` | raw | 0.4353 | 0.4329 |
| `hid_weights` | raw | 0.4514 | 0.4426 |

Energy-dominance was real, but it is not hiding a strong signal underneath: 0.40 → 0.37, not
0.40 → 0.20. Cosine beats L1 in nearly every row. `hid_weights` actually gets *worse* under PCA
removal (−PC5 → 0.4776) and is the weakest component in every single test run so far.

### Part B — train a readout on fingerprints, test on unseen speakers

Trained on 80000 VoxCeleb2 clips / 6108 speakers, evaluated on held-out VoxCeleb1 speakers.

| | EER | R@1 | mAP |
|---|---|---|---|
| no learning (standardised features, cosine) | 0.4021 | 0.0096 | 0.0097 |
| **linear head, trained** | **0.3451** | 0.0086 | 0.0107 |
| MLP-512, trained | 0.3511 | 0.0114 | 0.0121 |

**The full ladder:**

```
chance                          0.500
raw unsupervised (best)         0.403
energy directions removed       0.373
learned readout (unseen spk)    0.345
ECAPA from audio alone          0.133
```

The fingerprint gets roughly a quarter of the way from chance to ECAPA.

**Caveat on this number:** both probes reached `train_acc = 1.0000` by epoch 14 — 80000 clips
memorised into 6108 classes — and the *linear* head transferred better than the MLP
(0.3451 vs 0.3511), the classic overfitting signature. So 0.345 is a loose **lower** bound on
extractability, not a ceiling. A regularised probe (weight decay, dropout, early stopping on a
held-out VoxCeleb2 speaker split) would likely do somewhat better.

---

## 6. Experiment 5 — is it complementary to the audio? (notebook cell 15)

The decisive test. "Weak" and "useless" are not the same thing: a weak but *orthogonal* side
channel can still improve a strong model. This test uses the already-saved
`best_ecapa_C1024_nofilm.pt` (epoch 8, EER 0.1328) — **no ECAPA retraining** — and fuses its
pair scores with the fingerprint probe's:

```
fused_score = cos_ecapa + w * cos_fingerprint
```

`w` is swept **on the evaluation set itself**, deliberately. This makes the test generous to the
hypothesis: if even the best possible weighting cannot beat `w = 0`, the negative result is
solid.

**Correlation between the two systems' pair scores: +0.2249** — low. They judge pairs
differently.

| system | EER | R@1 | R@5 | mAP | vs baseline |
|---|---|---|---|---|---|
| fingerprint alone | 0.3451 | 0.0086 | 0.0347 | 0.0107 | |
| **ECAPA alone (w=0)** | **0.1353** | 0.2816 | 0.5111 | 0.1930 | baseline |
| + fingerprint w=0.05 | 0.1345 | 0.2839 | 0.5140 | 0.1943 | better 0.0008 |
| + fingerprint w=0.10 | 0.1340 | 0.2843 | 0.5138 | 0.1947 | better 0.0013 |
| **+ fingerprint w=0.20** | **0.1335** | 0.2849 | 0.5134 | 0.1939 | **better 0.0018** |
| + fingerprint w=0.30 | 0.1350 | 0.2745 | 0.5052 | 0.1894 | better 0.0003 |
| + fingerprint w=0.50 | 0.1384 | 0.2585 | 0.4874 | 0.1775 | worse 0.0030 |
| + fingerprint w=0.75 | 0.1441 | 0.2287 | 0.4517 | 0.1563 | worse 0.0087 |
| + fingerprint w=1.00 | 0.1541 | 0.1982 | 0.4055 | 0.1353 | worse 0.0187 |
| + fingerprint w=1.50 | 0.1752 | 0.1420 | 0.3266 | 0.1001 | worse 0.0398 |
| + fingerprint w=2.00 | 0.1932 | 0.1010 | 0.2632 | 0.0755 | worse 0.0579 |

(ECAPA scores 0.1353 on this 5000-clip subsample vs 0.1328 on the full 22496-clip gallery —
a normal, small gallery-size effect. The `w=0` row is the correct baseline here.)

**What this says:**

- The curve is clean and unimodal — rises to a peak at w=0.20, then falls monotonically. That
  shape means the tiny gain is a **real effect, not noise**.
- The gain is **0.0018 EER with the weight tuned on test**, i.e. the optimistic version. FiLM
  independently gave 0.0014. Two unrelated mechanisms, same magnitude.
- The optimal weight being small (0.2) is exactly what you expect when fusing a 0.345 system
  with a 0.135 system.

**Conclusion: the fingerprint is complementary but too weak for that complementarity to be
worth anything.** The bottleneck is the fingerprint, not the fusion.

---

## 7. Corrections made along the way

Recorded so they are not re-derived or re-believed:

1. **The old `USE_FILM=False` baseline (EER 0.2635) was not comparable** — it came from the
   fp16-pooling code. Retired; §2 supersedes it.
2. **"The fingerprints are pinned to one shared attractor" — disproven** by Experiment 3.
   Displacement from init (0.659) equals between-clip spread (0.644): clips move far, and to
   different places.
3. **"Experiment 4 Part B is an upper bound on what FiLM could extract" — wrong.** With
   `train_acc = 1.0` it is an overfit probe, so 0.345 is a loose lower bound. Do not quote it as
   a ceiling.
4. **The unsupervised probes were measuring against the wrong target.** Per the project's design
   intent (below), the SNN was never meant to make different sessions of one speaker converge —
   that job belongs to the ANN backend. Cross-session retrieval on raw fingerprints with no
   learned readout tests something the design does not claim. Experiment 5, which uses a learned
   readout, is the one that matches the intent.

---

## 8. Design intent, and the measurement gap it exposes

**Stated intent:** it was never the goal for different *sessions* of the same person to converge
to the same STDP pattern — different sessions producing different patterns is expected and has
been happening all along. The intended property is that **utterances from the same session
converge to nearly the same fingerprint**, and the ANN backend exists precisely to learn the
speaker-invariant mapping over those per-session patterns.

**The gap:** the corpus protocol stores **1 random utterance per session**. Every session
therefore contributes exactly one fingerprint to the existing `.npz` collection, so the core
design property — same-session convergence — **cannot be measured or exploited from the data on
disk at all.**

This is the most important open item. If same-session utterances really do collapse together,
session-level averaging is free denoising and could move the fingerprint's 0.345 substantially —
which, given the low 0.22 correlation with audio, is exactly what would turn a 1% curiosity into
a real gain.

---

## 9. Recommended next steps, in priority order

1. **Measure same-session convergence.** Re-run `prepare_fingerprints.ipynb` keeping several
   utterances per session for a subset of speakers, then compare within-session against
   between-session fingerprint distance. This is the only experiment that can change the
   conclusion, and it tests the design's actual claim.
2. **If convergence is tight: try session-averaged fingerprints** and re-run Experiments 4B
   and 5. A stronger fingerprint fused at correlation 0.22 is where a real gain would come from.
3. **Fix the overfitting before any further training runs.** Cut `EPOCHS` to 15–20; decide
   whether to restore crop-position jitter (which requires deciding whether the fingerprint is
   an identity prior for the utterance or a frame-aligned description of that exact 2s window —
   currently it is the latter by construction).
4. **Do not spend the multi-seed sweep** on the current fingerprint. With seed-level noise
   plausibly ~0.002 and the effect at 0.0014, resolving it would need on the order of 16 paired
   seeds (~110 GPU-hours) to measure something we now know is ~1%.
5. Optional, cheap: a **regularised probe** to tighten the 0.345 figure and the 0.0018 floor.

---

## 10. How to reproduce

The five diagnostic cells are **temporary** and live at the end of `ecapa_film_snn.ipynb`.
Delete them when done — nothing else in the notebook reads anything they define.

| cell | experiment | prerequisites | runtime |
|---|---|---|---|
| 12 | Exp. 2 — L1 retrieval on raw fingerprints | cells 1, 2, 3 | ~1 min |
| 13 | Exp. 3 — variance / PCA / Fisher diagnostic | cells 1, 2, 3 | ~2 min |
| 14 | Exp. 4 — energy removal + supervised probe | cells 1, 2, 3 | ~5 min |
| 15 | Exp. 5 — score fusion with saved ECAPA | cells 1, 2, 3, 4, 14 | ~5 min |

None of them need the ~90-minute corpus audio decode. Cell 15 decodes only the 5000 evaluation
clips it needs (~3 min) and requires `best_ecapa_C1024_nofilm.pt` to be present in `CKPT_DIR`.

Full training runs (Experiment 1): set `USE_FILM` / `FINGERPRINT_PARTS` in the CONFIG cell and
run everything. `TAG` encodes the active configuration so checkpoints from different ablations
never collide.
