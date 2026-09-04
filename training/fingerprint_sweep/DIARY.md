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

## Round 1 — 4 epochs is free, narrow+low-threshold helps, and the session table reframes the problem (2026-09-03)

**Ran on Kaggle CPU (4 workers), 480 wavs / 60 speakers / 240 sessions, 42 min total.**

**Variants:**

| variant | epochs | R_EXC | vth_rest |
|---|---|---|---|
| `baseline` | 16 | 11 | 0.60 |
| `ep4` | 4 | 11 | 0.60 |
| `ep4_r5` | 4 | 5 | 0.60 |
| `ep4_r5_vth` | 4 | 5 | 0.45 |

**Results:**

```
     variant     EER     R@1     mAP |  EER iw  EER ia |  PC1-5  PC1-20  hid PC5 |   dead     F |  s/wav
    baseline  0.4042  0.0563  0.0563 |  0.3869  0.3951 |  0.363   0.557    0.863 |  15.4%  2.02 |    2.3
         ep4  0.4034  0.0458  0.0537 |  0.3951  0.3965 |  0.446   0.641    0.906 |  15.4%  2.28 |    0.9
      ep4_r5  0.3868  0.0396  0.0565 |  0.3965  0.3963 |  0.389   0.635    0.832 |  16.1%  2.23 |    0.8
  ep4_r5_vth  0.3789  0.0563  0.0595 |  0.3882  0.3951 |  0.366   0.616    0.833 |  12.5%  2.19 |    0.8

     variant  same session   same spk, diff sess  diff speaker  session gap
    baseline        0.2798                0.0920       -0.0028       0.1878
         ep4        0.3287                0.1091       -0.0032       0.2196
      ep4_r5        0.3229                0.1047       -0.0022       0.2182
  ep4_r5_vth        0.3169                0.1062       -0.0026       0.2107
```

**Reading:**

- **Defect 3 confirmed, and it is free money.** `ep4` matches `baseline` on EER (0.4034 vs 0.4042)
  at 2.6x less wall clock. Twelve of sixteen epochs genuinely do nothing. Adopt `n_epochs=4`
  unconditionally — every later experiment gets cheaper with no measured cost.
- **Defects 1 and 2 need each other, exactly as predicted.** `ep4_r5` alone pushed dead units *up*
  (15.4% -> 16.1%) — the named risk of a narrower receptive field materialised. Adding
  `vth_rest=0.45` pulled them to **12.5%**, the lowest of the round, and took EER with it.
  Best variant: **`ep4_r5_vth`, EER 0.3789 vs 0.4042 baseline (-6.3% relative) at 2.9x cheaper.**
- **`PC1-5` failed as the steering metric and is retired.** `ep4` had the *worst* PC1-5 (0.446)
  with baseline-equal EER; `ep4_r5_vth` had baseline-equal PC1-5 (0.366) with the best EER. It
  is not even comparable across variants — the weight tensors shrink with `r_exc`, so the
  concatenated feature dimension changes between rows. **`hid PC5` is the comparable one**
  (always 128-d) and it did track EER correctly: 0.863 -> 0.832 under `r5`. Steer by `hid PC5`.
- **The session table is the round's real result.** Read as a variance decomposition
  (`x = speaker + session + noise`; same-session cos = spk+sess, cross-session same-speaker
  cos = spk), `ep4_r5_vth` gives:

  | component | share of the fingerprint |
  |---|---|
  | speaker | **0.11** |
  | session | 0.21 |
  | per-utterance noise | **0.68** |

  The design's premise holds — the session gap is large (0.21) — but the actionable reading is
  that **two-thirds of the fingerprint is per-utterance noise and the session component is 2x
  the speaker component.** That, not fan-in, is why EER sits at 0.38. Round 1's knobs were
  tuning a term that is only 11% of the signal.

**Decision:** next round — but redirected. Per `PROTOCOL.md`, a large session gap means denoising
beats further knob-tuning. Round 2 attacks the 0.68 noise term instead of the architecture.

**Gate position:** -6.3% relative. Reaching 0.28 from 0.345 needs -19%. Round 2 must move
substantially more than round 1 did, or round 3 closes the direction.

**Changed for next round:**

- Carry `baseline` and `ep4_r5_vth` forward (cached, free). Same 480 clips and seed, so the
  round-1 npz files stay valid.
- `r3_vth` (r=3) and `r5_vth35` (vth=0.35) push the two axes that worked one step further —
  cheap, and they bound how much is left in them.
- **`clip4s_ep2` is the round's real test:** 2 epochs over a 4s clip is the *same 8s of simulated
  time* as `ep4` over 2s, but spends it on new audio instead of repeats. Round 1 proved repeats
  buy nothing; the 0.68 noise term says averaging over more content should. Compute-matched, so
  the comparison is clean. `clip4s_ep4` adds the uncontrolled version for reference.
- Three eval-cell additions that need **no new compute** (they re-read the cached npz):
  session-averaged EER (the N=2 point on the denoising curve), nuisance projection (dominant
  within-speaker directions learned on half the speakers, projected out, evaluated on the
  held-out half — the per-utterance-usable version of removing the session term, since at test
  time only one utterance exists), and per-component EER for all four blocks.

---

## Round 2 — the knobs are exhausted; audio length is the lever (2026-09-03)

**Kaggle CPU, 4 workers, same 480 clips, 41 min. Champion re-run instead of cached.**

**Results:**

```
     variant     EER     R@1     mAP |  EER iw  EER ia |  PC1-5  PC1-20  hid PC5 |   dead     F |  s/wav
  ep4_r5_vth  0.3792  0.0583  0.0594 |  0.3875  0.3958 |  0.367   0.616    0.833 |  12.5%  2.20 |    0.9
      r3_vth  0.3788  0.0625  0.0598 |  0.4021  0.3968 |  0.360   0.601    0.757 |  12.1%  2.10 |    0.8
    r5_vth35  0.3799  0.0646  0.0626 |  0.3785  0.3951 |  0.352   0.601    0.832 |  10.2%  2.16 |    0.8
  clip4s_ep2  0.3587  0.1042  0.0843 |  0.3521  0.3826 |  0.369   0.618    0.855 |   8.5%  2.56 |    0.9
  clip4s_ep4  0.3601  0.1146  0.0888 |  0.3458  0.3812 |  0.337   0.592    0.849 |   8.6%  2.39 |    1.3

     variant  same session   same spk, diff sess  diff speaker  session gap
  ep4_r5_vth        0.3205                0.1061       -0.0027       0.2144
      r3_vth        0.3121                0.1075       -0.0018       0.2046
    r5_vth35        0.3201                0.1041       -0.0028       0.2160
  clip4s_ep2        0.4069                0.1272       -0.0031       0.2797
  clip4s_ep4        0.3888                0.1188       -0.0032       0.2700

     variant     EER  sess-avg |  held-out  proj-20 | in_weight hid_weigh input_act hidden_ac
  ep4_r5_vth  0.3792    0.3600 |    0.3805   0.3909 |    0.3875    0.3896    0.3958    0.4122
      r3_vth  0.3788    0.3638 |    0.3827   0.4076 |    0.4021    0.3795    0.3968    0.3982
    r5_vth35  0.3799    0.3539 |    0.3792   0.3944 |    0.3785    0.3850    0.3951    0.4118
  clip4s_ep2  0.3587    0.3444 |    0.3674   0.3778 |    0.3521    0.3757    0.3826    0.3986
  clip4s_ep4  0.3601    0.3431 |    0.3561   0.3611 |    0.3458    0.3792    0.3812    0.3993
```

**Reading:**

- **The noise floor is 0.0003.** The champion re-ran at 0.3792 against round 1's 0.3789 on the
  same clips. Averaging over 480 clips and ~115k pairs washes out the per-clip simulation noise
  almost completely. Every comparison below is far above this floor, and the carry-the-champion
  protocol paid for itself on its first use.

- **Defects 1 and 2 are refuted as EER levers. This is the round's most decisive result.**
  Both mechanisms moved exactly as `FINGERPRINT_IMPROVE.md` designed, and EER did not follow:

  | variant | its mechanism | EER vs champion |
  |---|---|---|
  | `r3_vth` | `hid PC5` 0.833 -> **0.757** (large drop in redundancy) | **-0.0004** (tied) |
  | `r5_vth35` | `dead` 12.5% -> **10.2%** | **+0.0007** (tied) |

  Redundancy and dead units were the two headline defects, and driving each further produced
  nothing. They are not what limits the fingerprint. **Freeze `r_exc=5`, `vth_rest=0.45` and
  stop tuning them.** Round 1's win came from the pair jointly, not from either mechanism
  continuing to pay.

- **Audio length is the lever, and the control held.** `clip4s_ep2` beats the champion by
  **-0.0205** at *identical* cost — 0.9 s/wav for both, confirming the clips really are >= 4s
  and that 8s of simulated time was genuinely spent on new audio rather than repeats.
  `R@1` nearly doubled (0.0583 -> 0.1042), `mAP` +42%, Fisher 2.20 -> 2.56, and `dead` fell to
  8.5% as a side effect — longer audio fixes dead units better than lowering the threshold did.

- **Repeats still buy nothing, now confirmed at 4s too.** `clip4s_ep4` vs `clip4s_ep2`: EER
  0.3601 vs 0.3587 at 1.4x the cost. Doubling exposure over the same audio does nothing for EER,
  though it does help `R@1` (0.1146) and `in_weights` EER (0.3458). Two independent clip lengths
  now agree: **exposure time is worthless, distinct content is valuable.**

- **The mechanism is visible in the variance split.** 2s -> 4s:

  | | speaker | session | noise |
  |---|---|---|---|
  | `ep4_r5_vth` (2s) | 0.106 | 0.214 | 0.679 |
  | `clip4s_ep2` (4s) | **0.127** | 0.280 | **0.593** |

  Noise -0.086, speaker +0.021 — the predicted direction. Note the session term grew faster
  (+0.066) than the speaker term, so the fingerprint is becoming a better session detector even
  as it becomes a better speaker detector. Cross-session EER still improved because the noise
  reduction dominates, but this is worth watching if audio length keeps increasing.

- **Nuisance projection failed and should not be pursued as specified.** `proj-20` is *worse*
  than unprojected on all five variants (+0.010 to +0.025). The dominant within-speaker
  directions evidently carry speaker information too, so the speaker signal does not live in a
  subspace cleanly orthogonal to the nuisance. Round 3 retries at smaller k before abandoning it.

- **The 4-block concatenation dilutes.** For both 4s variants, `in_weights` alone beats the full
  concat (0.3521 vs 0.3587; **0.3458** vs 0.3601). `hidden_activity` is the weakest block
  everywhere (~0.40-0.41). Block selection is free post-processing and is now worth a proper
  subset search.

- **Session averaging helps, less than predicted.** -0.014 to -0.026 across variants, versus a
  model that predicted the speaker share rising 0.106 -> 0.16. Real but second-order, and it
  still requires several utterances per session at test time.

**Decision:** round 3, and per the agreed gate it is the last sweep round. The trend justifies
it: round 1 gave -6.3% relative, round 2 gives **-11.3%** (`clip4s_ep2`) to **-14.4%**
(`clip4s_ep4` `in_weights` alone) against the round-1 baseline of 0.4042. The gate needs -19%.
The single lever that produced this has an obvious untested continuation — 4s was as far as the
round went, not as far as the effect goes.

**Gate position:** -14.4% of the -19% needed. Round 3 tests 8s and 16s. If EER keeps falling at
anything like the 2s->4s rate, the gate is reachable; if it flattens, it closes.

---

## Round 3 — the corpus caps at 5s, so the round could not test its hypothesis (2026-09-03)

**Kaggle CPU, 4 workers, same 480 clips, 30 min.**

**The duration cell fired before the sweep and invalidated the design:**

```
clip duration over 480 wavs (ms): min 2585 | p25 4992 | median 4992 | p75 4992 | max 5056
 clip_ms  % clips at full length   mean audio actually used
    4000                   94.2%                   3962 ms
    8000                    0.0%                   4887 ms
   16000                    0.0%                   4887 ms
```

**Every VoxCeleb1 utterance in this sample is ~5s.** `clip_ms=8000` and `clip_ms=16000` therefore
both mean "use the whole file", and `clip8s_ep1` / `clip16s_ep1` saw **identical audio**. The
round was designed to test 2x and 4x the audio; it actually tested **1.23x**. The octave-up
continuation of round 2's winning lever never happened. That was my design error — I set the
clip lengths before knowing the durations.

**Results:**

```
     variant     EER     R@1     mAP |  EER iw  EER ia |  PC1-5  PC1-20  hid PC5 |   dead     F |  s/wav
  clip4s_ep2  0.3621  0.1021  0.0853 |  0.3535  0.3817 |  0.370   0.618    0.856 |   8.5%  2.55 |    0.9
  clip8s_ep1  0.3601  0.1042  0.0847 |  0.3514  0.3809 |  0.387   0.624    0.865 |   7.7%  2.66 |    0.8
  clip8s_ep2  0.3528  0.1062  0.0881 |  0.3399  0.3806 |  0.365   0.620    0.861 |   7.8%  2.63 |    1.0
 clip16s_ep1  0.3570  0.1104  0.0850 |  0.3525  0.3812 |  0.387   0.624    0.865 |   7.7%  2.66 |    0.7

     variant  same session   same spk, diff sess  diff speaker  session gap
  clip4s_ep2        0.4057                0.1289       -0.0031       0.2768
  clip8s_ep1        0.4176                0.1353       -0.0027       0.2823
  clip8s_ep2        0.4220                0.1332       -0.0031       0.2889
 clip16s_ep1        0.4173                0.1354       -0.0028       0.2819

     variant     EER  sess-avg |  held-out   proj-2   proj-5  proj-20 | in_weight hid_weigh input_act hidden_ac
  clip4s_ep2  0.3621    0.3399 |    0.3653   0.3624   0.3597   0.3610 |    0.3535    0.3729    0.3817    0.3975
  clip8s_ep1  0.3601    0.3472 |    0.3785   0.3736   0.3653   0.3599 |    0.3514    0.3708    0.3809    0.3965
  clip8s_ep2  0.3528    0.3374 |    0.3618   0.3618   0.3528   0.3458 |    0.3399    0.3646    0.3806    0.4021
 clip16s_ep1  0.3570    0.3497 |    0.3778   0.3708   0.3583   0.3700 |    0.3525    0.3715    0.3812    0.3986

     variant             in_w         in+hid_w      in_w+in_act  in+hid_w+in_act         all four
  clip4s_ep2           0.3535            0.3591            0.3532*           0.3604            0.3621
  clip8s_ep1           0.3514*           0.3590            0.3517            0.3586            0.3601
  clip8s_ep2           0.3399            0.3528            0.3389*           0.3510            0.3528
 clip16s_ep1           0.3525            0.3562            0.3500*           0.3562            0.3570
```

**Reading:**

- **The noise floor is ~0.003, ten times larger than round 2 measured.** Two independent
  estimates now agree: the champion moved 0.3587 -> 0.3621 across rounds (0.0034), and the
  accidental `clip8s_ep1` / `clip16s_ep1` replicate — identical audio, different RNG — differ by
  0.0031. **Round 2's 0.0003 was a lucky draw from a single sample.** Every conclusion below
  uses 0.003, and every earlier conclusion was re-checked against it (see next point).

- **Round 2's conclusions survive the corrected floor.** `clip4s`'s -0.0205 is 6.8x the floor
  and still solid; the `r3_vth` / `r5_vth35` ties (-0.0004, +0.0007) are still ties; the
  `clip4s_ep4` vs `clip4s_ep2` tie (0.0014) is still a tie. Only the *precision claim* changes,
  not any decision made from it.

- **The audio lever is not exhausted — it is untestable on this corpus.** 4s -> 5s (all there
  is) gave -0.0020, well inside noise. That is not evidence the lever died; it is 23% more
  audio where round 2's win came from 100% more. VoxCeleb1's segmentation caps the experiment.

- **One epoch is too few; two is enough.** `clip8s_ep1` -> `clip8s_ep2` is -0.0073 (2.4x floor)
  at the same audio. Combined with rounds 1-2 (16 vs 4 tied, 4 vs 2 tied at 4s), the rule is
  **two passes, and nothing beyond** — not "repeats are worthless" as round 2 put it.

- **Two blocks of the fingerprint should be dropped.** `in_weights + input_activity` beats all
  four blocks on 3 of 4 variants, by 0.009-0.018 (3-6x floor), consistently signed. Best number
  of the whole loop: **`clip8s_ep2`, `in_w+in_act` = 0.3389.** `hid_weights` and
  `hidden_activity` dilute, and dropping them halves the tensor.

- **Session averaging is the one denoiser that consistently works** (-0.013 to -0.022, 4-7x
  floor). It still needs several utterances per session at test time.

- **Nuisance projection is inconclusive and should be dropped.** It hurt every variant in round
  2 and mildly helps some here; the held-out half is only 30 speakers, so these are
  noise-dominated. Not worth more attention.

**Decision: stop sweeping. This closes the sweep loop at the agreed 2-3 rounds.**

**Gate position: -16.2%** (0.4042 -> 0.3389) against the **-19%** needed, with the trend
decelerating hard: **-6.3% -> -14.4% -> -16.2%**. Per `PROTOCOL.md` I am not proposing a round 4
of knob-tuning. Every knob in `FINGERPRINT_IMPROVE.md` is now measured and two of its three
proposed changes are refuted as levers.

**But the promotion test has never been run, and it is the only measurement that answers the
actual question.** The sweep is a proxy; three rounds have accumulated a -16.2% change that has
never been checked against the real evaluation. That is the next action, not another sweep.

### Promotion configuration (best of the loop)

| setting | value | came from |
|---|---|---|
| `n_epochs` | **2** | round 3: 1 is too few, 4 and 16 add nothing |
| `r_exc` | **5** | round 1 |
| `vth_rest` | **0.45** | round 1 |
| `clip_ms` | **8000** (i.e. the whole utterance) | rounds 2-3 |
| fingerprint blocks | **`in_weights` + `input_activity` only** | round 3 |

Steps, per `PROTOCOL.md`: regenerate one full shard with these settings, run
`../ecapa_film_snn/ecapa_film_snn.ipynb` cell 14 Part B, check fingerprint-alone EER against
**0.28** (from 0.345). Note this is also ~8x cheaper per wav than the original 16-epoch 2s
pipeline in simulated time, so the shard regeneration is affordable regardless of the outcome.

### The one hypothesis the loop never tested

Round 2 showed doubling *distinct* audio was worth -0.0205, and round 3 could not extend it
because utterances are 5s. The untested continuation is **concatenating several utterances of
the same session** into one 10-20s input. This is deliberately **not** proposed as round 4 —
the gate says stop, and it should only be revisited if the promotion test lands close to 0.28
(say 0.28-0.30), where one more push would decide it rather than merely encourage it.

---

## Round 4 — systematic screen of the previously-hardcoded knobs (2026-09-03)

**The gate was reopened by user decision** (see `PROTOCOL.md`). The justification is real:
rounds 1-3 varied four hyperparameters because only four were exposed. The core now
parameterises **all** of `DEFAULTS` — ~25 knobs — so the space can actually be searched rather
than assumed exhausted.

**Core refactor, verified before running.** Constants now reach Brian2 through explicit
per-object `namespace=` dicts instead of f-string literals baked into the equations.
Verified locally against the old core:

- all precompute matrices bit-identical at champion settings;
- with `sigma_noise=0` (deterministic), a 300ms run gives identical hidden spike counts and
  weights to **1e-16** — i.e. float rounding;
- unknown hyperparameter names now raise instead of being silently ignored;
- as a side effect the Cython code compiles **once** for the whole sweep instead of once per
  distinct `vth_rest`, since no value is embedded in the model text any more.

It also fixes a latent fragility: the old core only worked when `net.run()` was called from
inside its own module, because Brian2 scraped the constants from the caller's frame.

**Design: one factor at a time, champion + exactly one change per variant.** Round 2's `r3_vth`
moved two knobs at once and cost a clean attribution; not repeating that.

| variant | change | from | why |
|---|---|---|---|
| `champ` | — | | reference |
| `champ_rep` | — (identical) | | replicate: EER floor **and** per-fingerprint reproducibility |
| `noise_lo` | `sigma_noise` 0.010 | 0.030 | is the ~59% within-utterance variance intrinsic noise? |
| `tau_h_50` | `tau_h_ms` 50 | 150 | 150ms is long for speech; speaker structure is at 10-50ms |
| `norm_dt_5` | `norm_dt_ms` 5 | 25 | **columns regrow to 2.7x the L1 budget between ticks** |
| `norm_exc_06` | `norm_limit_exc` 0.60 | 1.0455 | the binding constraint itself, never varied |
| `stdp_2x` | `apre`/`apost` x2 | 0.008/-0.0096 | learning rate, ratio preserved |
| `inh_weak` | `norm_limit_inh` 0.10 | 0.40 | lateral competition strength |
| `enc_sust` | `sust_gain` 1.0 | 0.3 | onsets currently dominate sustained energy 10:1 |

`norm_dt_5` comes from something the local testing turned up: the L1 column budget is documented
as the binding constraint, but normalisation only runs every 25ms and STDP regrows columns to
**2.79** against a limit of **1.0455** in between. The budget is therefore far looser in practice
than the design assumes. This is pre-existing behaviour, present in the old core too — not
introduced by the refactor.

**New measurement: simulation reproducibility.** The replicate pair is scored by cosine between
the *same wav's* fingerprint across the two runs. This separates simulation noise from real
acoustic content, and decides `sigma_noise`'s relevance directly: if fingerprints reproduce
closely, the 59% within-utterance variance is content and `sigma_noise` is a red herring; if
they do not, `noise_lo` should win outright.

**Cost:** 9 variants x ~8 min ~= **70 min**, nothing attached. Actual: **77 min**.

**Results:**

```
     variant     EER     R@1     mAP |  EER iw  EER ia |  PC1-5  PC1-20  hid PC5 |   dead     F |  s/wav
           champ  0.3556  0.1250  0.0922 |  0.3388  0.3806 |  0.365   0.620    0.861 |   7.8%  2.63 |    1.2
       champ_rep  0.3524  0.1125  0.0891 |  0.3424  0.3819 |  0.365   0.619    0.860 |   7.8%  2.61 |    1.0
        noise_lo  0.3528  0.1146  0.0889 |  0.3417  0.3820 |  0.364   0.619    0.861 |   7.7%  2.62 |    1.0
        tau_h_50  0.3667  0.1083  0.0792 |  0.3584  0.3826 |  0.386   0.645    0.844 |  12.4%  2.68 |    1.0
       norm_dt_5  0.3558  0.1146  0.0898 |  0.3434  0.3833 |  0.366   0.620    0.860 |   7.8%  2.62 |    1.1
     norm_exc_06  0.3701  0.0896  0.0764 |  0.3607  0.3812 |  0.369   0.621    0.829 |  15.1%  2.55 |    1.0
         stdp_2x  0.3472  0.1125  0.0895 |  0.3340  0.3819 |  0.340   0.593    0.851 |   7.9%  2.51 |    1.0
        inh_weak  0.3465  0.1062  0.0902 |  0.3326  0.3826 |  0.304   0.540    0.887 |   5.3%  2.28 |    1.0
        enc_sust  0.3340  0.1375  0.1047 |  0.3222  0.3809 |  0.317   0.589    0.883 |   2.4%  2.51 |    1.0

SESSION CONVERGENCE           same session   same spk, diff sess   diff speaker   session gap
           champ                    0.4231                0.1334        -0.0031        0.2897
       champ_rep                    0.4218                0.1321        -0.0031        0.2897
        noise_lo                    0.4216                0.1321        -0.0031        0.2895
        tau_h_50                    0.4196                0.1310        -0.0027        0.2886
       norm_dt_5                    0.4215                0.1322        -0.0031        0.2893
     norm_exc_06                    0.3996                0.1210        -0.0025        0.2787
         stdp_2x                    0.3979                0.1257        -0.0032        0.2722
        inh_weak                    0.3633                0.1154        -0.0034        0.2479
        enc_sust                    0.3906                0.1316        -0.0036        0.2591

BLOCK SUBSETS         in_w    in+hid_w   in_w+in_act   in+hid_w+in_act    all four
           champ   0.3388*     0.3527        0.3396          0.3535        0.3556
       champ_rep   0.3424*     0.3535        0.3430          0.3528        0.3524
        noise_lo   0.3417*     0.3532        0.3424          0.3527        0.3528
        tau_h_50   0.3584      0.3667        0.3576*         0.3663        0.3667
       norm_dt_5   0.3434      0.3577        0.3417*         0.3573        0.3558
     norm_exc_06   0.3607      0.3667        0.3573*         0.3674        0.3701
         stdp_2x   0.3340*     0.3483        0.3340          0.3476        0.3472
        inh_weak   0.3326*     0.3465        0.3340          0.3458        0.3465
        enc_sust   0.3222      0.3347        0.3192*         0.3323        0.3340

SIMULATION REPRODUCIBILITY  (champ vs champ_rep, 480 wavs)
  same wav, two runs      cos  0.8483
  different wavs, one run cos -0.0005
  => 15.2% of the fingerprint does not survive a re-run.
```

**Noise floor:** |0.3556 − 0.3524| = **0.0032**, a third agreeing estimate (round 3 gave 0.0034
and 0.0031). Reference for this round is the pair mean, **0.3540**; the decision threshold is
2x floor = 0.0064.

| variant | Δ vs 0.3540 | in floor units | verdict |
|---|---|---|---|
| `noise_lo` | −0.0012 | 0.4x | tie |
| `norm_dt_5` | +0.0018 | 0.6x | tie |
| `stdp_2x` | −0.0068 | 2.1x | win (marginal) |
| `inh_weak` | −0.0075 | 2.3x | win |
| `enc_sust` | **−0.0200** | **6.2x** | **clear win** |
| `tau_h_50` | +0.0127 | 4.0x | worse |
| `norm_exc_06` | +0.0161 | 5.0x | worse |

**Reading:**

- **One mechanism explains almost the whole table: hidden-layer activity.** Across all nine
  rows, EER correlates with the dead-unit fraction at **r = +0.96**. Sorting by `dead` sorts
  by EER, with `stdp_2x` the only exception. Every change that raised activity helped
  (`enc_sust` 7.8%→2.4% dead, `inh_weak` →5.3%); every change that lowered it hurt
  (`norm_exc_06` →15.1%, `tau_h_50` →12.4%). **Defect 2 is the live lever, not defect 1.**

- **`enc_sust` is the new champion, EER 0.3340**, and it is the largest single-knob gain of the
  whole loop. `sust_gain` 0.3→1.0 narrows the onset:sustained drive ratio from 10:1 to 3:1.
  It also improves R@1 (0.1250→0.1375) and mAP (0.0922→0.1047), which EER alone can hide.
  Note this knob is in the **auditory front-end**, not the network — the encoder was never
  varied in rounds 1-3, and it turned out to matter more than any network parameter tested.

- **`sigma_noise` is retired.** 3x less noise changed nothing (0.4x floor) even though the
  reproducibility measurement shows 15.2% of the fingerprint does not survive a re-run. So the
  irreproducible part carries no speaker information either way, and the ~59% within-utterance
  variance identified in rounds 2-3 is real acoustic content, exactly as the round-3 reading
  guessed. Don't spend runs here again.

- **The L1 column budget is not the binding constraint the plan says it is.** `norm_dt_5`
  (normalising 5x more often, which does bind the budget — columns regrow to 2.79 against a
  1.0455 limit between 25 ms ticks) is a **tie at 0.6x floor**. Enforcing the documented
  constraint properly changes nothing. Meanwhile loosening the limit's *value* downward
  (`norm_exc_06`) is one of the two worst rows. Read together: the budget matters only through
  how much drive it permits, not as a competition mechanism. `FINGERPRINT_IMPROVE.md` §2's
  framing of it as "the binding constraint" should be read as a statement about weight
  saturation only.

- **The variance decomposition has stopped predicting EER and is retired as a steering metric.**
  It drove round 2's redirection, but this round its speaker/(session+noise) ratio is *anti*-
  correlated with the outcome: `inh_weak` has the worst ratio of any variant (0.131 vs the
  champion's 0.154) and still wins. `enc_sust` wins big at an essentially unchanged ratio
  (0.152). The gain therefore comes from the *shape* of the speaker subspace — more live
  units carrying it — not from its share of total variance. Use `dead` and EER instead.

- **Block subsets, fourth round running.** `hid_weights` and `hidden_activity` dilute on every
  single variant. Best number of the entire loop is `enc_sust` on `in_weights + input_activity`:
  **0.3192**. Under unsupervised cosine retrieval the case for dropping the two hidden blocks
  from the tensor is now about as strong as this sweep can make it (caveat: `ecapa_film_snn`
  uses a *learned* readout, which may extract something cosine cannot).

**Decision:** continue. `enc_sust` becomes the champion. The stopping rule did not fire —
three variants beat the champion by more than 2x floor.

**Changed for round 5:** the champion carries `sust_gain=1.0`. The round is built around the
one mechanism that explained round 4: push hidden activity from four independent directions and
find where it turns over, plus re-test round 4's two smaller winners on top of the new champion
to see whether they add or merely overlap.

---

## Round 5 — the activity lever turns over; no variant wins (2026-09-03)

Champion: `n_epochs=2, r_exc=5, vth_rest=0.45, clip_ms=8000, sust_gain=1.0` (EER 0.3340).

| variant | change | from | question |
|---|---|---|---|
| `champ` | — | | reference |
| `champ_rep` | — (identical) | | noise floor + reproducibility |
| `sust_2` | `sust_gain` 2.0 | 1.0 | 0.3→1.0 won big. Does it keep winning, or is there an optimum? |
| `onset_1` | `onset_gain` 1.0 | 3.0 | is it the sustained:onset **ratio** or the sustained **level**? |
| `scale_2` | `enc_scale` 2.0 | 1.0 | is it sustained drive specifically, or just more total drive? |
| `vth_035` | `vth_rest` 0.35 | 0.45 | the other dead-unit knob, now on a much livelier network |
| `norm_exc_15` | `norm_limit_exc` 1.5 | 1.0455 | `norm_exc_06` was one of the worst rows — is the curve monotone the other way? |
| `inh_weak` | `norm_limit_inh` 0.10 | 0.40 | won 2.3x floor in round 4 — does it still, on top of `enc_sust`? |
| `stdp_2x` | `apre`/`apost` x2 | 0.008/−0.0096 | won 2.1x floor in round 4 — same question |

Five of these (`sust_2`, `onset_1`, `scale_2`, `vth_035`, `norm_exc_15`) are activity levers on
different parts of the pipeline: encoder gain, encoder balance, global input scale, neuron
threshold, weight budget. If activity is genuinely the mechanism they should move together and
the r=0.96 dead↔EER line should extend. If they don't, "activity" was a proxy for something
narrower and the table will say which.

The last two are the deliberate additivity test. Round 4 scored them against the *old* champion,
so a repeat win means they act on a different mechanism from `enc_sust`; a collapse to a tie
means all three were routes to the same thing and only the biggest is worth keeping.

**Cost:** 9 variants ~= **77 min**, nothing attached. Actual: **77 min**.

**Results:**

```
     variant     EER     R@1     mAP |  EER iw  EER ia |  PC1-5  PC1-20  hid PC5 |   dead     F |  s/wav
           champ  0.3348  0.1437  0.1058 |  0.3236  0.3806 |  0.317   0.589    0.884 |   2.4%  2.50 |    1.2
       champ_rep  0.3313  0.1375  0.1035 |  0.3226  0.3799 |  0.318   0.590    0.884 |   2.4%  2.54 |    1.0
          sust_2  0.3409  0.1437  0.1062 |  0.3347  0.3803 |  0.299   0.556    0.896 |   1.1%  2.36 |    1.0
         onset_1  0.3608  0.1000  0.0827 |  0.3607  0.3826 |  0.347   0.603    0.876 |   4.2%  2.41 |    1.0
         scale_2  0.3424  0.1229  0.0993 |  0.3417  0.3809 |  0.296   0.557    0.898 |   0.7%  2.32 |    1.0
         vth_035  0.3272  0.1521  0.1092 |  0.3181  0.3812 |  0.310   0.582    0.885 |   2.0%  2.50 |    1.0
     norm_exc_15  0.3299  0.1667  0.1118 |  0.3132  0.3799 |  0.322   0.601    0.898 |   1.5%  2.60 |    1.0
        inh_weak  0.3292  0.1313  0.1018 |  0.3188  0.3806 |  0.266   0.512    0.902 |   1.3%  2.16 |    1.0
         stdp_2x  0.3326  0.1292  0.1037 |  0.3257  0.3799 |  0.301   0.561    0.877 |   2.5%  2.41 |    1.0

SESSION CONVERGENCE           same session   same spk, diff sess   diff speaker   session gap
           champ                    0.3908                0.1319        -0.0036        0.2590
       champ_rep                    0.3918                0.1329        -0.0036        0.2588
          sust_2                    0.3542                0.1204        -0.0036        0.2338
         onset_1                    0.3902                0.1200        -0.0026        0.2703
         scale_2                    0.3508                0.1212        -0.0037        0.2296
         vth_035                    0.3883                0.1330        -0.0037        0.2553
     norm_exc_15                    0.4026                0.1384        -0.0037        0.2642
        inh_weak                    0.3216                0.1132        -0.0037        0.2085
         stdp_2x                    0.3681                0.1248        -0.0036        0.2433

DENOISING              EER  sess-avg |  held-out   proj-2   proj-5  proj-20
           champ    0.3348    0.3083 |    0.3436   0.3402   0.3396   0.3251
       champ_rep    0.3313    0.3044 |    0.3451   0.3375   0.3361   0.3265
          sust_2    0.3409    0.3084 |    0.3542   0.3625   0.3680   0.3411
         onset_1    0.3608    0.3414 |    0.3729   0.3640   0.3632   0.3382
         scale_2    0.3424    0.3167 |    0.3543   0.3556   0.3480   0.3423
         vth_035    0.3272    0.3167 |    0.3354   0.3347   0.3311   0.3125
     norm_exc_15    0.3299    0.3083 |    0.3431   0.3375   0.3313   0.3256
        inh_weak    0.3292    0.3041 |    0.3382   0.3404   0.3457   0.3429
         stdp_2x    0.3326    0.3083 |    0.3423   0.3333   0.3375   0.3292

BLOCK SUBSETS         in_w    in+hid_w   in_w+in_act   in+hid_w+in_act    all four
           champ   0.3236      0.3330       0.3201*         0.3333        0.3348
       champ_rep   0.3226      0.3306       0.3201*         0.3313        0.3313
          sust_2   0.3347      0.3396       0.3278*         0.3385        0.3409
         onset_1   0.3607      0.3604       0.3542*         0.3610        0.3608
         scale_2   0.3417      0.3379       0.3340*         0.3400        0.3424
         vth_035   0.3181*     0.3277       0.3188          0.3250        0.3272
     norm_exc_15   0.3132*     0.3299       0.3164          0.3288        0.3299
        inh_weak   0.3188      0.3319       0.3160*         0.3278        0.3292
         stdp_2x   0.3257*     0.3333       0.3264          0.3299        0.3326

SIMULATION REPRODUCIBILITY  (champ vs champ_rep, 480 wavs)
  same wav, two runs      cos  0.7936   =>  20.6% does not survive a re-run
```

**Noise floor:** |0.3348 − 0.3313| = **0.0035** (fourth agreeing estimate). Reference 0.3330,
threshold 2x floor = 0.0070. R@1 has its own floor from the same pair: 0.0062.

| variant | Δ EER | floors | verdict | Δ R@1 | floors |
|---|---|---|---|---|---|
| `vth_035` | −0.0058 | 1.7x | tie | +0.0115 | 1.9x |
| `inh_weak` | −0.0038 | 1.1x | tie | −0.0093 | 1.5x |
| `norm_exc_15` | −0.0031 | 0.9x | tie | **+0.0261** | **4.2x** |
| `stdp_2x` | −0.0004 | 0.1x | tie | −0.0114 | 1.8x |
| `sust_2` | +0.0079 | 2.2x | worse | +0.0031 | 0.5x |
| `scale_2` | +0.0094 | 2.7x | worse | −0.0177 | 2.9x |
| `onset_1` | +0.0278 | 7.9x | worse | −0.0406 | 6.5x |

**NO VARIANT WON. The stopping-rule counter is at 1 of 2.**

**Reading:**

- **The activity lever is exhausted, and the turnover was exactly where the round predicted it.**
  Round 4's dead↔EER line was r = +0.96. This round it is **+0.52** overall — but **+0.92 with
  `sust_2` and `scale_2` removed**, and those two are precisely the rows that drove `dead` to its
  lowest values in the loop (1.1% and 0.7%) and got *worse*. The relationship is an inverted U
  and the champion at ~2% dead sits at its bottom. This was pre-registered in the notebook's
  reading cell ("a rising EER at a *lower* dead fraction is the signal that we have passed the
  optimum"), so it is a confirmation rather than a post-hoc story.

- **`onset_1` answers "ratio or level": it is the LEVEL.** `onset_gain` 3.0→1.0 gives a
  sust:onset ratio of 1:1 — *more* sustained-dominant than the champion's 1:3 — and it is the
  worst row of the round at 7.9x floor. Combined with `scale_2` (more of everything, also worse),
  round 4's `enc_sust` win came from raising the **sustained channel's absolute drive**
  specifically, and 1.0 is already its optimum. The encoder is done.

- **The additivity test came back negative.** `inh_weak` (2.3x floor in round 4) and `stdp_2x`
  (2.1x) collapse to 1.1x and 0.1x ties on top of the new champion. They were **alternative
  routes to the same activity increase** as `enc_sust`, not independent mechanisms. This is
  why the protocol re-scores round-N winners against the round-N+1 champion instead of
  assuming they stack — assuming it here would have produced a champion that is no better and
  three knobs harder to explain.

- **The round's real find: nuisance projection started working.** In round 2, `proj-20` hurt
  every variant. Here it **helps 8 of 9**, by a consistent −0.019 (≈5x floor):

  | | held-out | proj-20 |
  |---|---|---|
  | `champ` | 0.3436 | 0.3251 |
  | `vth_035` | 0.3354 | **0.3125** |

  0.3125 is the best generalisation number of the entire loop. The likely cause is round 4's
  activity fix: with 15.4% of units dead the within-speaker subspace was partly *structural*
  (which units happened to fire), and projecting it out removed signal; at 2.4% dead it is
  genuinely nuisance. **This is shippable** — the projection is learned on training speakers
  and needs no session labels at test time, unlike `sess-avg`.

- **`norm_exc_15` is an EER tie that wins on everything else.** Best R@1 of the loop (0.1667,
  4.2x floor), best mAP (0.1118), best `in_weights`-alone EER of the loop (**0.3132**), highest
  Fisher ratio (2.60). An EER tie with a 4.2x-floor R@1 gain is not nothing — it means the
  correct speaker reaches rank 1 more often even though the score distribution's crossing point
  did not move. Same pattern, weaker, for `vth_035`.

- **Reproducibility fell from 84.8% to 79.4%.** Expected: the champion now drives the network
  harder, so the unseeded noise has more to amplify. It does not change the round-4 conclusion —
  the EER floor is unchanged at 0.0035 and `sigma_noise` remains irrelevant.

**Decision:** the champion does not change — the rule is 2x floor and nothing reached it.
Round 6 abandons the activity family entirely and screens the four families never touched in
five rounds.

**Changed for next round:** `vth_035` and `norm_exc_15` both trended right on EER, R@1 and mAP
without hurting anything, so round 6 carries their **combination** as one variant. That is a
deliberate exception to one-factor-at-a-time, flagged here so it is not mistaken for drift; the
justification is that both are sub-threshold on EER but agree across three metrics, which is
exactly the case where a combination is worth one slot.

---

## Round 6 — all four untouched families are inert; the stopping rule fires (2026-09-03)

Champion unchanged: `n_epochs=2, r_exc=5, vth_rest=0.45, clip_ms=8000, sust_gain=1.0`
(EER 0.3330 as the round-5 pair mean).

Rounds 1-5 have now closed the encoder (gain, balance, scale), the activity axis (threshold,
inhibition budget, excitation budget in both directions), exposure (epochs, clip length),
`sigma_noise`, and normalisation timing. What has **never** been varied:

| variant | change | from | family / question |
|---|---|---|---|
| `champ` | — | | reference |
| `champ_rep` | — (identical) | | noise floor + reproducibility |
| `vth_nexc` | `vth_rest` 0.35 **+** `norm_limit_exc` 1.5 | 0.45 / 1.0455 | the two round-5 sub-threshold trends, combined |
| `beta_lo` | `beta` 1.5 | 3.5 | **input-layer adaptation strength.** Controls how strongly input neurons adapt after firing — i.e. how much the front end emphasises novelty over steady tone. Never varied. |
| `tau_a_120` | `tau_a_ms` 120 | 40 | how long that adaptation lasts. Never varied. |
| `vthj_2` | `vth_jump` 2.0 | 1.0 | **hidden spike-frequency adaptation.** Not a gain knob: it silences the *most active* units selectively, which is a competition mechanism rather than the uniform drive change the activity family tested. This is `FINGERPRINT_IMPROVE.md` §7's "intrinsic homeostatic plasticity" in the form the model already has. |
| `tau_vth_200` | `tau_vth_ms` 200 | 60 | how long that competition persists. Never varied. |
| `triplet_off` | `a3pre`/`a3post` 0 | 0.004 / −0.002 | **is the triplet STDP term earning its place?** If turning it off is a tie, the learning rule can be simplified. |
| `p_exc_1` | `p_exc` 1 | 3 | **topographic profile shape.** `max(0, 1−(d/r)^p)`: p=3 is near-boxcar, p=1 is a linear cone. This sharpens the effective receptive field **without cutting fan-in**, which is the defect-1 fix `r_exc` could not deliver without killing units. Never varied. |

Direction-finding (e.g. `vth_jump` down rather than up) is round 7's job if a family responds.

**Cost:** 9 variants ~= **70 min**, nothing attached. Actual: **70 min**.

**Results:**

```
     variant     EER     R@1     mAP |  EER iw  EER ia |  PC1-5  PC1-20  hid PC5 |   dead     F
           champ  0.3354  0.1354  0.1044 |  0.3208  0.3799 |  0.318   0.590    0.884 |   2.4%  2.52
       champ_rep  0.3333  0.1396  0.1048 |  0.3209  0.3799 |  0.318   0.589    0.884 |   2.4%  2.51
        vth_nexc  0.3323  0.1458  0.1113 |  0.3125  0.3809 |  0.318   0.595    0.899 |   1.3%  2.59
         beta_lo  0.3424  0.1229  0.1003 |  0.3334  0.3785 |  0.324   0.598    0.888 |   1.8%  2.49
       tau_a_120  0.3368  0.1437  0.1038 |  0.3268  0.3833 |  0.327   0.593    0.872 |   3.4%  2.45
          vthj_2  0.3327  0.1396  0.1064 |  0.3181  0.3799 |  0.311   0.574    0.890 |   2.2%  2.46
     tau_vth_200  0.3340  0.1417  0.1019 |  0.3190  0.3799 |  0.308   0.561    0.895 |   1.9%  2.37
     triplet_off  0.3319  0.1604  0.1089 |  0.3167  0.3812 |  0.329   0.610    0.887 |   2.4%  2.59
         p_exc_1  0.3375  0.1604  0.1070 |  0.3319  0.3806 |  0.341   0.621    0.874 |   2.6%  2.62

SESSION CONVERGENCE           same session   same spk, diff sess   diff speaker   session gap
           champ                    0.3929                0.1322        -0.0036        0.2607
       champ_rep                    0.3896                0.1320        -0.0036        0.2576
        vth_nexc                    0.3985                0.1385        -0.0037        0.2600
         beta_lo                    0.3798                0.1298        -0.0035        0.2500
       tau_a_120                    0.3873                0.1339        -0.0034        0.2534
          vthj_2                    0.3870                0.1299        -0.0035        0.2571
     tau_vth_200                    0.3684                0.1252        -0.0033        0.2432
     triplet_off                    0.4131                0.1388        -0.0036        0.2743
         p_exc_1                    0.4259                0.1410        -0.0034        0.2848

DENOISING              EER  sess-avg |  held-out   proj-2   proj-5  proj-20
           champ    0.3354    0.3056 |    0.3429   0.3361   0.3431   0.3278
       champ_rep    0.3333    0.3068 |    0.3458   0.3417   0.3375   0.3194
        vth_nexc    0.3323    0.3000 |    0.3389   0.3347   0.3326   0.3291
         beta_lo    0.3424    0.3111 |    0.3577   0.3514   0.3472   0.3514
       tau_a_120    0.3368    0.3083 |    0.3417   0.3431   0.3347   0.3485
          vthj_2    0.3327    0.3056 |    0.3444   0.3431   0.3347   0.3292
     tau_vth_200    0.3340    0.3114 |    0.3464   0.3445   0.3443   0.3250
     triplet_off    0.3319    0.3056 |    0.3445   0.3375   0.3403   0.3278
         p_exc_1    0.3375    0.3083 |    0.3432   0.3431   0.3374   0.3292

BLOCK SUBSETS         in_w    in+hid_w   in_w+in_act   in+hid_w+in_act    all four
           champ   0.3208      0.3354       0.3177*         0.3331        0.3354
       champ_rep   0.3209      0.3306       0.3194*         0.3317        0.3333
        vth_nexc   0.3125      0.3295       0.3124*         0.3298        0.3323
         beta_lo   0.3334      0.3431       0.3292*         0.3424        0.3424
       tau_a_120   0.3268      0.3375       0.3229*         0.3347        0.3368
          vthj_2   0.3181      0.3319       0.3174*         0.3306        0.3327
     tau_vth_200   0.3190      0.3299       0.3188*         0.3319        0.3340
     triplet_off   0.3167*     0.3326       0.3198          0.3313        0.3319
         p_exc_1   0.3319      0.3340       0.3291*         0.3341        0.3375

SIMULATION REPRODUCIBILITY  cos 0.7947  =>  20.5% does not survive a re-run
```

**Noise floor:** this round's replicate gives 0.0021, the smallest of the five estimates
(0.0034 / 0.0031 / 0.0032 / 0.0035 / 0.0021). **Judged against the pooled estimate 0.0031**, not
this round's single draw — round 2 made exactly that mistake with a lucky 0.0003. Reference
0.3344, threshold 2x pooled = **0.0061**.

| variant | Δ EER | floors | verdict | Δ R@1 | floors |
|---|---|---|---|---|---|
| `triplet_off` | −0.0025 | 0.8x | tie | **+0.0229** | **5.5x** |
| `vth_nexc` | −0.0020 | 0.7x | tie | +0.0083 | 2.0x |
| `vthj_2` | −0.0016 | 0.5x | tie | +0.0021 | 0.5x |
| `tau_vth_200` | −0.0003 | 0.1x | tie | +0.0042 | 1.0x |
| `tau_a_120` | +0.0025 | 0.8x | tie | +0.0062 | 1.5x |
| `p_exc_1` | +0.0032 | 1.0x | tie | **+0.0229** | **5.5x** |
| `beta_lo` | +0.0081 | 2.6x | worse | −0.0146 | 3.5x |

**NO VARIANT WON. Second consecutive empty round — the stopping rule fires.**

**Reading:**

- **All four untouched families are inert.** Input-layer adaptation, hidden spike-frequency
  adaptation, the triplet STDP term and the topographic profile all land inside the tie band.
  The only row outside it is `beta_lo`, and it is *worse*. Six rounds have now covered the
  encoder, the activity axis, exposure, noise, normalisation (value and timing), both STDP
  terms, inhibition, input adaptation, hidden competition, and topology in two independent
  parameters. **The hyperparameter space is searched.**

- **`triplet_off` is a tie, so the triplet term is dead weight.** It can be removed from the
  learning rule at no measured cost — the pair term alone does the same job. Worth taking as a
  simplification even though it is not an improvement.

- **The combination heuristic is dead.** `vth_nexc` stacks round 5's two sub-threshold trends and
  lands at 0.7x floor — no amplification at all. Sub-threshold agreement across EER/R@1/mAP is
  therefore *not* a usable signal, and one-factor-at-a-time remains the only thing to trust.
  `vth_nexc` does hold the loop's best `in_weights` EER (**0.3125**) and best subset (0.3124),
  and was never worse than the champion on any metric across two rounds.

- **Defect 1 is definitively not the lever.** `p_exc_1` sharpened the profile as intended and its
  mechanism moved — `hid PC5` 0.884 → 0.874, and the session gap jumped to 0.2848, the highest
  of the round. EER did not move. The plan's headline defect has now been attacked through
  `r_exc` (rounds 1-2) and the profile exponent (round 6) with the mechanism confirmed to
  respond both times and EER indifferent both times.

- **`EER ia` has never moved in six rounds.** Across all nine variants here it spans
  0.3785-0.3833, and it has sat at 0.380 ± 0.003 in every round. `beta_lo` and `tau_a_120` were
  aimed directly at it and did not move it. Whatever `input_activity` encodes, no knob in this
  model reaches it.

- **Correction to the round-5 reading.** I called `vth_035 + proj-20 = 0.3125` the best
  generalisation number of the loop. This round's replicate shows the proj-20 statistic's own
  spread between two *identical* configs is **0.0084** — 2.7x the EER floor. Individual proj-20
  values therefore cannot be ranked, and that 0.3125 was over-stated. The *aggregate* finding
  survives and is what matters: proj-20 helps 8 of 9 variants again, mean **−0.0131**, the same
  direction and rough size as round 5, and the opposite of round 2. Treat it as "nuisance
  projection is worth roughly 0.013-0.019 EER", not as a per-variant score.

**Decision: STOP. The stopping rule agreed after round 3 has fired** — rounds 5 and 6 both
produced no variant beating the champion by more than 2x the noise floor.

---

## Loop closed — final state (2026-09-03)

**Six rounds, ~7.7 hours of Kaggle CPU, 54 variants.**

| | baseline (round 0) | champion (final) |
|---|---|---|
| `n_epochs` | 16 | 2 |
| `r_exc` | 11 | 5 |
| `vth_rest` | 0.60 | 0.45 |
| `clip_ms` | 2000 | 8000 (= whole file, ~4.9s) |
| `sust_gain` | 0.3 | 1.0 |
| **EER** | **0.4042** | **0.3330** (−17.6%) |
| R@1 | 0.0563 | 0.1375 (2.4x) |
| mAP | 0.0563 | 0.1046 (1.9x) |
| dead units | 15.4% | 2.4% |
| cost | 2.3 s/wav | 1.0 s/wav |

Plus two post-processing findings that cost no simulation: **nuisance projection** (proj-20,
worth ~0.013-0.019 EER, needs no session labels at test time) and **block selection**
(`in_weights + input_activity`; the two hidden blocks diluted on every variant of every round).

**What moved, in order of size:** activity (dead units 15.4% → 2.4%, via `vth_rest`, `r_exc` and
above all `sust_gain`), then exposure length, then nothing else.

**What never moved, despite being aimed at directly:** hidden-layer redundancy as a route to EER
(defect 1 — two independent parameters, mechanism confirmed responsive both times), the
`input_activity` block (0.380 ± 0.003 in every round, six families of knobs), `sigma_noise`, and
normalisation timing.

**The gate was not reached.** −17.6% against the −19% needed for the 0.28 promotion target. The
sweep is a proxy on 60 speakers and cannot answer whether that shortfall matters; only the
promotion test can.

**Next step — the promotion test, which has still never been run:**

1. Regenerate one full shard with `prepare_fingerprints.ipynb` at the champion settings.
   Recommended variant: **`vth_nexc`** (`vth_rest=0.35`, `norm_limit_exc=1.5` on top of the
   champion). It is a statistical tie with `champ` on EER, but was equal-or-better on every
   secondary metric across rounds 5 and 6 and holds the loop's best `in_weights` number. Either
   is defensible; `champ` is the conservative choice.
2. Store only `in_weights + input_activity`, dropping the two hidden blocks.
3. Run `../ecapa_film_snn/ecapa_film_snn.ipynb` **cell 14 Part B** → fingerprint-alone EER,
   against the current **0.345** and the **0.28** target.
4. Add nuisance projection to the readout and re-measure.

If Part B lands near 0.28, cell 15 fusion and a short FiLM run follow. If it lands near 0.345 —
i.e. the sweep's −17.6% did not transfer to a learned readout — then the sweep was measuring
something the downstream model does not use, and that is the answer.

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
