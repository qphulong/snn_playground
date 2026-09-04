# Working protocol — fingerprint improvement loop

**Purpose:** find an SNN fingerprint design strong enough to be worth fusing with ECAPA, using
as little of your attention as possible. You run, Claude reads and decides, repeat.

---

## The files here

| file | what it is |
|---|---|
| `PROTOCOL.md` | this file — how the loop works |
| `DIARY.md` | the running record. One entry per round: what was tried, what came back, what it meant, what changed |
| `FINGERPRINT_IMPROVE.md` | the technical plan — the three measured defects, the proposed changes, and the decision gate |
| `fingerprint_sweep.ipynb` | the experiment you actually run |
| `../ecapa_film_snn/RESULTS.md` | how we got here: the ECAPA+FiLM experiments that showed the fingerprint is the bottleneck |

---

## The loop

**1. Claude hands you a variant set.** Already configured in the notebook's CONFIG cell. You
should not need to edit anything.

**2. You run `fingerprint_sweep.ipynb`.** Run all cells. Roughly 30-60 min at the default
sample size on 4 cores, depending on the round; the CONFIG cell prints a calibrated ETA before
anything expensive starts. One `.npz` per variant lands in `sweep_out/`, so an interrupted run
resumes within the same session.

A committed Kaggle run starts from an empty working directory, so `sweep_out/` does **not**
survive between commits. That is by design here — the protocol re-runs the champion rather than
caching it (see below), so nothing needs attaching. `SEED_CACHE_FROM_INPUT` exists only to
resume a round that died partway; leave it `False` otherwise, or a stale cached champion will
silently replace the noise-floor measurement.

**3. You paste back the two tables.** That is all that's needed:
- `MAIN COMPARISON`
- `SESSION CONVERGENCE`

Paste the run log too if something errored, otherwise the tables are enough.

**4. Claude reads them and returns one of three things:**
- a new variant set for the next round, with the reasoning
- "this one is good enough — promote it" (see below)
- "stop, this isn't working, here's why"

**5. Claude updates `DIARY.md`** with the round's numbers and interpretation before proposing
the next one, so the history survives even if the conversation doesn't.

---

## Design rules for each round

**Wide rounds, few rounds.** Your compute runs unattended, your attention does not. Every round
tests 4–9 variants covering *different* hypotheses.

**From round 4: one factor at a time, many factors per round.** Each variant is the champion
plus exactly ONE change, so an effect is attributable. Confounded variants (round 2's `r3_vth`
moved both `r_exc` and `vth_rest`) cost a comparison and are not worth the saved run. Screen
broadly first, then combine only the knobs that individually moved something.

**Every round includes a deliberate replicate** — two variants with identical config, listed in
`REPLICATES`. It costs one variant's runtime and buys both the EER noise floor and the
per-fingerprint reproducibility measure, without which "this variant is better" is a guess.
Round 2 estimated the floor at 0.0003 from a single sample; round 3's accidental replicate
showed it is ~0.003. Never trust a single estimate of it again.

**Carry the reigning champion, and re-run it — do not cache it.** Every round includes the
best variant so far as its reference row, recomputed from scratch. Beaten variants are not
re-run; they stay in `DIARY.md` as history.

Two reasons this beats caching. The clip set is fixed by `SEED`, so the champion's number stays
comparable to its earlier value in the diary. And because the hidden neurons carry unseeded
noise (`sigma_noise * xi`), re-running it measures the **run-to-run noise floor** — the number
that says whether a gap between two new variants is real. A cached row cannot provide that.
It also means Kaggle's lack of a cache across commits costs nothing: nothing needs attaching.

The cost is one variant's runtime per round (~7 min at the default sample size).

**Re-anchor when the sample changes.** Dropping the original `baseline` means slow drift is
only checkable against the diary. That is sound while `SEED`, `N_SPEAKERS`, `SESSIONS_PER_SPK`,
`UTTS_PER_SESSION` and the corpus are unchanged. If any of them change, re-run `baseline` once
in that round to re-anchor, because absolute EERs are no longer comparable to earlier rounds.

**Ranking, not measurement.** 60 speakers is enough to rank variants that share identical clips.
It is not enough to quote an absolute EER. Two variants within a hair of each other are tied —
don't pick a winner between them.

---

## Promotion: what happens when a variant wins

The sweep is a **cheap proxy**, not the goal. A winner must survive the real test:

1. Regenerate **one full shard** with the winning settings (`prepare_fingerprints.ipynb`,
   with the variant's `n_epochs` / `r_exc` / `vth_rest`).
2. Run `../ecapa_film_snn/ecapa_film_snn.ipynb` **cell 14 Part B** → fingerprint-alone EER.
3. **Target: below 0.28** (from the current 0.345).

Only if that passes: cell 15 fusion test, then a short FiLM training run at 15–20 epochs.

---

## The gate — revised 2026-09-03

**Original gate (rounds 1-3):** after 2-3 rounds, either fingerprint-alone EER is trending
toward 0.28 or the direction closes.

**That gate was reached and reopened by the user's decision after round 3.** Round 3 ended at
-16.2% against the -19% needed, with the trend decelerating (-6.3% -> -14.4% -> -16.2%), and I
recommended stopping and running the promotion test. The user has time for more sweeps and chose
to continue instead. Recorded here so the change is deliberate rather than drift.

**What changed with it.** Rounds 1-3 only ever varied four hyperparameters, because only four
were wired up; the other ~25 were hardcoded. From round 4 the core exposes all of them
(`_fp_sweep_core.DEFAULTS`), so "we tried everything" is now a statement that can actually be
earned. That is the justification for continuing: the space was never searched, only four
corners of it were.

**Revised stopping rule.** The loop stops when **two consecutive rounds** produce no variant
beating the reigning champion by more than **2x the measured noise floor**.

**The rule fired after round 6 (2026-09-03) and the loop is closed.** Rounds 5 and 6 both came
back empty. Final champion: `n_epochs=2, r_exc=5, vth_rest=0.45, clip_ms=8000, sust_gain=1.0`,
**EER 0.3330 from a 0.4042 baseline (-17.6%)** at 2.3x less compute. The -19% needed for the
0.28 target was not reached. Full accounting in `DIARY.md` under "Loop closed — final state".

Judge against the **pooled** noise floor (0.0031 over five estimates), never a single round's
draw — round 2's lucky 0.0003 produced conclusions that had to be re-checked, and round 6's
0.0021 would have manufactured two spurious winners. That is a stopping
rule the search itself can trigger, rather than a round count. The -19% / EER 0.28 target is
unchanged and still the thing that decides whether any of this ships.

**The promotion test remains the only measurement that answers the real question** and has still
never been run. It costs one shard regeneration plus cell 14 Part B. **With the loop closed it is
now the only remaining step**, and `DIARY.md`'s "Loop closed" entry gives the four-step recipe:
regenerate one shard at the champion (or `vth_nexc`) settings, store only
`in_weights + input_activity`, run cell 14 Part B against 0.345 / 0.28, then add nuisance
projection and re-measure.

---

## What not to spend runs on

- Full ECAPA training runs, until fingerprint-alone EER improves. Cells 14 and 15 answer the
  same question in minutes instead of 14 hours.
- Multi-seed FiLM sweeps. Resolving a 0.0014 effect would take ~16 paired seeds (~110 GPU-hours).
- Per-component FiLM ablations, for the same reason.
- Chasing weight saturation. It cannot happen in this architecture (see
  `FINGERPRINT_IMPROVE.md` §2), so its absence means nothing.
- `sigma_noise`. Round 4 settled it: 3x less noise moved EER by 0.4x the floor, while the
  replicate showed 15.2% of the fingerprint does not survive a re-run. That irreproducible part
  carries no speaker information, so the within-utterance variance term is real acoustic content.
- Making the L1 column normalisation stricter in *time*. Round 4's `norm_dt_5` bound a budget
  that otherwise regrows to 2.79 against a 1.0455 limit, and scored a tie. The budget matters
  only through how much drive it permits, not as a competition mechanism.

---

## If the session-convergence table shows a large gap

Then the design premise holds — same-session utterances really do converge — and **session-level
averaging becomes the most promising move available**, likely more so than any architecture change
in the sweep. Flag it and Claude will redirect the next round accordingly rather than continuing
to tune knobs.

**This fired in round 1** (gap 0.21). Reading the three pair types as a variance decomposition
gave speaker 0.11 / session 0.21 / per-utterance noise 0.68, so round 2 was redirected away from
knob-tuning and onto the 0.68 noise term. Note the caveat that emerged with it: session averaging
needs several utterances of one session at test time, which the real application does not have.
The transferable form is removing the nuisance subspace, which the sweep now reports as
`held-out` vs `proj-20`.

**Retired as a steering metric after round 4.** The decomposition is now anti-correlated with the
outcome: `inh_weak` had the round's worst speaker/(session+noise) ratio and still won, and the
champion `enc_sust` won by 6x the noise floor at an essentially unchanged ratio. The gain comes
from the *shape* of the speaker subspace, not its share of variance. Keep reading the table — it
is cheap and the session gap is still worth watching — but steer on **`dead` and EER**, which
round 4 measured as correlated at **r = +0.96** across all nine variants.
