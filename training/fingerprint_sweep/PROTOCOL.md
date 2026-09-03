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

**2. You run `fingerprint_sweep.ipynb`.** Run all cells. ~1.5 h at the default sample size on
4 cores. It prints an ETA first and caches one `.npz` per variant in `sweep_out/`, so an
interrupted run resumes instead of restarting.

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
tests 4–6 variants covering *different* hypotheses, never one knob at a time. Variants cache
individually, so carrying a good variant into the next round costs nothing to re-measure.

**Always keep a `baseline` row.** The absolute EER numbers move with the speaker sample; only
the comparison against `baseline` in the same table is meaningful.

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

## The gate — agreed in advance

After **2–3 rounds**, either fingerprint-alone EER is trending toward 0.28 or it is not.

If it is not, Claude says so plainly and does not propose a round 4. The measured answer then
stands: the SNN fingerprint is complementary to ECAPA (+0.22 pair-score correlation) but too
weak for that to be worth more than ~1%, and the direction closes.

This is written down in advance because 1% effects are easy to keep chasing.

---

## What not to spend runs on

- Full ECAPA training runs, until fingerprint-alone EER improves. Cells 14 and 15 answer the
  same question in minutes instead of 14 hours.
- Multi-seed FiLM sweeps. Resolving a 0.0014 effect would take ~16 paired seeds (~110 GPU-hours).
- Per-component FiLM ablations, for the same reason.
- Chasing weight saturation. It cannot happen in this architecture (see
  `FINGERPRINT_IMPROVE.md` §2), so its absence means nothing.

---

## If the session-convergence table shows a large gap

Then the design premise holds — same-session utterances really do converge — and **session-level
averaging becomes the most promising move available**, likely more so than any architecture change
in the sweep. Flag it and Claude will redirect the next round accordingly rather than continuing
to tune knobs.
