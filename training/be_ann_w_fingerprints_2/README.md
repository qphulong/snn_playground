# be_ann_w_fingerprints_2

Speaker-verification pipeline built on **triplet-STDP** weight fingerprints. The extractor
is **identical to `training/tonotopic_plasticity_bound/train.py`** (triplet STDP with
Pfister–Gerstner slow detectors, soft-refractory-gated lateral inhibition, larger STDP
amplitudes) — unlike the older `be_ann_w_fingerprints/` pipeline, which is frozen on the
pair-STDP recipe.

Two Kaggle-oriented notebooks:

| notebook | what it does |
|---|---|
| `prepare_fingerprints.ipynb` | one Kaggle input shard folder → one fingerprint `.npz` |
| `ann_backend.ipynb` | loads all fingerprint shards, trains the ResNet + attentive-pool + ArcFace embedding, evaluates open-set |

Supporting modules (also embedded verbatim in the prepare notebook via `%%writefile`, so it
is self-contained on Kaggle):

- `_fingerprint_core_2.py` — the SNN extractor + fingerprint transform. Merges `train.py`'s
  architecture/hyperparams/mechanism with the collection/transform + silence-masking of
  `be_ann_w_fingerprints/_fingerprint_core.py`. Also hosts the multiprocessing worker helpers
  (`_init_worker`, `process_one`) so `spawn` workers can import them from a real module.
- `spike_encoding.py`, `audio_utils.py` — verbatim copies of `src/utils/*` (flat imports),
  so no repo upload is needed on Kaggle.

## 1. Generate fingerprints (per shard)

On Kaggle, open `prepare_fingerprints.ipynb`, edit the **config cell**, run all:

```python
INPUT_ROOT  = "/kaggle/input/datasets/qphulong/vox2-voices-200person-shard01"
OUTPUT_NAME = "vox2_200person_shard01_fingerprints.npz"   # vox1_... for the test set
PER_SESSION = 4          # ≤ 4 wavs per session, all sessions, all persons
WORKERS     = os.cpu_count()
```

Expected input layout: `{INPUT_ROOT}/{person_id}/{session_id}/{wav}.m4a` (VoxCeleb2) — `.wav`
also accepted. Output lands in `/kaggle/working/{OUTPUT_NAME}` with fields:

```
weights (N,4,33,384) f16 | input_activity (N,384) f16 | hidden_activity (N,384) f16
person_ids (N,) | record_ids (N,) | labels (N,)   # labels = "person/session/wav"
```

Requirements: `gammatone` (pip cell) and `ffmpeg` (preinstalled on Kaggle). Each wav is a
4-epoch triplet-STDP run (~tens of seconds on CPU), parallelised across `WORKERS` via
`spawn`. **No mid-shard resume** — size shards to finish inside the Kaggle session limit.

Run once per shard; download each npz and bundle them all into one Kaggle dataset for step 2.

## 2. Train the ANN backend

Open `ann_backend.ipynb`, point `INPUT_DIR` at the uploaded fingerprint dataset:

```python
DEV_GLOB  = os.path.join(INPUT_DIR, "vox2_*_fingerprints.npz")   # train (closed-set)
TEST_GLOB = os.path.join(INPUT_DIR, "vox1_*_fingerprints.npz")   # eval  (open-set)
```

- **dev / train** = VoxCeleb2 shards; **test / eval** = VoxCeleb1 shards. The two speaker
  sets (`id0xxxx` vs `id1xxxx`) are disjoint, so the open-set assumption holds automatically.
- Each compact sample expands to `(9,33,384)` at model time (4 weight + 4 input-rate + 1
  hidden-rate channels). `FINGERPRINT_MODE` ablates channel groups (weights vs rates) with a
  zero-variance fill, keeping the architecture/param-count identical across modes.
- Metrics: EER, Rank-1/5, mAP (cosine), plus a t-SNE of the test embeddings.

## Relationship to the other directories

- `tonotopic_plasticity_bound/train.py` — source of truth for the SNN architecture. Keep
  `_fingerprint_core_2.py`'s hyperparameters/equations in sync if that file changes.
- `be_ann_w_fingerprints/` — the previous pipeline (pair-STDP core, VoxCeleb1 dev/test split,
  manifest-driven sharded generator). Same fingerprint format and ANN model.
