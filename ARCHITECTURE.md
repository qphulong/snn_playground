# SNN Playground — Architecture

## Overview

A modular, config-driven research platform for Spiking Neural Networks (SNNs).
Designed to support rapidly changing experiment designs:

- **Multiple architectures**: dense, sparse, graph-based networks
- **Pluggable components**: layers, connectivity patterns, learning rules
- **Lazy-loading UI**: weight matrices loaded in viewport tiles, not all at once
- **Checkpoint system**: save, resume, and compare training runs
- **Extensible visualizations**: register new UI components without touching core code

**Core Principle**: Code contains no hyperparameters. All experiment settings live in YAML files.

---

## Directory Structure

```
snn_playground/
├── snn_core/                     # Core SNN simulation modules
│   ├── audio_encoding.py         # Audio → input current matrix
│   ├── network.py                # Architecture-agnostic Brian2 network
│   ├── training.py               # Training loop with checkpointing
│   ├── dataset.py                # Dataset loader
│   ├── layers/
│   │   ├── base.py               # Abstract Layer class
│   │   ├── lif.py                # AdaptiveLIF, SimpleLIF
│   │   ├── connectivity.py       # Dense, Sparse, Graph connectivity
│   │   └── custom/               # Drop-in custom layer types
│   └── learning_rules/
│       ├── base.py               # Abstract LearningRule class
│       ├── stdp.py               # STDP with homeostatic normalization
│       └── custom/               # Drop-in custom learning rules
│
├── config/
│   ├── schema.py                 # Dataclass schemas for all config fields
│   ├── loader.py                 # YAML → ExperimentConfig (deep-merges base)
│   └── defaults/
│       ├── base.yaml             # Base defaults (all experiments inherit)
│       └── experiments/
│           ├── exp_stdp_baseline.yaml
│           └── _template_experiment.yaml
│
├── storage/
│   ├── checkpoint_manager.py     # Save/load full training state (pickle)
│   └── model_storage.py          # HDF5 chunked weight storage for lazy loading
│
├── web_api/
│   ├── app.py                    # FastAPI app + AppState + startup
│   ├── routes/
│   │   ├── model.py              # /api/model/* endpoints
│   │   ├── metrics.py            # /api/metrics/* endpoints
│   │   └── training.py           # /api/training/* endpoints
│   ├── schemas/                  # Pydantic response models
│   └── serializers/              # NumPy-aware JSON serializer
│
├── web_ui/                       # React + Vite frontend
│   └── src/
│       ├── App.jsx               # Tab-based layout
│       ├── services/api.js       # API client (axios)
│       ├── components/
│       │   ├── visualizations/
│       │   │   ├── VisualizationRegistry.js   # Plugin system
│       │   │   ├── WeightHeatmap.jsx          # Paged heatmap
│       │   │   ├── WeightEvolution.jsx        # Training curves
│       │   │   └── SpikeStats.jsx             # Spike activity
│       │   └── panels/
│       │       ├── ModelPanel.jsx             # Architecture summary
│       │       └── TrainingPanel.jsx          # Checkpoint control
│
├── scripts/
│   ├── train.py                  # CLI: run training from config
│   └── validate_config.py        # CLI: validate config without running
│
├── notebooks/                    # Exploratory notebooks
│   ├── main.ipynb
│   └── experiments/
│
└── checkpoints/                  # Saved training states
```

---

## Data Flow

```
YAML Config
    │
    ▼
config/loader.py  (deep-merges base + experiment)
    │
    ▼
ExperimentConfig  (validated dataclass tree)
    │
    ├──► SNNNetwork._build_from_config()
    │         Resolves layer types from LAYER_TYPES registry
    │         Resolves connectivity from CONNECTIVITY_TYPES registry
    │         Resolves learning rules from LEARNING_RULE_TYPES registry
    │
    └──► Trainer.run_training(dataset)
              │
              ├── audio_encoding.compute_input_current()
              │         Audio → (N_in, T) current matrix
              │
              ├── SNNNetwork.run_simulation(input_currents, weights)
              │         Creates Brian2 NeuronGroups + Synapses fresh each call
              │         Returns updated weights + spike data
              │
              ├── LearningRule.apply_homeostasis(weights)
              │
              └── CheckpointManager.save(...)
                        Pickle: weights + history + config + sample_idx

Checkpoints ──► FastAPI ──► React UI
                  │               │
                  │  lazy region  │  fetches only visible viewport
                  └───────────────┘
```

---

## Config System

Every experiment is a YAML file. The loader deep-merges it with `config/defaults/base.yaml`, so experiments only declare what differs.

**Example structure** (`config/experiments/exp_stdp_baseline.yaml`):
```yaml
name: "STDP Baseline (900→900)"
network:
  layers:
    - name: "input"
      type: "adaptive_lif"
      N_neurons: 900
      neuron_params: { tau_m: 10, tau_a: 100, beta: 0.2, v_th: 1.0 }
    - name: "hidden"
      type: "simple_lif"
      N_neurons: 900
      neuron_params: { tau_m: 10, v_th: 1.0 }
  synapses:
    - name: "input_to_hidden"
      src_layer: "input"
      tgt_layer: "hidden"
      connectivity: { type: "dense" }
      learning_rule:
        type: "stdp"
        taupre: 20
        wmax: 0.3
        homeostasis_norm: 225.0
training:
  max_samples: 30
  checkpoint_interval: 5
```

---

## Extensibility

### New Layer Type

1. Create `snn_core/layers/custom/my_layer.py` subclassing `Layer`
2. Register in `snn_core/layers/__init__.py`:
   ```python
   LAYER_TYPES['my_layer'] = MyLayer
   ```
3. Use in config: `type: "my_layer"`

### New Connectivity Pattern

1. Create `snn_core/layers/connectivity.py` subclassing `Connectivity`
   (or add to `custom/`)
2. Register in `CONNECTIVITY_TYPES`
3. Use in config: `connectivity: { type: "my_pattern", ... }`

### New Learning Rule

1. Create `snn_core/learning_rules/custom/my_rule.py` subclassing `LearningRule`
2. Register in `LEARNING_RULE_TYPES`
3. Use in config: `learning_rule: { type: "my_rule", ... }`

### New Visualization

1. Create `web_ui/src/components/visualizations/MyViz.jsx`
2. At the bottom, call:
   ```js
   registerVisualization('my_viz', {
     component: MyViz,
     label: 'My Visualization',
     description: 'What it shows.',
   });
   ```
3. Import it in `VisualizationRegistry.js`
4. It will automatically appear in the Visualizations tab — no other changes needed.

---

## Running

### Training (CLI)

```bash
# New training run
python scripts/train.py \
  --config config/experiments/exp_stdp_baseline.yaml \
  --dataset datasets/vox1_small

# Resume from checkpoint
python scripts/train.py \
  --config config/experiments/exp_stdp_baseline.yaml \
  --dataset datasets/vox1_small \
  --resume-from ckpt_sample_000030

# Validate config without running
python scripts/validate_config.py config/experiments/exp_stdp_baseline.yaml
```

### Web API

```bash
uvicorn web_api.app:app --reload --port 8000
# API docs: http://localhost:8000/docs

# Override defaults via environment variables
SNN_EXPERIMENT_CONFIG=config/experiments/my_exp.yaml \
SNN_CHECKPOINT_DIR=checkpoints/my_exp \
uvicorn web_api.app:app --reload
```

### Web UI

```bash
cd web_ui
npm run dev
# Opens at http://localhost:5173
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Brian2 objects recreated each simulation | Brian2 state resets with `start_scope()`; weight matrix is the persistent state |
| HDF5 chunked storage | Enables loading only the visible viewport region for large weight matrices |
| YAML deep-merge | Experiments only specify overrides; base defaults are DRY |
| Visualization registry | New viz types added without touching App.jsx or routing |
| `vars(dataclass)` for config dicts | Layer/LearningRule constructors receive plain dicts, keeping them decoupled from the schema |
