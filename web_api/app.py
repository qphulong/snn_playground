"""
FastAPI application entry point.

All application state lives in `app_state` — a single object imported by routes.
Routes never hardcode experiment structure; all data comes from loaded config/checkpoint.

Run:
    uvicorn web_api.app:app --reload --port 8000
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.loader import load_config
from snn_core.network import SNNNetwork
from storage.checkpoint_manager import CheckpointManager
from storage.weight_history_store import WeightHistoryStore

from web_api.routes import model as model_routes
from web_api.routes import metrics as metrics_routes
from web_api.routes import training as training_routes


# ── Application state ──────────────────────────────────────────────────────

@dataclass
class AppState:
    experiment_config:   Optional[object] = None
    network:             Optional[SNNNetwork] = None
    weights:             Optional[Dict[str, np.ndarray]] = None
    checkpoint_mgr:      Optional[CheckpointManager] = None
    weight_history_store: Optional[WeightHistoryStore] = None
    is_training:         bool = False


app_state = AppState()

# ── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="SNN Playground API",
    description="REST API for SNN research — model inspection, training control, metrics",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(model_routes.router)
app.include_router(metrics_routes.router)
app.include_router(training_routes.router)


# ── Startup ────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """
    Load default experiment on startup.
    Tries to resume from latest checkpoint if one exists.
    """
    default_config = os.environ.get(
        "SNN_EXPERIMENT_CONFIG",
        "config/experiments/exp_stdp_baseline.yaml",
    )
    checkpoint_dir = os.environ.get("SNN_CHECKPOINT_DIR", "checkpoints")

    app_state.checkpoint_mgr = CheckpointManager(checkpoint_dir)

    wh_path = os.path.join(checkpoint_dir, 'weight_history.h5')
    if os.path.exists(wh_path):
        app_state.weight_history_store = WeightHistoryStore(wh_path)

    if os.path.exists(default_config):
        cfg     = load_config(default_config)
        network = SNNNetwork(cfg.network)

        app_state.experiment_config = cfg
        app_state.network           = network

        # Try to restore weights from latest checkpoint
        ckpt = app_state.checkpoint_mgr.load_latest()
        if ckpt is not None:
            app_state.weights = ckpt['weights']
            print(f"Loaded checkpoint: sample {ckpt.get('sample_idx')}")
        else:
            app_state.weights = network.init_weights()
            print("No checkpoint found — initialised fresh weights")

        print(f"Experiment: {cfg.name}")
        for info in network.layer_info():
            print(f"  Layer  '{info['name']}': {info['N']} neurons ({info['type']})")
        for info in network.synapse_info():
            print(f"  Synapse '{info['name']}': {info['src']} → {info['tgt']} ({info['connectivity']})")
    else:
        print(f"Config not found at {default_config}. Load experiment via POST /api/training/load-experiment")


@app.get("/")
async def root():
    return {"status": "ok", "experiment": getattr(app_state.experiment_config, 'name', None)}
