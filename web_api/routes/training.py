from fastapi import APIRouter, HTTPException
from typing import Optional

from ..schemas.metrics import CheckpointListResponse, TrainingStatusResponse

router = APIRouter(prefix="/api/training", tags=["training"])


def _get_state():
    from ..app import app_state
    return app_state


@router.get("/status", response_model=TrainingStatusResponse)
async def get_training_status():
    state = _get_state()
    ckpts = state.checkpoint_mgr.list_checkpoints()

    if ckpts:
        meta = state.checkpoint_mgr.get_metadata(ckpts[-1])
        return {
            'is_training':        state.is_training,
            'current_checkpoint': ckpts[-1],
            'samples_trained':    meta.get('sample_idx'),
        }
    return {
        'is_training':        state.is_training,
        'current_checkpoint': None,
        'samples_trained':    None,
    }


@router.get("/checkpoints", response_model=CheckpointListResponse)
async def list_checkpoints():
    state   = _get_state()
    details = state.checkpoint_mgr.list_with_metadata()
    return {
        'checkpoints': [
            {
                'id':           d['id'],
                'sample_idx':   d.get('sample_idx', 0),
                'timestamp':    d.get('timestamp'),
                'config_name':  d.get('config_name'),
            }
            for d in details
        ]
    }


@router.post("/load-checkpoint")
async def load_checkpoint(checkpoint_id: str):
    """Load a checkpoint into the active session (updates current weights)."""
    state = _get_state()
    try:
        ckpt = state.checkpoint_mgr.load(checkpoint_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    state.weights = ckpt['weights']
    return {'status': 'loaded', 'checkpoint_id': checkpoint_id, 'sample_idx': ckpt.get('sample_idx')}


@router.post("/load-experiment")
async def load_experiment(config_path: str):
    """
    Load a new experiment config. Resets weights.
    Does NOT start training — call /start to begin.
    """
    import os
    from config.loader import load_config
    from snn_core.network import SNNNetwork

    state = _get_state()

    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail=f"Config not found: {config_path}")

    cfg     = load_config(config_path)
    network = SNNNetwork(cfg.network)

    state.experiment_config = cfg
    state.network            = network
    state.weights            = network.init_weights()

    return {
        'status':      'loaded',
        'experiment':  cfg.name,
        'layers':      network.layer_info(),
        'synapses':    network.synapse_info(),
    }
