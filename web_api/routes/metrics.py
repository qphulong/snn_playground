from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from ..schemas.metrics import TrainingHistoryResponse
from ..serializers.numpy_encoder import to_jsonable

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


def _get_state():
    from ..app import app_state
    return app_state


@router.get("/history", response_model=TrainingHistoryResponse)
async def get_training_history(checkpoint_id: Optional[str] = None):
    """
    Return full training history from checkpoint.
    All metrics are returned — the UI decides which to display.
    """
    state = _get_state()

    if checkpoint_id:
        ckpt = state.checkpoint_mgr.load(checkpoint_id)
    else:
        ckpt = state.checkpoint_mgr.load_latest()

    if ckpt is None:
        raise HTTPException(status_code=404, detail="No checkpoints found")

    history = ckpt.get('history', {})
    return {
        'num_samples': ckpt.get('sample_idx', 0),
        'metrics':     to_jsonable(history),
    }


@router.get("/weight-evolution")
async def get_weight_evolution(
    synapse_name: str,
    metric: str = Query('mean_w', description="'mean_w' or 'std_w' or 'delta_w'"),
):
    """Return a single scalar metric over training time."""
    state = _get_state()
    ckpt  = state.checkpoint_mgr.load_latest()
    if ckpt is None:
        raise HTTPException(status_code=404, detail="No checkpoints found")

    key = f"{synapse_name}/{metric}"
    history = ckpt.get('history', {})
    if key not in history:
        raise HTTPException(status_code=404, detail=f"Metric '{key}' not found in history")

    return {'synapse_name': synapse_name, 'metric': metric, 'values': history[key]}


@router.get("/spike-stats")
async def get_spike_stats(layer_name: str):
    """Return spike count per sample for a given layer."""
    state = _get_state()
    ckpt  = state.checkpoint_mgr.load_latest()
    if ckpt is None:
        raise HTTPException(status_code=404, detail="No checkpoints found")

    key = f"{layer_name}/num_spikes"
    history = ckpt.get('history', {})
    if key not in history:
        raise HTTPException(status_code=404, detail=f"Layer '{layer_name}' not found in history")

    return {'layer_name': layer_name, 'num_spikes_per_sample': history[key]}


@router.get("/specific-weight")
async def get_specific_weight_evolution(
    synapse_name: str,
    src_idx: int = Query(..., ge=0),
    tgt_idx: int = Query(..., ge=0),
):
    """
    Return how a single synapse weight w[src_idx, tgt_idx] evolved over training.
    Requires save_weights_history: true in experiment config.
    """
    state = _get_state()
    if state.weight_history_store is None:
        raise HTTPException(
            status_code=404,
            detail="Weight history not available. Set save_weights_history: true in config and retrain."
        )

    sample_indices, values = state.weight_history_store.get_weight_evolution(
        synapse_name, src_idx, tgt_idx
    )

    if not sample_indices:
        raise HTTPException(status_code=404, detail=f"No weight history found for synapse '{synapse_name}'")

    return {
        'synapse_name': synapse_name,
        'src_idx':      src_idx,
        'tgt_idx':      tgt_idx,
        'sample_indices': sample_indices,
        'values':         values,
    }
