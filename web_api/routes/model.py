from fastapi import APIRouter, Query, HTTPException
from typing import Optional
import numpy as np

from ..schemas.model import (
    ModelInfoResponse, WeightRegionResponse, WeightStatsResponse, ConnectivityResponse,
)
from ..serializers.numpy_encoder import to_jsonable

router = APIRouter(prefix="/api/model", tags=["model"])


def _get_state():
    """Import app state lazily to avoid circular imports."""
    from ..app import app_state
    return app_state


# ── Model info ─────────────────────────────────────────────────────────────

@router.get("/info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Return network structure from loaded config.
    All fields come from the config file — nothing is hardcoded.
    """
    state = _get_state()
    if state.network is None:
        raise HTTPException(status_code=404, detail="No experiment loaded")

    config = state.network.config

    layers = [
        {'name': l.name, 'N': l.N_neurons, 'type': l.type, 'neuron_params': l.neuron_params}
        for l in config.layers
    ]
    synapses = [
        {
            'name': s.name,
            'src': s.src_layer,
            'tgt': s.tgt_layer,
            'connectivity': s.connectivity.type,
            'learning_rule': s.learning_rule.type,
        }
        for s in config.synapses
    ]

    return {
        'config_name': state.experiment_config.name,
        'description': state.experiment_config.description,
        'layers':   layers,
        'synapses': synapses,
    }


# ── Weight region (lazy loading) ────────────────────────────────────────────

@router.get("/weights", response_model=WeightRegionResponse)
async def get_weight_region(
    synapse_name: str,
    row_start: int = Query(0, ge=0),
    row_end:   Optional[int] = Query(None),
    col_start: int = Query(0, ge=0),
    col_end:   Optional[int] = Query(None),
):
    """
    Return a rectangular region of a weight matrix.

    The UI requests only the visible viewport. The server loads only
    that HDF5 chunk — full matrices are never loaded unless requested.
    """
    state = _get_state()
    if state.weights is None or synapse_name not in state.weights:
        raise HTTPException(status_code=404, detail=f"Synapse '{synapse_name}' not found")

    w = state.weights[synapse_name]
    r_end = row_end if row_end is not None else w.shape[0]
    c_end = col_end if col_end is not None else w.shape[1]
    r_end = min(r_end, w.shape[0])
    c_end = min(c_end, w.shape[1])

    region = w[row_start:r_end, col_start:c_end]

    return {
        'synapse_name':  synapse_name,
        'full_shape':    list(w.shape),
        'region_bounds': {'rows': [row_start, r_end], 'cols': [col_start, c_end]},
        'region':        to_jsonable(region),
        'stats': {
            'mean': float(region.mean()),
            'std':  float(region.std()),
            'min':  float(region.min()),
            'max':  float(region.max()),
        },
    }


# ── Weight stats ────────────────────────────────────────────────────────────

@router.get("/weights/stats", response_model=WeightStatsResponse)
async def get_weight_stats(synapse_name: str):
    """Per-neuron incoming/outgoing weight statistics."""
    state = _get_state()
    if state.weights is None or synapse_name not in state.weights:
        raise HTTPException(status_code=404, detail=f"Synapse '{synapse_name}' not found")

    w = state.weights[synapse_name]
    return {
        'synapse_name':  synapse_name,
        'per_tgt_mean':  to_jsonable(w.mean(axis=0)),
        'per_tgt_std':   to_jsonable(w.std(axis=0)),
        'per_src_mean':  to_jsonable(w.mean(axis=1)),
        'per_src_std':   to_jsonable(w.std(axis=1)),
        'global_mean':   float(w.mean()),
        'global_std':    float(w.std()),
    }


# ── Connectivity ────────────────────────────────────────────────────────────

@router.get("/connectivity", response_model=ConnectivityResponse)
async def get_connectivity(synapse_name: str):
    """
    Return edge list for the requested synapse.
    Used by the graph visualization component.
    """
    state = _get_state()
    if state.network is None:
        raise HTTPException(status_code=404, detail="No experiment loaded")

    syn_info = state.network.synapses.get(synapse_name)
    if syn_info is None:
        raise HTTPException(status_code=404, detail=f"Synapse '{synapse_name}' not found")

    src_layer = state.network.layers[syn_info['src']]
    tgt_layer = state.network.layers[syn_info['tgt']]
    src_idx, tgt_idx = syn_info['connectivity'].get_connections(src_layer.N, tgt_layer.N)

    return {
        'synapse_name':     synapse_name,
        'connectivity_type': type(syn_info['connectivity']).__name__,
        'src_layer':        syn_info['src'],
        'tgt_layer':        syn_info['tgt'],
        'num_connections':  int(len(src_idx)),
        'src_indices':      to_jsonable(src_idx[:5000]),   # cap for dense networks
        'tgt_indices':      to_jsonable(tgt_idx[:5000]),
    }
