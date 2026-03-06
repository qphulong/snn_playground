from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class LayerInfo(BaseModel):
    name: str
    N: int
    type: str
    neuron_params: Dict[str, Any]


class SynapseInfo(BaseModel):
    name: str
    src: str
    tgt: str
    connectivity: str
    learning_rule: str


class ModelInfoResponse(BaseModel):
    config_name: str
    description: str
    layers: List[LayerInfo]
    synapses: List[SynapseInfo]


class WeightRegionResponse(BaseModel):
    synapse_name: str
    full_shape: List[int]
    region_bounds: Dict[str, List[int]]
    region: List[List[float]]
    stats: Dict[str, float]


class WeightStatsResponse(BaseModel):
    synapse_name: str
    per_tgt_mean: List[float]
    per_tgt_std: List[float]
    per_src_mean: List[float]
    per_src_std: List[float]
    global_mean: float
    global_std: float


class ConnectivityResponse(BaseModel):
    synapse_name: str
    connectivity_type: str
    src_layer: str
    tgt_layer: str
    num_connections: int
    src_indices: List[int]
    tgt_indices: List[int]
