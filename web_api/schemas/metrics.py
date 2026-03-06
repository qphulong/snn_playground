from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class TrainingHistoryResponse(BaseModel):
    num_samples: int
    metrics: Dict[str, List[float]]   # {metric_name: [values]}


class CheckpointMeta(BaseModel):
    id: str
    sample_idx: int
    timestamp: Optional[float]
    config_name: Optional[str]


class CheckpointListResponse(BaseModel):
    checkpoints: List[CheckpointMeta]


class TrainingStatusResponse(BaseModel):
    is_training: bool
    current_checkpoint: Optional[str]
    samples_trained: Optional[int]
