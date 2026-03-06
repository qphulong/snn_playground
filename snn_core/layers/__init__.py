from .lif import AdaptiveLIF, SimpleLIF
from .connectivity import DenseConnectivity, SparseConnectivity, GraphConnectivity

LAYER_TYPES = {
    'adaptive_lif': AdaptiveLIF,
    'simple_lif': SimpleLIF,
}

CONNECTIVITY_TYPES = {
    'dense': DenseConnectivity,
    'sparse': SparseConnectivity,
    'graph': GraphConnectivity,
}
