import torch
import torch.nn as nn

class DormantEdgePool:
    def __init__(self):
        self.dormant_edges = {}
    def store(self, edge_id, state_dict):
        self.dormant_edges[edge_id] = {k: v.cpu().clone() for k, v in state_dict.items()}
    def restore(self, edge_id):
        if edge_id in self.dormant_edges: return self.dormant_edges.pop(edge_id)
        return None

def prune_edges(layer, threshold=0.01):
    pruned = []
    for edge_id, basis in layer.edges.items():
        score = torch.sigmoid(basis.alpha) * basis.spline_coeffs.norm()
        if score < threshold:
            if not hasattr(layer, 'dormant_pool'): layer.dormant_pool = DormantEdgePool()
            layer.dormant_pool.store(edge_id, basis.state_dict())
            basis.alpha.data.fill_(-10.0)
            pruned.append(edge_id)
    return pruned
