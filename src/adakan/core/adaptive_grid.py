import torch
import torch.nn as nn
import torch.nn.functional as F
from .hybrid_basis import HybridBasisFunction

class GradientStatistics:
    def __init__(self, decay=0.9):
        self.decay = decay
        self.running_grad_var = None
    def update(self, grads):
        if self.running_grad_var is None:
            self.running_grad_var = torch.zeros_like(grads)
        curr_var = grads.pow(2)
        self.running_grad_var = self.decay * self.running_grad_var + (1 - self.decay) * curr_var
    def get_importance(self):
        return self.running_grad_var

class AdaptiveGridKANLayer(nn.Module):
    def __init__(self, in_features, out_features, initial_grid=5, max_grid=20):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initial_grid = initial_grid
        self.max_grid = max_grid
        self.edges = nn.ModuleDict()
        self.grid_sizes = {}
        for i in range(out_features):
            for j in range(in_features):
                edge_id = f"{i}_{j}"
                self.edges[edge_id] = HybridBasisFunction(spline_grid=initial_grid)
                self.grid_sizes[edge_id] = initial_grid
        self.grad_stats = GradientStatistics()
        
    def forward(self, x):
        batch_size = x.size(0)
        output = torch.zeros(batch_size, self.out_features, device=x.device)
        for i in range(self.out_features):
            for j in range(self.in_features):
                edge_id = f"{i}_{j}"
                output[:, i] += self.edges[edge_id](x[:, j])
        return output

    def update_gradient_stats(self):
        pass # Simplified for restore

    def refine_grids(self, top_k_ratio=0.2):
        pass # Simplified for restore
