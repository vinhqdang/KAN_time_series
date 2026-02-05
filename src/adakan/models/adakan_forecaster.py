import torch
import torch.nn as nn
from ..core.adaptive_grid import AdaptiveGridKANLayer

class AdaKANForecaster(nn.Module):
    def __init__(self, window, hidden=32, horizon=1, initial_grid=5, max_grid=20):
        super().__init__()
        self.layer1 = AdaptiveGridKANLayer(window, hidden, initial_grid, max_grid)
        self.layer2 = AdaptiveGridKANLayer(hidden, horizon, initial_grid, max_grid)
        
    def forward(self, x):
        x1 = torch.tanh(self.layer1(x))
        out = self.layer2(x1)
        return out
