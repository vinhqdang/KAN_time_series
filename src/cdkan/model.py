import torch
import torch.nn as nn
from .layers import CDKANLayer, KANLayer

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        Reversible Instance Normalization for solving distribution shift.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + 1e-10)
        x = x * self.stdev
        x = x + self.mean
        return x

class ResidualKANBlock(nn.Module):
    def __init__(self, in_dim, out_dim, grid_size=10, dropout=0.1):
        super().__init__()
        self.kan = KANLayer(in_dim, out_dim, grid_size=grid_size)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # Residual connection
        res = self.skip(x)
        # KAN Path
        out = self.kan(x)
        out = self.norm(out)
        out = self.dropout(out)
        return out + res

class CDKANForecaster(nn.Module):
    """
    SOTA Causal Discovery KAN Forecaster.
    Features:
    - Residual Deep KAN Backbone
    """
    def __init__(self, in_features, hidden_dim=64, out_features=1, max_lag=10, n_layers=3, dropout=0.1, learn_structure=True):
        super().__init__()
        
        # 1. Reversible Instance Norm
        self.revin = RevIN(in_features)
        
        # 2. Causal Structure Discovery Layer
        # Maps [Batch, Window, Vars] -> [Batch, Vars] (Aggregation over time via causal lags)
        self.cd_layer = CDKANLayer(in_features, in_features, max_lag=max_lag, grid_size=10, learn_structure=learn_structure) # Grid=10 for precision
        
        # 3. Mixing / Deep Reasoning Backbone
        # Input to backbone is [Batch, Vars] (the "Current State" extracted from history)
        # We process this state with a deep Residual KAN
        self.backbone = nn.ModuleList()
        # First projection from Vars -> Hidden
        self.backbone.append(ResidualKANBlock(in_features, hidden_dim, grid_size=10, dropout=dropout))
        
        # Hidden -> Hidden layers
        for _ in range(n_layers - 1):
            self.backbone.append(ResidualKANBlock(hidden_dim, hidden_dim, grid_size=10, dropout=dropout))
            
        # 4. Output Projection
        self.output_head = KANLayer(hidden_dim, out_features, grid_size=10)
        
    def forward(self, x):
        # x: [Batch, Window, Features]
        
        # 1. Normalize
        x = self.revin(x, 'norm')
        
        # 2. Causal Extraction (The "Causal Twist")
        # This layer learns WHICH past time points matter for the current state.
        # Output: [Batch, Features]
        x_causal = self.cd_layer(x)
        
        # 3. Deep Reasoning
        x_hidden = x_causal
        for block in self.backbone:
            x_hidden = block(x_hidden)
            
        # 4. Prediction
        out = self.output_head(x_hidden)
        
        # 5. Denormalize
        # Note: RevIN denorm expects same shape as input usually, but we are predicting different shape?
        # Usually RevIN is for [Batch, Window, Vars] -> [Batch, Window, Vars].
        # Here we predict [Batch, Out_Vars]. 
        # If Out_Vars == In_Features, we can use RevIN.
        # Assuming Out_Features (Prediction Target) is a subset or same as In_Features.
        # For this dataset, we predict ALL assets (Next Step), so Out == In.
        if out.shape[-1] == self.revin.num_features:
            out = out.unsqueeze(1)
            out = self.revin(out, 'denorm')
            out = out.squeeze(1)
            
        return out
        
    def set_temperature(self, t):
        if hasattr(self.cd_layer, 'temperature'):
            self.cd_layer.temperature.data.fill_(t)
            
    def prune(self, density=0.2):
        self.cd_layer.prune_edges(target_density=density)
