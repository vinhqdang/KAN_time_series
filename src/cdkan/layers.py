import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import LagAttention, TemporalModulator, gumbel_sigmoid_sample

# Simple B-Spline Function for a single edge
class BSplineFunction(nn.Module):
    def __init__(self, grid_size=10, order=3):
        super().__init__()
        self.grid_size = grid_size

        self.order = order
        # Coeffs: grid + order
        self.coeffs = nn.Parameter(torch.randn(grid_size + order) * 0.1)
        
    def forward(self, x):
        # x: [batch]
        # Simplified linear B-spline for prototype speed (can upgrade to cubic)
        # Mapping x (assumed normalized/bounded) to grid
        # For robust implementation we should use the same logic as efficient-kan or pykan
        # Here implementing a simple grid lookup for "Univariate Function"
        
        # Norm to [0, 1] loosely
        x_clamped = torch.sigmoid(x) # continuous relaxation 
        
        # Grid index
        idx = x_clamped * (self.grid_size - 1)
        k = idx.long()
        w1 = idx - k
        w0 = 1.0 - w1
        
        # Safe lookup
        c0 = self.coeffs[torch.clamp(k, 0, len(self.coeffs)-1)]
        c1 = self.coeffs[torch.clamp(k+1, 0, len(self.coeffs)-1)]
        
        return c0 * w0 + c1 * w1

class CDKANLayer(nn.Module):
    def __init__(self, in_features, out_features, max_lag=10, grid_size=5, learn_structure=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_lag = max_lag
        
        self.edge_functions = nn.ModuleDict()
        self.lag_attention = nn.ModuleDict()
        self.modulators = nn.ModuleDict()
        self.edge_logits = nn.ParameterDict() # For structure learning
        
        self.learn_structure = learn_structure
        
        for i in range(out_features):
            for j in range(in_features):
                edge_id = f"{i}_{j}"
                
                # 1. KAN Function
                self.edge_functions[edge_id] = BSplineFunction(grid_size=grid_size)
                
                # 2. Lag Attention
                self.lag_attention[edge_id] = LagAttention(max_lag=max_lag)
                
                # 3. Modulator
                self.modulators[edge_id] = TemporalModulator(input_dim=1)
                
                # 4. Structure Prob (Logits)
                if learn_structure:
                    # Init logits to higher value to encourage edges initially (sigmoid(2.0) ~= 0.88)
                    self.edge_logits[edge_id] = nn.Parameter(torch.tensor(2.0))
                    
        self.register_buffer('temperature', torch.tensor(1.0))
        
    def forward(self, x_history):
        # x_history: [batch, seq_len, in_features]
        # seq_len must be >= max_lag + 1
        
        batch_size, seq_len, _ = x_history.shape
        output = torch.zeros(batch_size, self.out_features, device=x_history.device)
        
        # Current time step is the last one
        t_curr = seq_len - 1
        
        for i in range(self.out_features):
            edge_accum = 0.0
            for j in range(self.in_features):
                edge_id = f"{i}_{j}"
                
                # A. Lag Attention
                w_lag = self.lag_attention[edge_id].get_weights() # [max_lag+1]
                
                # Extract lagged input: sum(w[tau] * x[t - tau])
                # We can vectorize this extraction
                # Slice history: x_history[:, t_curr-max_lag : t_curr+1, j]
                # Reversing the slice to match lags [0, 1, ..., max_lag]
                history_window = x_history[:, t_curr-self.max_lag : t_curr+1, j] # [Batch, lag+1]
                history_window = torch.flip(history_window, dims=[1]) # Now index 0 is lag 0 (t_curr), index 1 is lag 1...
                
                x_lagged = (history_window * w_lag).sum(dim=1) # [Batch]
                
                # B. KAN Function
                y_edge = self.edge_functions[edge_id](x_lagged)
                
                # C. Temporal Modulation
                # We use the full history series for the modulator RNN
                alpha = self.modulators[edge_id](x_history[:, :, j:j+1]) # [Batch, 1]
                
                # D. Structure Learning
                mask = 1.0
                if self.learn_structure:
                    if self.training:
                         mask = gumbel_sigmoid_sample(self.edge_logits[edge_id], self.temperature)
                    else:
                         mask = torch.sigmoid(self.edge_logits[edge_id]) > 0.5
                         mask = mask.float()
                
                edge_accum = edge_accum + y_edge * alpha.squeeze(-1) * mask.squeeze(-1)
                
            output[:, i] = edge_accum
            
        return output

    def get_causal_graph(self):
        adj = torch.zeros(self.out_features, self.in_features)
        lags = torch.zeros(self.out_features, self.in_features)
        
        with torch.no_grad():
            for i in range(self.out_features):
                for j in range(self.in_features):
                    edge_id = f"{i}_{j}"
                    if self.learn_structure:
                        prob = torch.sigmoid(self.edge_logits[edge_id])
                        adj[i, j] = prob
                    else:
                        adj[i, j] = 1.0
                        
                    lags[i, j] = self.lag_attention[edge_id].get_expected_lag()
        return adj, lags

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=20):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.edge_functions = nn.ModuleDict()
        
        for i in range(out_features):
            for j in range(in_features):
                edge_id = f"{i}_{j}"
                self.edge_functions[edge_id] = BSplineFunction(grid_size=grid_size)
                
    def forward(self, x):
        # x: [Batch, In]
        batch, _ = x.shape
        out = torch.zeros(batch, self.out_features, device=x.device)
        for i in range(self.out_features):
            term = 0.0
            for j in range(self.in_features):
                edge_id = f"{i}_{j}"
                term = term + self.edge_functions[edge_id](x[:, j])
            out[:, i] = term
        return out
