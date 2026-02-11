import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import LagAttention, TemporalModulator, gumbel_sigmoid_sample

# Simple B-Spline Function for a single edge
class BSplineFunction(nn.Module):
    def __init__(self, grid_size=10, spline_order=3, scale_noise=0.1, scale_base=1.0, grid_eps=0.02, grid_range=[-1, 1]):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        
        # Grid parameters
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(1, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid) # [1, G + 2k + 1]
        
        # Trainable coefficients
        self.coef = nn.Parameter(
            (scale_noise * (torch.rand(1, grid_size + spline_order) - 0.5))
            * scale_base
        )
        
    def b_splines(self, x):
        """
        Compute B-spline bases for x.
        x: [batch_size]
        Returns: [batch_size, num_coeffs]
        """
        # x shape adjustment
        if x.dim() == 1:
            x = x.unsqueeze(-1)
            
        assert x.dim() == 2 and x.size(1) == 1
        
        
        grid = self.grid
        # x = x.expand(-1, grid.size(1)) # Incorrect expansion causing shape mismatch
        
        k = self.spline_order
        
        # Basis order 0
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        
        # Recursive Cox-de Boor
        for i in range(1, k + 1):
            # bases at iteration i has shape [batch, grid_len - i]
            # We need to compute [batch, grid_len - i - 1]
            
            # term 1
            left_num = x - grid[:, :-(i+1)]
            left_den = grid[:, i:-1] - grid[:, :-(i+1)]
            term1 = (left_num / (left_den + 1e-8)) * bases[:, :-1]
            
            # term 2
            right_num = grid[:, i+1:] - x
            right_den = grid[:, i+1:] - grid[:, 1:-i]
            term2 = (right_num / (right_den + 1e-8)) * bases[:, 1:]
            
            bases = term1 + term2
            
        return bases

    def forward(self, x):
        # x is [batch] or [batch, 1]
        # Normalize/Clamp x to grid range? 
        # KAN usually expects inputs in [-1, 1]. We should use Tanh or similar before?
        # Or Just clamp.
        original_shape = x.shape
        x = x.view(-1, 1)
        
        # Optional: Dynamic grid update (not implementing here for simplicity)
        x_clamped = torch.clamp(x, self.grid_range[0], self.grid_range[1])
        
        bases = self.b_splines(x_clamped) # [Batch, Coeffs]
        
        # y = sum(c_i * B_i(x))
        y = torch.matmul(bases, self.coef.t()) # [Batch, 1]
        
        return y.view(original_shape)

class CausalStructure(nn.Module):
    def __init__(self, num_nodes, init_scale=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        # Initialize logits near 0 (prob ~ 0.5) or slightly negative to encourage sparsity start
        self.adj_logits = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        nn.init.uniform_(self.adj_logits, -init_scale, init_scale)

    def forward(self, temperature=1.0, hard=False):
        """
        Returns the adjacency matrix Probabilities (or sampled mask).
        """
        if self.training:
            # Gumbel-Sigmoid for differentiable discrete sampling
            return gumbel_sigmoid_sample(self.adj_logits, temperature, hard=hard)
        else:
            # Deterministic/Soft during Inference
            return torch.sigmoid(self.adj_logits)

    def get_adj(self):
        """Returns raw probabilities for analysis"""
        return torch.sigmoid(self.adj_logits)

class CDKANLayer(nn.Module):
    def __init__(self, in_features, out_features, max_lag=10, grid_size=5, learn_structure=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_lag = max_lag
        
        self.edge_functions = nn.ModuleDict()
        self.lag_attention = nn.ModuleDict()
        self.modulators = nn.ModuleDict()
        
        self.learn_structure = learn_structure
        
        # Centralized Structure Learner
        if learn_structure:
            self.causal_structure = CausalStructure(num_nodes=max(in_features, out_features))

        for i in range(out_features):
            for j in range(in_features):
                edge_id = f"{i}_{j}"
                
                # 1. KAN Function
                self.edge_functions[edge_id] = BSplineFunction(grid_size=grid_size)
                
                # 2. Lag Attention
                self.lag_attention[edge_id] = LagAttention(max_lag=max_lag)
                
                # 3. Modulator
                self.modulators[edge_id] = TemporalModulator(input_dim=1)
                    
        self.register_buffer('temperature', torch.tensor(1.0))
        
    def forward(self, x_history):
        # x_history: [batch, seq_len, in_features]
        batch_size, seq_len, _ = x_history.shape
        output = torch.zeros(batch_size, self.out_features, device=x_history.device)
        t_curr = seq_len - 1
        
        # Get Structure Mask [Out, In]
        if self.learn_structure:
            full_adj = self.causal_structure(self.temperature, hard=self.training)
            # Slice to relevant dimensions if non-square
            adj_mask = full_adj[:self.out_features, :self.in_features]
        else:
            adj_mask = torch.ones(self.out_features, self.in_features, device=x_history.device)

        for i in range(self.out_features):
            edge_accum = 0.0
            for j in range(self.in_features):
                edge_id = f"{i}_{j}"
                
                # A. Lag Attention
                w_lag = self.lag_attention[edge_id].get_weights()
                history_window = x_history[:, t_curr-self.max_lag : t_curr+1, j] 
                history_window = torch.flip(history_window, dims=[1])
                x_lagged = (history_window * w_lag).sum(dim=1)
                
                # B. KAN Function
                y_edge = self.edge_functions[edge_id](x_lagged)
                
                # C. Temporal Modulation
                alpha = self.modulators[edge_id](x_history[:, :, j:j+1])
                
                # D. Apply Mask
                mask = adj_mask[i, j]
                
                edge_accum = edge_accum + y_edge * alpha.squeeze(-1) * mask
                
            output[:, i] = edge_accum
            
        return output
    
    def get_adjacency(self):
        """Returns the learned soft adjacency matrix [Out, In]"""
        if self.learn_structure:
            full_adj = self.causal_structure.get_adj()
            return full_adj[:self.out_features, :self.in_features]
        return torch.ones(self.out_features, self.in_features)
    
    def get_feature_importance(self):
        """
        Compute edge importance based on:
        Importance = Probability * Mean(Abs(Coefficients))
        """
        importance = torch.zeros(self.out_features, self.in_features)
        
        # Move probabilities to same device/cpu for calculation
        # It's safer to keep on CPU for numpy conversion later
        if self.learn_structure:
            adj_probs = self.get_adjacency().detach().cpu() 
        else:
            adj_probs = torch.ones(self.out_features, self.in_features)
            
        with torch.no_grad():
            for i in range(self.out_features):
                for j in range(self.in_features):
                    edge_id = f"{i}_{j}"
                    
                    # 1. Structure Probability
                    prob = adj_probs[i, j]
                        
                    # 2. Coefficient Magnitude
                    coef_norm = self.edge_functions[edge_id].coef.abs().mean().cpu()
                    
                    importance[i, j] = prob * coef_norm
                    
        return importance

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
