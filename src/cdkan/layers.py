import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import LagAttention, TemporalModulator, gumbel_sigmoid_sample


# ---------------------------------------------------------------------------
# B-Spline function for a single KAN edge
# ---------------------------------------------------------------------------

class BSplineFunction(nn.Module):
    """Learnable B-spline function for one KAN edge: y = sum_k c_k B_k(x)."""

    def __init__(self, grid_size=10, spline_order=3, scale_noise=0.1, scale_base=1.0,
                 grid_eps=0.02, grid_range=(-1, 1)):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = list(grid_range)

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(1, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)  # [1, G + 2k + 1]

        self.coef = nn.Parameter(
            (scale_noise * (torch.rand(1, grid_size + spline_order) - 0.5)) * scale_base
        )

    def b_splines(self, x):
        """Compute B-spline bases.  x: [batch, 1]  ->  [batch, num_coeffs]"""
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        assert x.dim() == 2 and x.size(1) == 1

        grid = self.grid
        k = self.spline_order
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        for i in range(1, k + 1):
            left_num  = x - grid[:, :-(i+1)]
            left_den  = grid[:, i:-1] - grid[:, :-(i+1)]
            term1     = (left_num / (left_den + 1e-8)) * bases[:, :-1]
            right_num = grid[:, i+1:] - x
            right_den = grid[:, i+1:] - grid[:, 1:-i]
            term2     = (right_num / (right_den + 1e-8)) * bases[:, 1:]
            bases     = term1 + term2

        return bases

    def forward(self, x):
        original_shape = x.shape
        x = x.view(-1, 1)
        x_clamped = torch.clamp(x, self.grid_range[0], self.grid_range[1])
        bases = self.b_splines(x_clamped)            # [batch, C]
        y = torch.matmul(bases, self.coef.t())       # [batch, 1]
        return y.view(original_shape)


# ---------------------------------------------------------------------------
# Causal structure modules
# ---------------------------------------------------------------------------

class CausalStructure(nn.Module):
    """2-D adjacency (d × d) – retained for backward compatibility."""

    def __init__(self, num_nodes, init_scale=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.adj_logits = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        nn.init.uniform_(self.adj_logits, -init_scale, init_scale)

    def forward(self, temperature=1.0, hard=False):
        if self.training:
            return gumbel_sigmoid_sample(self.adj_logits, temperature, hard=hard)
        return torch.sigmoid(self.adj_logits)

    def get_adj(self):
        return torch.sigmoid(self.adj_logits)


class LagAwareAdjacency(nn.Module):
    """
    3-D lag-indexed adjacency: adj_logits[lag, effect, cause].

    This correctly separates each lag's contribution instead of conflating
    all lags into one d×d matrix (the key reviewer critique). Acyclicity
    is naturally satisfied for lagged edges (time's arrow); we only apply
    the NOTEARS h(W) constraint on the *summary graph* (see losses.py).
    """

    def __init__(self, num_nodes: int, max_lag: int, init_scale: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.max_lag   = max_lag
        # Shape: [max_lag, d, d]  — lag 0 means lag-1 in actual time
        self.adj_logits = nn.Parameter(
            torch.zeros(max_lag, num_nodes, num_nodes)
        )
        nn.init.uniform_(self.adj_logits, -init_scale, init_scale)

    def forward(self, temperature: float = 1.0, hard: bool = False):
        """Return stochastic/soft adjacency [max_lag, d, d]."""
        if self.training:
            return gumbel_sigmoid_sample(self.adj_logits, temperature, hard=hard)
        return torch.sigmoid(self.adj_logits)

    def get_lag_adj(self, k: int) -> torch.Tensor:
        """Soft adjacency for a single lag k: [d, d]."""
        return torch.sigmoid(self.adj_logits[k])

    def get_summary_adj(self) -> torch.Tensor:
        """Summary graph: max over lags [d, d] — used for evaluation & plotting."""
        return torch.sigmoid(self.adj_logits).max(dim=0).values

    def get_expected_lag(self) -> torch.Tensor:
        """
        Expected lag for each edge (i, j):
            E[lag | edge i<-j] = sum_k k * P_k(i,j) / sum_k P_k(i,j)
        Returns [d, d] tensor of expected lags.
        """
        probs = torch.sigmoid(self.adj_logits)          # [L, d, d]
        lags  = torch.arange(1, self.max_lag + 1,
                              dtype=probs.dtype,
                              device=probs.device).view(-1, 1, 1)
        weighted = (probs * lags).sum(0)                # [d, d]
        normaliser = probs.sum(0).clamp(min=1e-8)       # [d, d]
        return weighted / normaliser


# ---------------------------------------------------------------------------
# CDKANLayer with lag-aware masking
# ---------------------------------------------------------------------------

class CDKANLayer(nn.Module):
    """
    Causal Discovery KAN Layer.

    For each (effect i, cause j) pair:
      1. Compute lag-specific attention weights w_k over lags 1..max_lag.
      2. Weight each lag k by the learned per-lag adjacency mask A[k, i, j].
      3. Pass the masked, weighted sum through a per-edge B-spline function.
      4. Scale by a temporal modulation factor (GRU-based).

    The adjacency uses LagAwareAdjacency so each lag has independent
    parameters, fully addressing the reviewer's temporal structure critique.
    """

    def __init__(self, in_features: int, out_features: int, max_lag: int = 10,
                 grid_size: int = 5, learn_structure: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.max_lag      = max_lag

        self.edge_functions = nn.ModuleDict()
        self.lag_attention  = nn.ModuleDict()
        self.modulators     = nn.ModuleDict()
        self.learn_structure = learn_structure

        if learn_structure:
            self.lag_adj = LagAwareAdjacency(
                num_nodes=max(in_features, out_features),
                max_lag=max_lag,
            )

        for i in range(out_features):
            for j in range(in_features):
                eid = f"{i}_{j}"
                self.edge_functions[eid] = BSplineFunction(grid_size=grid_size)
                self.lag_attention[eid]  = LagAttention(max_lag=max_lag)
                self.modulators[eid]     = TemporalModulator(input_dim=1)

        self.register_buffer('temperature', torch.tensor(1.0))

    def forward(self, x_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_history: [batch, seq_len, in_features]
        Returns:
            [batch, out_features]
        """
        batch_size, seq_len, _ = x_history.shape
        output = torch.zeros(batch_size, self.out_features, device=x_history.device)

        # Get adjacency masks: [max_lag, out, in] (stochastic during training)
        if self.learn_structure:
            full_adj = self.lag_adj(self.temperature.item(), hard=self.training)
            # Slice to the actual in/out dimensions
            adj_mask = full_adj[:, :self.out_features, :self.in_features]  # [L, out, in]
        else:
            adj_mask = torch.ones(self.max_lag, self.out_features, self.in_features,
                                  device=x_history.device)

        for i in range(self.out_features):
            edge_accum = torch.zeros(batch_size, device=x_history.device)

            for j in range(self.in_features):
                eid = f"{i}_{j}"

                # ------ A. Lag-weighted sum using per-lag mask ------
                # lag_attention gives shared weights over lags [0..max_lag]
                w_lag = self.lag_attention[eid].get_weights()          # [max_lag+1]

                # Extract history aligned to max_lag steps
                t_start = max(0, seq_len - self.max_lag - 1)
                history_window = x_history[:, t_start:, j]             # [batch, L+1]
                history_window = torch.flip(history_window, dims=[1])  # [batch, L+1] newest first

                # Per-lag mask for edge (i, j): [max_lag]
                lag_mask = adj_mask[:, i, j]                           # [max_lag]

                # Combine attention weight with lag-specific adjacency mask
                # We index the first max_lag lags (exclude lag-0 = current step)
                L = min(self.max_lag, history_window.shape[1] - 1)
                combined_weights = w_lag[1:L+1] * lag_mask[:L]         # [L]
                combined_weights = combined_weights / (combined_weights.sum() + 1e-8)

                x_lagged = (history_window[:, 1:L+1] * combined_weights).sum(dim=1)  # [batch]

                # ------ B. KAN non-linear transform ------
                y_edge = self.edge_functions[eid](x_lagged)             # [batch]

                # ------ C. Temporal modulation ------
                alpha = self.modulators[eid](x_history[:, :, j:j+1])  # [batch, 1]

                edge_accum = edge_accum + y_edge * alpha.squeeze(-1)

            output[:, i] = edge_accum

        return output

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_adjacency(self) -> torch.Tensor:
        """Summary adjacency [out, in] for evaluation/plotting."""
        if self.learn_structure:
            return self.lag_adj.get_summary_adj()[:self.out_features, :self.in_features]
        return torch.ones(self.out_features, self.in_features)

    def get_lag_adjacency(self, k: int) -> torch.Tensor:
        """Lag-k adjacency [out, in]."""
        if self.learn_structure:
            return self.lag_adj.get_lag_adj(k)[:self.out_features, :self.in_features]
        return torch.ones(self.out_features, self.in_features)

    def get_expected_lags(self) -> torch.Tensor:
        """Expected lag per edge [out, in] — useful for managerial reporting."""
        if self.learn_structure:
            return self.lag_adj.get_expected_lag()[:self.out_features, :self.in_features]
        return torch.zeros(self.out_features, self.in_features)

    def get_feature_importance(self) -> torch.Tensor:
        """
        Edge importance = summary_prob[i,j] × mean(|spline coefficients|).
        Returns [out, in] tensor.
        """
        importance = torch.zeros(self.out_features, self.in_features)
        adj_probs = self.get_adjacency().detach().cpu() if self.learn_structure \
            else torch.ones(self.out_features, self.in_features)

        with torch.no_grad():
            for i in range(self.out_features):
                for j in range(self.in_features):
                    eid = f"{i}_{j}"
                    prob      = adj_probs[i, j]
                    coef_norm = self.edge_functions[eid].coef.abs().mean().cpu()
                    importance[i, j] = prob * coef_norm

        return importance


# ---------------------------------------------------------------------------
# Plain KAN layer (no structure learning) for the backbone
# ---------------------------------------------------------------------------

class KANLayer(nn.Module):
    """Standard KAN layer: each (out, in) pair has its own B-spline function."""

    def __init__(self, in_features: int, out_features: int, grid_size: int = 20):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.edge_functions = nn.ModuleDict()

        for i in range(out_features):
            for j in range(in_features):
                self.edge_functions[f"{i}_{j}"] = BSplineFunction(grid_size=grid_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, in_features]  ->  [batch, out_features]"""
        batch = x.shape[0]
        out = torch.zeros(batch, self.out_features, device=x.device)
        for i in range(self.out_features):
            term = torch.zeros(batch, device=x.device)
            for j in range(self.in_features):
                term = term + self.edge_functions[f"{i}_{j}"](x[:, j])
            out[:, i] = term
        return out
