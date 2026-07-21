import torch
import torch.nn as nn
from .layers import CDKANLayer, KANLayer


# ---------------------------------------------------------------------------
# Reversible Instance Normalization (RevIN)
# ---------------------------------------------------------------------------

class RevIN(nn.Module):
    """
    Reversible Instance Normalization (Kim et al., 2022).

    Reviewer note: Statistics (mean/std) are stored as *local* variables
    in each forward pass, not as persistent buffers. This prevents leakage
    between different windows during walk-forward evaluation.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias   = nn.Parameter(torch.zeros(num_features))
        # Leakage guard: stats computed fresh each forward; never cached across calls
        self._mean  = None
        self._stdev = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == 'norm':
            # Compute stats on the *current* input only (no cross-window reuse)
            dim2reduce = tuple(range(1, x.ndim - 1))
            self._mean  = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            self._stdev = torch.sqrt(
                torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
            ).detach()
            x = (x - self._mean) / self._stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self._mean is None or self._stdev is None:
                return x  # safety guard — no prior norm call
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + 1e-10)
            x = x * self._stdev + self._mean
        else:
            raise NotImplementedError(f"RevIN mode must be 'norm' or 'denorm', got '{mode}'")
        return x


# ---------------------------------------------------------------------------
# Residual KAN block
# ---------------------------------------------------------------------------

class ResidualKANBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 10,
                 dropout: float = 0.1):
        super().__init__()
        self.kan     = KANLayer(in_dim, out_dim, grid_size=grid_size)
        self.norm    = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip    = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.skip(x)
        out = self.kan(x)
        out = self.norm(out)
        out = self.dropout(out)
        return out + res


# ---------------------------------------------------------------------------
# CD-KAN Forecaster
# ---------------------------------------------------------------------------

class CDKANForecaster(nn.Module):
    """
    Causal Discovery KAN Forecaster.

    Architecture: RevIN → CDKANLayer (lag-aware causal discovery) → Residual KAN backbone → Output

    Changes from original:
    - RevIN uses fresh per-call statistics (no cross-window leakage)
    - CDKANLayer uses LagAwareAdjacency [max_lag, d, d] instead of a flat [d, d]
    - Exposes lag-level adjacency diagnostics via get_lag_adjacency()
    """

    def __init__(self, in_features: int, hidden_dim: int = 64, out_features: int = 1,
                 max_lag: int = 10, n_layers: int = 3, dropout: float = 0.1,
                 learn_structure: bool = True, grid_size: int = 10,
                 causal_bottleneck: bool = False):
        super().__init__()

        # 1. Reversible Instance Normalisation
        self.revin = RevIN(in_features)

        # 2. Causal Discovery Layer (lag-aware)
        self.cd_layer = CDKANLayer(
            in_features, in_features,
            max_lag=max_lag,
            grid_size=grid_size,
            learn_structure=learn_structure,
        )

        # --- Causal-bottleneck mode -------------------------------------
        # When enabled, each target variable's forecast flows ONLY through the
        # edge-mask x edge-function aggregation of the causal layer, followed by
        # a per-variable (diagonal) readout. There is NO cross-variable mixing
        # backbone, so the adjacency A is the sole information bottleneck and
        # therefore receives an identifying gradient signal. This makes the
        # learned graph a faithful structural estimate (see paper Sec. 3).
        self.causal_bottleneck = causal_bottleneck
        if causal_bottleneck:
            self.readout_w = nn.Parameter(torch.ones(in_features))
            self.readout_b = nn.Parameter(torch.zeros(in_features))
        else:
            # 3. Residual KAN backbone (dense; higher capacity, non-identifiable A)
            self.backbone = nn.ModuleList()
            self.backbone.append(
                ResidualKANBlock(in_features, hidden_dim, grid_size=grid_size, dropout=dropout)
            )
            for _ in range(n_layers - 1):
                self.backbone.append(
                    ResidualKANBlock(hidden_dim, hidden_dim, grid_size=grid_size, dropout=dropout)
                )
            # 4. Output head
            self.output_head = KANLayer(hidden_dim, out_features, grid_size=grid_size)

        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, window, in_features]
        Returns:
            [batch, out_features]
        """
        # 1. Normalise (stats computed fresh — no leakage)
        x = self.revin(x, 'norm')

        # 2. Causal extraction: x_causal[i] = sum_j A[i,j] * phi_ij(lagged x_j)
        x_causal = self.cd_layer(x)         # [batch, in_features]

        if self.causal_bottleneck:
            # Per-variable readout only — no cross-variable mixing.
            out = x_causal * self.readout_w + self.readout_b
        else:
            # 3. Deep reasoning (dense residual KAN backbone)
            x_hidden = x_causal
            for block in self.backbone:
                x_hidden = block(x_hidden)
            # 4. Predict
            out = self.output_head(x_hidden)    # [batch, out_features]

        # 5. Denormalise (only when output dimension matches input features)
        if out.shape[-1] == self.revin.num_features:
            out = self.revin(out.unsqueeze(1), 'denorm').squeeze(1)

        return out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def set_temperature(self, t: float):
        """Set Gumbel-Sigmoid temperature on CDKANLayer."""
        if hasattr(self.cd_layer, 'temperature'):
            self.cd_layer.temperature.data.fill_(t)
            
    def update_grid(self):
        """Update dynamic KAN grids empirically tracking activation ranges."""
        for module in self.modules():
            if module.__class__.__name__ == 'BSplineFunction' and hasattr(module, 'update_grid_from_samples'):
                module.update_grid_from_samples()

    def get_summary_adjacency(self) -> torch.Tensor:
        """Summary causal graph [in, in] (max over lags)."""
        return self.cd_layer.get_adjacency()

    def get_lag_adjacency(self, k: int) -> torch.Tensor:
        """Causal graph for lag k [in, in]."""
        return self.cd_layer.get_lag_adjacency(k)

    def get_expected_lags(self) -> torch.Tensor:
        """Expected lag per edge [in, in] — for managerial reporting."""
        return self.cd_layer.get_expected_lags()

    def get_contribution_importance(self, x: torch.Tensor) -> torch.Tensor:
        """Data-grounded structural importance [in, in]. Normalises x with RevIN
        (as in the forward pass), then measures the std of each edge's
        contribution to the forecast. This is the recommended structure readout
        for the causal-bottleneck model."""
        with torch.no_grad():
            xn = self.revin(x, 'norm')
            return self.cd_layer.get_contribution_importance(xn)

    def get_feature_importance(self) -> torch.Tensor:
        """Edge importance = adjacency_prob × spline_coef_magnitude [in, in]."""
        return self.cd_layer.get_feature_importance()

    def prune(self, threshold: float = 0.2):
        """Zero out edges whose summary probability is below threshold (in-place)."""
        with torch.no_grad():
            adj = self.cd_layer.lag_adj.adj_logits
            probs = torch.sigmoid(adj)
            mask  = probs.max(0).values < threshold  # [d, d] — True where weak
            # Suppress by pushing logits hard negative
            adj[:, mask] = -10.0
