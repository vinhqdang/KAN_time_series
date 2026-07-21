"""
CausalKAN: an identifiable, scalable causal-discovery KAN for time series.

Design (see the manuscript, Sec. 3):
  * **Information bottleneck / component-wise.** Each target i is predicted ONLY
    through per-(cause j, lag h) B-spline edge functions of the inputs; there is
    no shared cross-variable backbone. The edge functions are the sole path from
    cause to effect, so their contribution is an identifying structural signal.
  * **Group-lasso, not stochastic gates.** Whole (i, j) edges are driven to zero
    by a group-lasso over their stacked lag coefficients; structure is read from
    each edge's *contribution* to the forecast (deterministic, stable).
  * **Per-lag resolution + optional acyclicity.** A separate spline per
    (effect, cause, lag) gives lag attribution; a NOTEARS penalty can be applied
    to the contemporaneous (lag-0) block when instantaneous edges are modelled.

This implementation is **fully vectorized**: all d*d*(#lags) B-spline edges are
evaluated as batched tensor ops sharing one grid, so it scales to d ~ 50 and runs
on GPU. It preserves the interface used by the benchmark scripts.
"""
import torch
import torch.nn as nn
from .losses import _h_notears


class CausalKAN(nn.Module):
    def __init__(self, d: int, max_lag: int = 3, grid_size: int = 8,
                 spline_order: int = 3, contemporaneous: bool = False,
                 grid_range=(-3.0, 3.0)):
        super().__init__()
        self.d = d
        self.max_lag = max_lag
        self.contemporaneous = contemporaneous
        self.lags = list(range(0 if contemporaneous else 1, max_lag + 1))
        self.L = len(self.lags)
        self.k = spline_order
        self.n_basis = grid_size + spline_order
        self.grid_size = grid_size

        # shared knot vector for every edge
        lo, hi = grid_range
        step = (hi - lo) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * step + lo)
        self.register_buffer("grid", grid)                    # [G + 2k + 1]
        self.grid_range = (lo, hi)

        # coefficients: [effect d, cause d, lag L, n_basis]
        self.coef = nn.Parameter(0.1 * (torch.rand(d, d, self.L, self.n_basis) - 0.5))
        self.bias = nn.Parameter(torch.zeros(d))

    # ---- B-spline bases for a batch of scalar inputs -------------------------
    def _bases(self, x):
        """x: [..., ] arbitrary shape of scalar inputs -> [..., n_basis]."""
        x = x.clamp(self.grid_range[0], self.grid_range[1]).unsqueeze(-1)   # [...,1]
        g = self.grid
        b = ((x >= g[:-1]) & (x < g[1:])).to(x.dtype)          # [..., G+2k]
        for i in range(1, self.k + 1):
            ln = x - g[:-(i + 1)]
            ld = g[i:-1] - g[:-(i + 1)]
            rn = g[i + 1:] - x
            rd = g[i + 1:] - g[1:-i]
            b = (ln / (ld + 1e-8)) * b[..., :-1] + (rn / (rd + 1e-8)) * b[..., 1:]
        return b                                               # [..., n_basis]

    # ---- gather the lag-aligned inputs from a window -------------------------
    def _lagged(self, x_hist):
        """x_hist: [B, W, d]  ->  [B, d, L] input value per (cause, lag)."""
        W = x_hist.shape[1]
        cols = []
        for h in self.lags:
            idx = max(0, min(W - 1, W - h if h >= 1 else W - 1))
            cols.append(x_hist[:, idx, :])                     # [B, d]
        return torch.stack(cols, dim=-1)                       # [B, d, L]

    def forward(self, x_hist):
        xin = self._lagged(x_hist)                             # [B, d(cause), L]
        bases = self._bases(xin)                               # [B, d, L, n_basis]
        # edge output phi_{i j h}(x_{jh}) = <bases[b,j,h,:], coef[i,j,h,:]>
        # predict[b,i] = bias[i] + sum_{j,h} <bases[b,j,h,:], coef[i,j,h,:]>
        pred = torch.einsum("bjhk,ijhk->bi", bases, self.coef) + self.bias
        return pred                                            # [B, d]

    # ---- penalties & read-outs ----------------------------------------------
    def group_lasso(self):
        # L2 norm of each (effect i, cause j) group (over lags & basis)
        g = torch.linalg.vector_norm(self.coef, dim=(2, 3))    # [d, d]
        return g.sum()

    @torch.no_grad()
    def _edge_contrib(self, x_hist):
        """Per-(effect, cause, lag) contribution over the batch: [B, d, d, L]."""
        xin = self._lagged(x_hist)                             # [B, d, L]
        bases = self._bases(xin)                               # [B, d, L, n_basis]
        # contribution of edge (i,j,h) to target i for each sample b
        return torch.einsum("bjhk,ijhk->bijh", bases, self.coef)  # [B, i, j, h]

    @torch.no_grad()
    def importance(self, x_hist):
        """[d, d] structural score = std over the batch of each edge's total
        (summed over lags) contribution. Row = effect i, col = cause j."""
        c = self._edge_contrib(x_hist).sum(dim=3)              # [B, i, j]
        return c.std(dim=0)                                    # [d, d]

    @torch.no_grad()
    def expected_lags(self, x_hist):
        """[d, d] contribution-weighted mean lag per edge."""
        c = self._edge_contrib(x_hist).std(dim=0)              # [i, j, h]
        lags = torch.tensor([float(h) for h in self.lags], device=c.device)
        num = (c * lags.view(1, 1, -1)).sum(-1)
        den = c.sum(-1).clamp(min=1e-8)
        return num / den                                       # [d, d]

    def contemporaneous_h(self, x_hist):
        """NOTEARS acyclicity residual on the lag-0 contemporaneous block."""
        if not self.contemporaneous:
            return torch.zeros((), device=self.coef.device)
        c = self._edge_contrib(x_hist)[:, :, :, 0].std(dim=0)  # [d, d] lag-0
        return _h_notears(c)
