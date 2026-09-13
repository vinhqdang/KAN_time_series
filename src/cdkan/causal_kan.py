"""
SPADE (SPline Additive-noise DAG Estimation): an identifiable, scalable
causal-discovery KAN for time series. The headline instantaneous-DAG learner is
CausalKANInstant, exported below as the public alias ``SPADE``.

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
    """
    edge_type="spline" (default): original B-spline edges, G+p coefficients each.
    edge_type="mlp": ablation variant (Reviewer 2) — each per-(effect, cause,
    lag) edge is a small 1->mlp_hidden->1 MLP (tanh) instead of a B-spline,
    with mlp_hidden chosen so the per-edge parameter count is closely matched
    to the spline's. Everything else (group-lasso, training loop, forward
    interface, importance()/expected_lags() readouts) is identical between the
    two, so swapping this flag isolates the edge-parameterization contribution
    from the architecture/optimizer/regularization, per Reviewer 2.
    """
    def __init__(self, d: int, max_lag: int = 3, grid_size: int = 8,
                 spline_order: int = 3, contemporaneous: bool = False,
                 grid_range=(-3.0, 3.0), edge_type: str = "spline",
                 mlp_hidden: int = 3):
        super().__init__()
        self.d = d
        self.max_lag = max_lag
        self.contemporaneous = contemporaneous
        self.lags = list(range(0 if contemporaneous else 1, max_lag + 1))
        self.L = len(self.lags)
        self.k = spline_order
        self.n_basis = grid_size + spline_order
        self.grid_size = grid_size
        self.edge_type = edge_type

        if edge_type == "spline":
            # shared knot vector for every edge
            lo, hi = grid_range
            step = (hi - lo) / grid_size
            grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * step + lo)
            self.register_buffer("grid", grid)                    # [G + 2k + 1]
            self.grid_range = (lo, hi)

            # coefficients: [effect d, cause d, lag L, n_basis]  (n_basis=11 for
            # the paper's default grid_size=8, spline_order=3 -> 11 params/edge)
            self.coef = nn.Parameter(0.1 * (torch.rand(d, d, self.L, self.n_basis) - 0.5))
        elif edge_type == "mlp":
            # Matched-capacity MLP edge: 1 -> mlp_hidden -> 1, tanh, with bias.
            # mlp_hidden=3 (default) -> 3+3 + 3+1 = 10 params/edge, the closest
            # match to the spline's 11 (a 1->5->1 MLP would be 16, farther off).
            self.mlp_hidden = mlp_hidden
            self.w1 = nn.Parameter(0.5 * (torch.rand(d, d, self.L, mlp_hidden) - 0.5))
            self.b1 = nn.Parameter(torch.zeros(d, d, self.L, mlp_hidden))
            self.w2 = nn.Parameter(0.5 * (torch.rand(d, d, self.L, mlp_hidden) - 0.5))
            self.b2 = nn.Parameter(torch.zeros(d, d, self.L))
        else:
            raise ValueError(f"unknown edge_type: {edge_type!r} (expected 'spline' or 'mlp')")

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

    def _mlp_forward(self, xin):
        """Per-edge 1->mlp_hidden->1 MLP (tanh), evaluated for every (effect i,
        cause j, lag h) at once. xin: [B, d(cause), L] -> [B, i, j, h] (the
        edge's scalar output, i.e. the MLP analogue of the spline's
        <bases[b,j,h,:], coef[i,j,h,:]> inner product, pre-sum over j,h)."""
        hpre = torch.einsum("bjh,ijhk->bijhk", xin, self.w1) + self.b1
        hidden = torch.tanh(hpre)
        out = torch.einsum("bijhk,ijhk->bijh", hidden, self.w2) + self.b2
        return out                                             # [B, i, j, h]

    def forward(self, x_hist):
        xin = self._lagged(x_hist)                             # [B, d(cause), L]
        if self.edge_type == "spline":
            bases = self._bases(xin)                           # [B, d, L, n_basis]
            # edge output phi_{i j h}(x_{jh}) = <bases[b,j,h,:], coef[i,j,h,:]>
            # predict[b,i] = bias[i] + sum_{j,h} <bases[b,j,h,:], coef[i,j,h,:]>
            pred = torch.einsum("bjhk,ijhk->bi", bases, self.coef) + self.bias
        else:
            pred = self._mlp_forward(xin).sum(dim=(2, 3)) + self.bias
        return pred                                            # [B, d]

    # ---- penalties & read-outs ----------------------------------------------
    def group_lasso(self):
        # L2 norm of each (effect i, cause j) group (over lags & the edge's
        # own parameters -- basis coefficients for splines, all four weight/
        # bias tensors for MLP edges), identical semantics in both cases.
        if self.edge_type == "spline":
            g = torch.linalg.vector_norm(self.coef, dim=(2, 3))    # [d, d]
        else:
            stacked = torch.cat(
                [self.w1, self.b1, self.w2, self.b2.unsqueeze(-1)], dim=-1
            )                                                       # [d,d,L,3h+1]
            g = torch.linalg.vector_norm(stacked, dim=(2, 3))       # [d, d]
        return g.sum()

    @torch.no_grad()
    def _edge_contrib(self, x_hist):
        """Per-(effect, cause, lag) contribution over the batch: [B, d, d, L]."""
        xin = self._lagged(x_hist)                             # [B, d, L]
        if self.edge_type == "spline":
            bases = self._bases(xin)                           # [B, d, L, n_basis]
            # contribution of edge (i,j,h) to target i for each sample b
            return torch.einsum("bjhk,ijhk->bijh", bases, self.coef)  # [B, i, j, h]
        return self._mlp_forward(xin)                          # [B, i, j, h]

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
            return torch.zeros((), device=self.bias.device)
        c = self._edge_contrib(x_hist)[:, :, :, 0].std(dim=0)  # [d, d] lag-0
        return _h_notears(c)


class CausalKANContemp(nn.Module):
    """
    Full SVAR SPADE model with INSTANTANEOUS (contemporaneous) edges.

    Predicts each variable via a nonlinear structural equation:
        x_{t,i} = b_i + sum_{j != i} phi0_{ij}(x_{t,j})          (instantaneous)
                       + sum_{h>=1} sum_j phih_{ij}(x_{t-h,j})    (lagged)
    Self-loops are masked; the contemporaneous block W0 is constrained to be a DAG
    by the NOTEARS acyclicity penalty h(W0)=tr(e^{W0 o W0})-d, dualized with the
    Augmented Lagrangian Method. This is a nonlinear, KAN-based analogue of
    DYNOTEARS: the acyclicity constraint is exercised on the instantaneous graph,
    while lagged edges are unconstrained (ordered by time). Vectorized; fast.
    """
    def __init__(self, d, max_lag=2, grid_size=8, spline_order=3, grid_range=(-3.0, 3.0)):
        super().__init__()
        self.d = d; self.max_lag = max_lag; self.k = spline_order
        self.n_basis = grid_size + spline_order
        lo, hi = grid_range; step = (hi - lo) / grid_size
        g = torch.arange(-spline_order, grid_size + spline_order + 1) * step + lo
        self.register_buffer("grid", g); self.grid_range = (lo, hi)
        self.c0 = nn.Parameter(0.1 * (torch.rand(d, d, self.n_basis) - 0.5))
        self.cl = nn.Parameter(0.1 * (torch.rand(d, d, max_lag, self.n_basis) - 0.5))
        self.bias = nn.Parameter(torch.zeros(d))
        self.register_buffer("selfmask", (1 - torch.eye(d)).unsqueeze(-1))

    def _bases(self, x):
        x = x.clamp(*self.grid_range).unsqueeze(-1); g = self.grid
        b = ((x >= g[:-1]) & (x < g[1:])).to(x.dtype)
        for i in range(1, self.k + 1):
            b = ((x - g[:-(i + 1)]) / (g[i:-1] - g[:-(i + 1)] + 1e-8)) * b[..., :-1] \
                + ((g[i + 1:] - x) / (g[i + 1:] - g[1:-i] + 1e-8)) * b[..., 1:]
        return b

    def forward(self, x_cur, x_lag):
        """x_cur:[B,d] current step; x_lag:[B,max_lag,d] lags 1..L."""
        b0 = self._bases(x_cur)
        pred = torch.einsum("bjk,ijk->bi", b0, self.c0 * self.selfmask) + self.bias
        bl = self._bases(x_lag)
        pred = pred + torch.einsum("bhjk,ijhk->bi", bl, self.cl)
        return pred

    def group_lasso(self):
        g0 = torch.linalg.vector_norm(self.c0 * self.selfmask, dim=2)
        gl = torch.linalg.vector_norm(self.cl, dim=(2, 3))
        return g0.sum() + gl.sum()

    def h0(self):
        """Differentiable NOTEARS acyclicity residual on the contemporaneous
        coefficient-magnitude matrix (=0 iff the instantaneous graph is a DAG)."""
        W = torch.linalg.vector_norm(self.c0 * self.selfmask, dim=2)
        return _h_notears(W)

    @torch.no_grad()
    def contemp_importance(self, x_cur):
        c = torch.einsum("bjk,ijk->bij", self._bases(x_cur), self.c0 * self.selfmask)
        return c.std(0)                                    # [d, d]

    @torch.no_grad()
    def lagged_importance(self, x_lag):
        c = torch.einsum("bhjk,ijhk->bij", self._bases(x_lag), self.cl)
        return c.std(0)                                    # [d, d]


class CausalKANInstant(nn.Module):
    """
    Pure instantaneous (i.i.d.) non-linear DAG discovery with KAN edges.

    Self-masked structural equation model x_i = sum_{j!=i} phi_ij(x_j) + e_i, with
    the NOTEARS acyclicity h(W)=tr(e^{W o W})-d applied (via ALM) to the
    edge-importance matrix so the learned graph is a DAG. Learnable B-spline edges
    identify NON-LINEAR additive-noise mechanisms that linear DAG learners miss and
    that fixed-architecture nonlinear learners (NOTEARS-MLP, DAGMA-MLP) capture less
    accurately. importance() returns [effect, cause].
    """
    def __init__(self, d, grid_size=8, spline_order=3, grid_range=(-3.0, 3.0)):
        super().__init__()
        self.d = d; self.k = spline_order; self.n_basis = grid_size + spline_order
        lo, hi = grid_range; step = (hi - lo) / grid_size
        self.register_buffer("grid", torch.arange(-spline_order, grid_size + spline_order + 1) * step + lo)
        self.grid_range = (lo, hi)
        self.coef = nn.Parameter(0.1 * (torch.rand(d, d, self.n_basis) - 0.5))
        self.bias = nn.Parameter(torch.zeros(d))
        self.register_buffer("selfmask", (1 - torch.eye(d)).unsqueeze(-1))

    def _bases(self, x):
        x = x.clamp(*self.grid_range).unsqueeze(-1); g = self.grid
        b = ((x >= g[:-1]) & (x < g[1:])).to(x.dtype)
        for i in range(1, self.k + 1):
            b = ((x - g[:-(i + 1)]) / (g[i:-1] - g[:-(i + 1)] + 1e-8)) * b[..., :-1] \
                + ((g[i + 1:] - x) / (g[i + 1:] - g[1:-i] + 1e-8)) * b[..., 1:]
        return b

    def forward(self, x):
        return torch.einsum("bjk,ijk->bi", self._bases(x), self.coef * self.selfmask) + self.bias

    def group_lasso(self):
        return torch.linalg.vector_norm(self.coef * self.selfmask, dim=2).sum()

    def h(self):
        W = torch.linalg.vector_norm(self.coef * self.selfmask, dim=2)
        return _h_notears(W)

    @torch.no_grad()
    def importance(self, x):
        c = torch.einsum("bjk,ijk->bij", self._bases(x), self.coef * self.selfmask)
        return c.std(0)                                    # [effect, cause]


# ---------------------------------------------------------------------------
# Public method name. SPADE (SPline Additive-noise DAG Estimation) is the name
# used in the manuscript and README for the headline non-linear instantaneous
# DAG learner. The class remains CausalKANInstant for backward-compatible
# imports; SPADE is its canonical public alias.
# ---------------------------------------------------------------------------
SPADE = CausalKANInstant
