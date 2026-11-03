"""SPIRE: group-MCP spline additive DAG learner with residual-independence term.
Implements research/03_formulation.md.  Working name; not the published SPADE.

Whitened coordinates: edge (i<-j) contributes Z_j u_ij with Z_j^T Z_j / n = I (rank-deficient
directions zero-padded), so W_ij = ||u_ij|| is the std of the edge contribution.
Solver: proximal gradient with backtracking on the smooth part (MSE + augmented Lagrangian
+ mu * R_ind with the support frozen inside a step), exact group-MCP (firm-threshold) prox.
"""
import math, numpy as np, torch
from src.cdkan.causal_kan import CausalKANInstant


def spline_features(X, G=8):
    """Whitened centered B-spline features Z[n,d,K], plus unwhitening info for curves."""
    n, d = X.shape
    helper = CausalKANInstant(d, grid_size=G)
    Bs = helper._bases(torch.as_tensor(X, dtype=torch.float64)).to(torch.float64)   # [n,d,K]
    mu = Bs.mean(0, keepdim=True); Bc = Bs - mu
    K = Bc.shape[2]; Z = torch.zeros_like(Bc); T = torch.zeros(d, K, K, dtype=torch.float64)
    for j in range(d):
        Gm = Bc[:, j].T @ Bc[:, j] / n
        ev, U = torch.linalg.eigh(Gm); keep = ev > 1e-6 * ev.max()
        T[j][:, keep] = U[:, keep] / ev[keep].sqrt()          # theta = T u
        Z[:, j] = Bc[:, j] @ T[j]
    return Z.float(), T.float(), mu.float().squeeze(0), helper


def prox_gmcp(U, t, lam, gam):
    """Row-wise (last dim) group-MCP prox with step t; lam [d,1] per-effect-node, gam>t."""
    nrm = torch.linalg.vector_norm(U, dim=-1, keepdim=True).clamp_min(1e-12)
    shrink = torch.clamp(1 - t * lam / nrm, min=0) / (1 - t / gam)
    out = torch.where(nrm <= t * lam, torch.zeros_like(U), torch.where(nrm <= gam * lam, shrink * U, U))
    return out


def _nhsic_matrix(R, Xb, bw_x_cache):
    """R [nb,d] residuals (grad), Xb [nb,d] inputs. returns nHSIC [d_effect,d_cause]."""
    nb, d = R.shape
    def cent(Kmat):
        return Kmat - Kmat.mean(-1, keepdim=True) - Kmat.mean(-2, keepdim=True) + Kmat.mean((-1, -2), keepdim=True)
    Dr = (R.T[:, :, None] - R.T[:, None, :]) ** 2                    # [d,nb,nb]
    with torch.no_grad():
        bw = Dr.flatten(1).median(1).values.clamp_min(1e-6)         # median heuristic (squared dist)
    Kr = cent(torch.exp(-Dr / (bw[:, None, None])))                 # exp(-D/median)
    Kx = bw_x_cache
    num = torch.einsum("iab,jab->ij", Kr, Kx)
    den = torch.sqrt((Kr ** 2).sum((1, 2))[:, None] * (Kx ** 2).sum((1, 2))[None, :] + 1e-12)
    return num / den


def fit_spire(X, seed=0, c_lam=1.0, gam=3.0, mu=1.0, T=500, TA=250, nb=128, G=8, sigma=1.0,
              h_every=50, verbose=False, return_model=False, gate_grad=True):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    X = np.asarray(X, np.float32); n, d = X.shape
    Z, Tm, Bmu, helper = spline_features(X, G); K = Z.shape[2]
    Xt = torch.tensor(X); mask = (1 - torch.eye(d)).unsqueeze(-1)
    lam = torch.full((d, 1, 1), c_lam * sigma * math.sqrt((K + math.log(d)) / n))
    lam_eff = lam.squeeze(-1)                                         # [d,1] broadcast over cause axis
    U = torch.zeros(d, d, K)
    rho, al, h_prev = 1.0, 0.0, float('inf')
    nbb = min(nb, n)

    def h_of(W):
        return torch.trace(torch.linalg.matrix_exp(W * W)) - d

    def smooth(Uv, mu_t, S, idx, need_hsic=True):
        Ue = Uv * mask
        pred = torch.einsum("njk,ijk->ni", Z, Ue)
        mse = 0.5 * ((Xt - pred) ** 2).sum(1).mean()
        W = torch.sqrt((Ue ** 2).sum(-1) + 1e-12) * mask.squeeze(-1)
        h = h_of(W.T)                                                  # W.T[cause,effect]
        val = mse + al * h + 0.5 * rho * h * h
        if mu_t > 0 and S is not None and need_hsic:
            if gate_grad:
                pS = torch.einsum("njk,ijk->ni", Z, Ue * S.unsqueeze(-1))
                R = (Xt - (pred.detach() - pS.detach() + pS))[idx]
            else:
                R = (Xt - pred)[idx]
            Xb = Xt[idx]
            nh = _nhsic_matrix(R, Xb, KX_cache[0])
            val = val + mu_t * (nh * S).sum()
        return val

    KX_cache = [None]
    def set_batch():
        idx = torch.randperm(n, generator=gen)[:nbb]
        Xb = Xt[idx]
        Dx = (Xb.T[:, :, None] - Xb.T[:, None, :]) ** 2
        bw = Dx.flatten(1).median(1).values.clamp_min(1e-6)
        Kx = torch.exp(-Dx / bw[:, None, None])
        Kx = Kx - Kx.mean(-1, keepdim=True) - Kx.mean(-2, keepdim=True) + Kx.mean((-1, -2), keepdim=True)
        KX_cache[0] = Kx
        return idx

    eta = 0.5; S = torch.ones(d, d) - torch.eye(d)
    for t in range(T):
        mu_t = 0.0 if t < TA else mu * min(1.0, (t - TA + 1) / max(1, (T - TA) // 4))
        idx = set_batch()
        Uv = U.clone().requires_grad_(True)
        f0 = smooth(Uv, mu_t, S, idx); f0.backward(); g = Uv.grad * mask
        f0 = f0.item()
        for _ in range(30):
            with torch.no_grad():
                Un = prox_gmcp(U - eta * g, eta, lam_eff.unsqueeze(1) if False else lam, gam) * mask
                dlt = Un - U
                f1 = smooth(Un, mu_t, S, idx).item()
            if f1 <= f0 + (g * dlt).sum().item() + (dlt ** 2).sum().item() / (2 * eta) + 1e-7 * (1 + abs(f0)): break
            eta = max(eta * 0.5, 1e-4)
        U = Un.detach(); eta = min(max(eta, 1e-3) * 1.5, 2.0)
        S = (torch.linalg.vector_norm(U, dim=-1) > 0).float() * (1 - torch.eye(d))
        if (t + 1) % h_every == 0:
            with torch.no_grad():
                W = torch.linalg.vector_norm(U, dim=-1).T; hv = h_of(W).item()
            if verbose: print(t + 1, "h=%.2e nnz=%d eta=%.3f" % (hv, int(S.sum()), eta))
            if hv > 1e-8:
                if hv > 0.25 * h_prev: rho = min(rho * 2, 1e4)
                al += rho * hv
            h_prev = hv
    W = torch.linalg.vector_norm(U, dim=-1).numpy().T                 # [cause,effect]
    # ranking score: support strength W; ties among exactly-zero groups are broken by the KKT margin
    # ||grad_u MSE|| (how close a removed edge is to re-entering). Tie-break only (weight 1e-9).
    Uv = U.clone().requires_grad_(True)
    (0.5 * ((Xt - torch.einsum("njk,ijk->ni", Z, Uv * mask)) ** 2).sum(1).mean()).backward()
    gn = torch.linalg.vector_norm(Uv.grad * mask, dim=-1).numpy().T
    W_rank = W + 1e-9 * gn
    if not return_model: return W_rank
    return W_rank, dict(U=U, T=Tm, Bmu=Bmu, helper=helper, X=X)


def spire_curve(model, c, e, xs):
    """learned phi_ce(xs) (centered over data) in standardized-target units."""
    helper = model["helper"]; Bx = helper._bases(torch.as_tensor(np.asarray(xs, np.float32))[:, None].expand(-1, model["X"].shape[1]).contiguous())
    Bc = Bx[:, c, :] - model["Bmu"][c]
    th = model["T"][c] @ model["U"][e, c]
    return (Bc @ th).numpy()
