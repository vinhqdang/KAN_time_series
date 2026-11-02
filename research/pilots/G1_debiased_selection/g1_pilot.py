"""
G1 / RQ-A PILOT: debiased + stable edge selection for spline additive-noise DAGs (SPADE).

PROBLEM. SPADE fits x_i = b_i + sum_j phi_ij(x_j) with a group-lasso lam*sum_ij ||theta_ij||_2 on the
B-spline coefficients of each edge. The group-lasso shrinks every active group toward zero by a
constant amount (KKT: theta_ij = (1 - lam/||grad_ij||)_+ * unpenalised-like solution), so (i) the
amplitude of the learned curves phi_ij is biased low (flattened curves; the importance ranking is
only mildly affected), and (ii) with correlated parents (concurvity) weak spurious groups can
survive at the same lam that kills true weak ones (irrepresentable-condition failure).

ALGORITHM (DSS = Debiased Stable Selection; a combination of known ingredients, no novelty claimed).
 1. Stage 1: base SPADE fit (lam=0.01) -> group norms n1_ij.
 2. Adaptive group lasso (Zou 2006, group version): refit with weights w_ij=(n1_ij+eps)^-gamma.
 3. Stability: B half-sample base fits -> selection frequency pi_ij = fraction of subsamples in which
    edge ij is among the top-2d importance (Meinshausen-Buhlmann style, top-k variant).
 4. Support S = top-2d edges of  I_adapt_ij * pi_ij  (adaptive importance weighted by frequency).
 5. Debiased refit: refit restricted to S with NO group penalty but a P-spline second-difference
    smoothness penalty lam_s*sum ||D2 theta||^2 (+ the acyclicity ALM). Score = refit importance
    (zero off S).
THEORY SKETCH. (a) Amplitude: for a known support the group-lasso solution is the unpenalised fit
shrunk by a factor (1 - lam/||.||)_+ per group, so bias on edge ij is O(lam) regardless of signal
strength; adaptive weights give penalty lam*(n_ij+eps)^-gamma -> O(lam*n^-gamma) on strong edges
(oracle-type behaviour of the adaptive lasso, fixed d, n->inf), and refit on S removes the bias
altogether (zero penalty), leaving only variance, which the P-spline penalty trades for a small
smoothing bias. (b) Selection: stability selection bounds expected false selections
E[V] <= q^2/((2*pi_thr-1)*p) under exchangeability conditions; here it is used heuristically
(acyclicity-constrained fits, top-k variant, so the bound is NOT claimed to hold). Weighting I*pi
down-weights edges chosen only in some subsamples (concurvity-driven flips).
Expected failure mode: the unpenalised refit inflates variance and keeps spurious edges in S.

PILOT PROTOCOL (fixed before running).
 Data: iSCM-style generator gen_iscm: random DAG with expected degree ~2 (density 2/(d-1)), per-edge
   mechanism from {tanh, sin, quadratic, softsign} (same family as src.synthetic), uniform noise;
   EVERY variable is standardised to mean 0 / var 1 immediately after it is generated, before it is
   used as a parent. This removes varsortability; we report measured varsortability of this
   generator with and without standardisation. n=1500. True curve in standardised units:
   g_ce(x_c) = f_ce(x_c)/sd(v_e).
 Seeds: hyperparameter validation seeds 200-205 (d=10) [lam_s in {0,1e-3,1e-2}, chosen by mean curve
   R2 of dss, AUPRC tie-break]; test seeds 1000-1019 (d=10) and 2000-2019 (d=20), disjoint from
   validation and from seeds used in earlier SPADE work (0-4,100-119,200-219,300-319).
 Methods: base, adaptive, refit (step 5 on support from step 2 only, no stability), stab
   (selection-frequency score, ties broken by base importance), dss (full pipeline).
 Metrics per graph: edge AUROC, AUPRC (off-diagonal), curve-recovery R2 over TRUE edges
   (1 - SSE/SS_true of centred learned vs centred true curve at the data points; non-selected edges
   have zero curve, no thresholding, so amplitude bias counts) and calibration slope (median over
   edges of OLS slope of learned on true curve; <1 = shrinkage).
 Tests: paired two-sided Wilcoxon signed-rank vs base over 20 graphs per d; 4 methods x 3 metrics
   per d, Bonferroni threshold 0.05/12 reported separately.
 Compute: OMP_NUM_THREADS=1, torch threads 1, python3.13.
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from spade_variants import zscore, score
from src.cdkan.causal_kan import CausalKANInstant


def gen_iscm(n, d, seed, standardize=True):
    r = np.random.RandomState(seed); dens = 2.0 / (d - 1); perm = r.permutation(d)
    A = np.zeros((d, d), int)
    for a in range(d):
        for b in range(a):
            if r.rand() < dens: A[perm[b], perm[a]] = 1
    fns = [lambda x: 0.8 * np.tanh(1.5 * x), lambda x: 0.7 * np.sin(1.5 * x),
           lambda x: 0.3 * (x ** 2 - 1.0), lambda x: 0.8 * x / (1.0 + np.abs(x))]
    f = {(c, e): fns[r.randint(len(fns))] for c in range(d) for e in range(d) if A[c, e]}
    X = np.zeros((n, d)); sd = np.ones(d)
    for e in perm:
        v = r.uniform(-0.7, 0.7, n)
        for c in range(d):
            if A[c, e]: v = v + f[(c, e)](X[:, c])
        if standardize: sd[e] = v.std(); X[:, e] = (v - v.mean()) / sd[e]
        else: X[:, e] = v
    return X.astype(np.float32), A, f, sd


def varsortability(X, A):
    v = X.var(0); ok = tot = 0
    for c, e in zip(*np.nonzero(A)): tot += 1; ok += (v[c] < v[e]) + 0.5 * (v[c] == v[e])
    return ok / max(tot, 1)


def fit(Xt, seed, lg=0.01, ep=400, weights=None, mask=None, smooth=0.0):
    torch.manual_seed(seed); np.random.seed(seed)
    d = Xt.shape[1]; m = CausalKANInstant(d, grid_size=8)
    if mask is not None: m.selfmask.mul_(mask.unsqueeze(-1))
    opt = torch.optim.Adam(m.parameters(), lr=5e-3); rho, al = 1.0, 0.0
    for e in range(ep):
        opt.zero_grad(); mse = ((m(Xt) - Xt) ** 2).mean(); h = m.h()
        c = m.coef * m.selfmask; norms = torch.linalg.vector_norm(c, dim=2)
        pen = lg * ((norms * weights).sum() if weights is not None else norms.sum())
        if smooth > 0:
            d2 = c[..., 2:] - 2 * c[..., 1:-1] + c[..., :-2]; pen = pen + smooth * (d2 ** 2).sum()
        (mse + pen + al * h + 0.5 * rho * h * h).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if (e + 1) % 50 == 0:
            with torch.no_grad(): hv = m.h().item()
            if hv > 1e-8: al += rho * hv; rho = min(rho * 2, 1e10)
    return m


def topk_mask(imp, k):                      # imp [effect, cause]
    thr = torch.sort(imp.flatten(), descending=True)[0][min(k, imp.numel()) - 1]
    mk = (imp >= thr).float(); mk.fill_diagonal_(0); return mk


def pipeline(X, seed, smooths=(0.0,), B=4, gamma=1.0, eps=1e-2):
    """returns dict name -> (S[cause,effect], model-for-curves or None), Xt"""
    Xt = torch.tensor(zscore(X)); d = Xt.shape[1]; k = 2 * d; out = {}
    m1 = fit(Xt, seed); I1 = m1.importance(Xt); out["base"] = (I1.numpy().T, m1)
    with torch.no_grad():
        n1 = torch.linalg.vector_norm(m1.coef * m1.selfmask, dim=2); w = (n1 + eps) ** (-gamma); w = w / w.mean()
    m2 = fit(Xt, seed, weights=w); I2 = m2.importance(Xt); out["adaptive"] = (I2.numpy().T, m2)
    rng = np.random.RandomState(seed); freq = torch.zeros(d, d)
    for b in range(B):
        idx = torch.tensor(rng.choice(len(Xt), len(Xt) // 2, replace=False))
        mb = fit(Xt[idx], seed * 100 + b); freq += topk_mask(mb.importance(Xt[idx]), k)
    freq /= B
    out["stab"] = ((freq + 1e-3 * I1 / (I1.max() + 1e-12)).numpy().T, None)
    for sm in smooths:
        mk = topk_mask(I2, k); mr = fit(Xt, seed, lg=0.0, mask=mk, smooth=sm)
        out[f"refit_s{sm}"] = (mr.importance(Xt).numpy().T, mr)
        if sm != smooths[0]: continue                      # dss only at the frozen (first) smooth value
        comb = I2 / (I2.max() + 1e-12) * freq; mk2 = topk_mask(comb, k)
        md = fit(Xt, seed, lg=0.0, mask=mk2, smooth=sm)
        out[f"dss_s{sm}"] = (md.importance(Xt).numpy().T, md)
    return out, Xt


@torch.no_grad()
def curve_metrics(m, Xt, A, f, sd):
    """R2 and calibration slope of learned vs true edge curves over TRUE edges."""
    if m is None: return float("nan"), float("nan")
    bases = m._bases(Xt); r2 = []; sl = []
    for c, e in zip(*np.nonzero(A)):
        g = (m.coef * m.selfmask)[e, c]; ghat = (bases[:, c, :] @ g).numpy(); ghat = ghat - ghat.mean()
        gt = f[(c, e)](Xt[:, c].numpy().astype(float)) / sd[e]; gt = gt - gt.mean()
        r2.append(1 - ((ghat - gt) ** 2).sum() / (gt ** 2).sum()); sl.append((ghat * gt).sum() / (gt ** 2).sum())
    return float(np.mean(r2)), float(np.median(sl))


def evaluate(X, A, f, sd, seed, smooths):
    res, Xt = pipeline(X, seed, smooths); rows = {}
    for name, (S, m) in res.items():
        au, ap, f1 = score(S, A); r2, sl = curve_metrics(m, Xt, A, f, sd)
        rows[name] = dict(auroc=au, auprc=ap, f1=f1, curve_r2=r2, slope=sl)
    return rows


if __name__ == "__main__":
    import pandas as pd
    mode, d, s0, s1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    sm = (0.0, 1e-3, 1e-2) if mode == "val" else tuple(float(v) for v in sys.argv[5].split(","))
    outf = os.path.join(HERE, f"raw_{mode}_d{d}.csv"); rows = []
    if os.path.exists(outf): rows = pd.read_csv(outf).to_dict("records")
    done = {r["seed"] for r in rows}
    for s in range(s0, s1):
        if s in done: continue
        X, A, f, sd = gen_iscm(1500, d, s); t0 = time.time()
        for name, r in evaluate(X, A, f, sd, s, sm).items(): rows.append(dict(d=d, seed=s, method=name, **r))
        pd.DataFrame(rows).to_csv(outf, index=False)
        print(mode, d, s, round(time.time() - t0), "s", flush=True)
