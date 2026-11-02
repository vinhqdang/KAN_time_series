"""
G5 PILOT PROTOCOL (written BEFORE running; see results in g5_results.*)

Question: when series are integrated (I(1)), does a spline additive-noise lagged
learner (SPADE-lagged = src/cdkan/causal_kan.py::CausalKAN) recover the true lagged
graph better on levels, on first differences, or with an error-correction-style
recipe (ECM-SPADE)? Is any difference larger than seed noise?

Generator (known graph, d=6, T=500 after burn-in 100, lagged graph at lag 1):
  * Each seed draws a random lag-1 graph A (each off-diagonal pair w.p. 0.25,
    >=3 edges) and for each edge a non-linear mechanism g in {tanh(2u), sin(1.5u),
    2u*exp(-u^2/2)} times a random-sign coefficient in [0.6,1.0].
  * Scenario I1 (integrated, no cointegration): increments
        dx_i,t = sum_j c_ij g_ij(dx_j,t-1) + eps_it,  eps~N(0,0.5^2);  x_t = x_{t-1}+dx_t.
    Levels are random-walk driven; truth = A (edges act on increments).
  * Scenario I1C (integrated + 2 true cointegrating pairs k->i, disjoint targets):
        dx_i,t += -0.15 (x_i,t-1 - x_k,t-1)   (linear EC term, the standard VECM form)
    truth = A union {k->i}. The k->i edge acts through the LEVEL of x_k, so a pure
    difference learner has no access to it by construction.
  * Scenario ST (stationary control): x_it = 0.3 x_i,t-1 + sum_j c_ij g_ij(x_j,t-1) + eps.
    truth = A. Differencing here is over-differencing.
Seeds: validation seeds 100-102 (choose lambda_gl per method-variant from
{0.003,0.01,0.03} by mean AUROC over the three scenarios), test seeds 200-209
(10 seeds per scenario), disjoint; nothing is tuned on test seeds. Epochs/lr fixed
(250, 5e-3, full batch), max_lag=2 for all learners (true lag is 1).
Methods: SPADE-levels (z-scored levels, target x_t); SPADE-diff (z-scored dx, lags of dx);
ECM-SPADE (recipe, below); VAR-Lasso and PCMCI(ParCorr, tau_max=2, lag>=1 max |val|)
on levels and on differences. VAR-LiNGAM: lingam failed to pip-install here -> skipped.
Metric: edge AUROC and AUPRC over the d(d-1) off-diagonal ordered pairs vs the lagged truth
(same score_adj as scripts/honest_causal_benchmark.py). Report mean +- sd and the paired
(by seed) difference with its sd across seeds.

Recipe (ECM-SPADE): target dx_it (z-scored); inputs = lagged dx (lags 1..2) per cause
via one CausalKAN, PLUS lag-1 LEVEL of every series via a second CausalKAN (levels
z-scored); predictions summed, one joint group-lasso. Edge score_ij = std of the
diff-part contribution (summed over lags) + max(0, level-part std - mean level-part
std over K=3 placebo refits in which each level column is circularly shifted by an
independent random offset in [T/4, 3T/4]) . The placebo is the falsification check:
a level edge must beat a trend-matched null that destroys alignment with the target
but preserves each series' own trend structure. Guarantee vs heuristic: differenced
lagged edges are the stationary-regression part (standard asymptotics for I(1) series
with stationary increments: no spurious-regression problem; not proved for the spline
learner here). Cointegration edges are recoverable only through the level term; the
placebo-excess score is a heuristic calibration (no coverage/FDR guarantee; circular
shifts of a random walk create one jump, a known imperfection).
"""
import os, sys, json, time, warnings
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np, torch
torch.set_num_threads(1)
warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src.cdkan.causal_kan import CausalKAN
from honest_causal_benchmark import score_adj, bl_var_lasso, bl_pcmci
torch.set_num_threads(1)  # honest_causal_benchmark resets threads on import

D_, T_, BURN, LAGS = 6, 500, 100, 2
GS = [lambda u: np.tanh(2 * u), lambda u: np.sin(1.5 * u), lambda u: 2 * u * np.exp(-u ** 2 / 2)]

def zs(X): return (X - X.mean(0)) / (X.std(0) + 1e-8)

def gen(scn, seed):
    r = np.random.default_rng(seed); d = D_
    while True:
        A = (r.random((d, d)) < 0.25).astype(int); np.fill_diagonal(A, 0)  # A[i,j]: j->i
        if A.sum() >= 3: break
    mech = r.integers(0, 3, (d, d)); coef = r.choice([-1, 1], (d, d)) * r.uniform(0.6, 1.0, (d, d))
    truth = A.copy(); ec = []
    if scn == "I1C":
        pairs = r.permutation(d)[:4]
        ec = [(int(pairs[0]), int(pairs[1])), (int(pairs[2]), int(pairs[3]))]  # (k cause, i effect)
        for k, i in ec: truth[i, k] = 1
    n = T_ + BURN; x = np.zeros((n, d)); dx = np.zeros((n, d))
    for t in range(1, n):
        prev = dx[t - 1] if scn != "ST" else x[t - 1]
        new = r.normal(0, 0.5, d)
        for i in range(d):
            for j in range(d):
                if A[i, j]: new[i] += coef[i, j] * GS[mech[i, j]](prev[j])
        if scn == "ST":
            x[t] = 0.3 * x[t - 1] + new
        else:
            for k, i in ec: new[i] += -0.15 * (x[t - 1, i] - x[t - 1, k])
            dx[t] = new; x[t] = x[t - 1] + new
    return x[BURN:], truth

def _windows(H, W):  # hist of W rows ending at t-1, aligned with target index t>=W
    return np.stack([H[t - W:t] for t in range(W, len(H))]).astype(np.float32)

def fit_spade(blocks, target, lg, seed, epochs=250, lr=5e-3):
    """blocks: list of (hist[B,W,d], max_lag). Returns per-block importance."""
    torch.manual_seed(seed)
    d = target.shape[1]
    ms = [CausalKAN(d, max_lag=ml) for _, ml in blocks]
    ps = [p for m in ms for p in m.parameters()]
    hs = [torch.tensor(h) for h, _ in blocks]; y = torch.tensor(target, dtype=torch.float32)
    opt = torch.optim.Adam(ps, lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        pred = sum(m(h) for m, h in zip(ms, hs))
        loss = ((pred - y) ** 2).mean() + lg * sum(m.group_lasso() for m in ms)
        if torch.isfinite(loss):
            loss.backward(); torch.nn.utils.clip_grad_norm_(ps, 5.0); opt.step()
    return [m.importance(h).numpy() for m, h in zip(ms, hs)]

def run_methods(scn, seed, lgs, baselines=True):
    x, truth = gen(scn, seed); out = {}
    dx = zs(np.diff(x, axis=0)); xl = zs(x)
    h = _windows(xl, LAGS); y = xl[LAGS:]
    out["SPADE-levels"] = fit_spade([(h, LAGS)], y, lgs["levels"], seed)[0]
    h = _windows(dx, LAGS); y = dx[LAGS:]
    out["SPADE-diff"] = fit_spade([(h, LAGS)], y, lgs["diff"], seed)[0]
    # recipe: target dx[t]; level input = level before the step (xl[t]); diff lags dx[t-1], dx[t-2]
    hd = _windows(dx, LAGS)
    lev = xl[:-1]
    yy = dx[LAGS:]
    hl = lev[LAGS:][:, None, :].astype(np.float32)
    ri = fit_spade([(hd, LAGS), (hl, 1)], yy, lgs["ecm"], seed)
    rng = np.random.default_rng(seed + 7); plc = []
    for k in range(3):
        sh = np.stack([np.roll(lev[:, j], int(rng.integers(len(lev) // 4, 3 * len(lev) // 4))) for j in range(D_)], 1)
        plc.append(fit_spade([(hd, LAGS), (sh[LAGS:][:, None, :].astype(np.float32), 1)], yy, lgs["ecm"], seed + 1 + k)[1])
    out["ECM-SPADE"] = ri[0] + np.maximum(0, ri[1] - np.mean(plc, 0))
    out["ECM-SPADE(no-placebo)"] = ri[0] + ri[1]
    if baselines:
        for name, X in (("levels", xl), ("diff", dx)):
            out["VARLasso-" + name] = bl_var_lasso(X, LAGS)
            try: out["PCMCI-" + name] = bl_pcmci(X, LAGS)
            except Exception as e: out["PCMCI-" + name] = np.zeros((D_, D_))
    return {k: score_adj(v, truth) for k, v in out.items()}

if __name__ == "__main__":
    SCN = ["I1", "I1C", "ST"]; grid = [0.003, 0.01, 0.03]
    t0 = time.time(); val = {}
    for lg in grid:
        a = {"levels": [], "diff": [], "ecm": []}
        for s in (100, 101, 102):
            for scn in SCN:
                r = run_methods(scn, s, {"levels": lg, "diff": lg, "ecm": lg}, baselines=False)
                a["levels"].append(r["SPADE-levels"]["auroc"]); a["diff"].append(r["SPADE-diff"]["auroc"]); a["ecm"].append(r["ECM-SPADE"]["auroc"])
        val[lg] = {k: float(np.mean(v)) for k, v in a.items()}
        print("val", lg, val[lg], f"{time.time()-t0:.0f}s", flush=True)
    lgs = {k: max(grid, key=lambda g: val[g][k]) for k in ("levels", "diff", "ecm")}
    print("selected", lgs, flush=True)
    res = {}
    for scn in SCN:
        for s in range(200, 210):
            res[f"{scn}|{s}"] = run_methods(scn, s, lgs)
        print("done", scn, f"{time.time()-t0:.0f}s", flush=True)
    json.dump({"val": {str(k): v for k, v in val.items()}, "lgs": lgs, "test": res},
              open(os.path.join(os.path.dirname(__file__), "g5_results.json"), "w"))
