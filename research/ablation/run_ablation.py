"""Ablations of BASIL-W and band diagnostics (plan: PLAN.md).  Usage: python run_ablation.py OUTDIR noise seed_lo seed_hi variants(comma|all)  (resume-safe)"""
import os, sys, time, warnings
os.environ["OMP_NUM_THREADS"] = "1"; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); SP = os.path.join(HERE, "../spire")
sys.path.insert(0, SP); sys.path.insert(0, HERE)
import numpy as np, pandas as pd, torch
import bench
from bench import gen_full, zscore, metrics, ece, edge_curve_metrics, add_do, EXTRA, SIM
from basil_s2 import fit_basil_s
import scale_instantaneous as SI
torch.set_num_threads(1)

KGRID = np.round(np.arange(1.0, 3.0001, 0.05), 2)
VARIANTS = {
    "base": {}, "v1_R2cand": dict(cand="R2"), "v2_cold": dict(warm=False), "v3_cold_R2": dict(cand="R2", warm=False),
    "v4a_g_n/4": dict(gm=0.25), "v4b_g_4n": dict(gm=4.0), "v5a_pi_x0.5": dict(pm=0.5), "v5b_pi_x2": dict(pm=2.0),
    "v6_kmax2": dict(kmax=2), "v7_4ch_600d": dict(chains=4, it=600)}


def shared_spade(X, Xz, seed=0):
    Xt = torch.tensor(Xz); t = time.time(); mk = SI.screen_mask(Xz, 15, seed)
    mm = SI.fit(Xt, seed, SI.lg_rule(X.shape[1], 0.2), ep=400, mask=mk); S0 = SI.chunked_importance(mm, Xt)
    return S0, time.time() - t


def order_from(S0):
    d = S0.shape[0]; inw = S0.sum(0).copy(); order0 = []
    for _ in range(d):
        i = int(np.argmin(inw)); order0.append(i); inw[i] = np.inf; inw -= np.where(np.isinf(inw), 0, S0[i, :])
    return order0


def cached_sim(sim):
    cache = {}
    def f(nsim, c=None, val=None, seed_=12345):
        if c is None: return sim(nsim, c, val, seed_)
        k = (nsim, c, float(val))
        if k not in cache: cache[k] = sim(nsim, c, val, seed_).mean(0)[None, :]
        return cache[k]
    return f


def run_variant(name, spec, Xz, A, f, sd, S0, order0, seed, detail):
    n, d = Xz.shape; kw = dict(m=20, iters=spec.get("it", 200) * d, chains=spec.get("chains", 2), burn_frac=0.3, select=True,
                               g=spec.get("gm", 1.0) * n, pi=spec.get("pm", 1.0) * 2.0 / (d - 1), kmax=spec.get("kmax", 3),
                               cand_score=None if spec.get("cand") == "R2" else S0, order0=order0 if spec.get("warm", True) else None)
    t0 = time.time(); o = fit_basil_s(Xz, seed, **kw); tfit = time.time() - t0
    EXTRA.clear(); S = o["P"]
    cr = edge_curve_metrics(lambda c, e, xs: o["mean_curve"](c, e), Xz, A, f, sd); nr = EXTRA["nrmse"]
    au, ap, ska, oa = metrics(S, A); ec = ece(S, A)
    rec = float(np.mean([c in o["cand"][e] for c, e in zip(*np.nonzero(A))]))
    add_do(o["coef"](stable=True), Xz, A, seed)
    erows = []
    for c, e in zip(*np.nonzero(A)):
        gt = f[(c, e)](Xz[:, c].astype(float)) / sd[e]; gt = gt - gt.mean(); m = o["mean_curve"](c, e); D, incl = o["band_draws"](c, e)
        dlo, dhi = np.percentile(D - m, [5, 95], axis=0)
        row = dict(variant=name, seed=seed, c=c, e=e, P=float(S[c, e]), in_cand=bool(c in o["cand"][e]), width1=float(np.mean(dhi - dlo)))
        for k in KGRID:
            cv = (gt >= m + k * dlo - 1e-9) & (gt <= m + k * dhi + 1e-9); row[f"cov_{k}"] = float(cv.mean()); row[f"all_{k}"] = float(cv.all()); row[f"w_{k}"] = float(k * np.mean(dhi - dlo))
        if detail:
            sdp = D.std(0); ok = sdp > 1e-8; z = (gt - m)[ok] / sdp[ok] if ok.sum() > 10 else np.array([np.nan])
            row.update(zmean=float(np.mean(z)), zsd=float(np.std(z)), rmsz=float(np.sqrt(np.mean(z ** 2))), inband_frac_nz=float(ok.mean()))
            if incl.sum() >= 20:
                Dc = D[incl]; lo_c, hi_c = np.percentile(Dc, [5, 95], axis=0); row["cov_cond"] = float(np.mean((gt >= lo_c) & (gt <= hi_c)))
            else: row["cov_cond"] = np.nan
            row["incl_frac"] = float(incl.mean())
            Zc = o["Z"][:, c, :]; co = np.linalg.lstsq(Zc, gt, rcond=None)[0]; res = gt - Zc @ co
            row.update(approx_rel=float(np.linalg.norm(res) / np.linalg.norm(gt)), approx_rms=float(np.sqrt(np.mean(res ** 2))), post_sd=float(sdp[ok].mean()) if ok.any() else np.nan,
                       bias_rms=float(np.sqrt(np.mean((m - gt) ** 2))), gt_rms=float(np.sqrt(np.mean(gt ** 2))))
        erows.append(row)
    cover = float(np.mean([r["cov_1.0"] for r in erows]))
    nrows = []
    if detail:
        Y = Xz
        for e in range(d):
            pa = [c for c in range(d) if A[c, e]]
            if not pa: continue
            r_ = Y[:, e] - sum(f[(c, e)](Xz[:, c].astype(float)) / sd[e] - (f[(c, e)](Xz[:, c].astype(float)) / sd[e]).mean() for c in pa)
            nrows.append(dict(variant=name, seed=seed, e=e, npa=len(pa), s2_true=float(np.var(r_)), s2_model=float(o["node_s2"](e))))
    row = dict(variant=name, seed=seed, nrmse=nr, slope=cr[1], curve_r2=cr[0], do_direct=EXTRA.get("do_direct"), do_indirect=EXTRA.get("do_indirect"), auroc=au, auprc=ap, orient=oa,
               cover=cover, all_cover=float(np.mean([r["all_1.0"] for r in erows])), ece=ec, recall=rec, gap=o["trace_gap"], final_score=o["final_score"], kept=o["kept"], time=tfit,
               width=float(np.mean([r["width1"] for r in erows])), nnz=int((S > 1e-6).sum()))
    return row, erows, nrows


if __name__ == "__main__":
    out, noise, lo, hi, vs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    names = list(VARIANTS) if vs == "all" else vs.split(","); os.makedirs(out, exist_ok=True)
    fm, fe, fn = [os.path.join(out, f"{noise}_{x}.csv") for x in ("main", "edges", "nodes")]
    done = set()
    if os.path.exists(fm): old = pd.read_csv(fm); done = set(zip(old.seed, old.variant))
    for s in range(lo, hi):
        todo = [v for v in names if (s, v) not in done]
        if not todo: continue
        X, A, f, sd, sim = gen_full(1000, 100, s, noise, "m1"); SIM[0] = cached_sim(sim); Xz = zscore(X)
        S0, tss = shared_spade(X, Xz); order0 = order_from(S0)
        for v in todo:
            row, er, nr = run_variant(v, VARIANTS[v], Xz, A, f, sd, S0, order0, s, detail=(v == "base"))
            row.update(noise=noise, t_spade=tss)
            for rr in er + nr: rr["noise"] = noise
            pd.DataFrame([row]).to_csv(fm, mode="a", header=not os.path.exists(fm), index=False)
            pd.DataFrame(er).to_csv(fe, mode="a", header=not os.path.exists(fe), index=False)
            if nr: pd.DataFrame(nr).to_csv(fn, mode="a", header=not os.path.exists(fn), index=False)
            print(s, noise, v, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in row.items() if k in ("nrmse", "auroc", "orient", "cover", "recall", "gap", "time", "do_indirect")}, flush=True)
