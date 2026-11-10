"""Evaluation helpers shared by run_q2.py and run_q5.py (added in step 2, before any scored run; PLAN.md Addendum 1).
Primary metrics are first-slice-excluded (PLAN.md section 2): SHD_x1n = SHD excluding pairs inside slice 1, divided by the number
of true edges; aoa_x1 = ancestral order accuracy over the true edges whose tail is not in slice 1.
"""
import numpy as np
import common_y as cy
from common_y import mask_inside_first_slice, slice_of_cols
from common import shd, sid, orient_score


def closure(M):
    """Transitive closure of a directed bool matrix (R[i, j]: path i -> ... -> j of length >= 1)."""
    R = M.copy().astype(bool)
    while True:
        Ri = R.astype(np.int32)
        R2 = R | ((Ri @ Ri) > 0)
        if (R2 == R).all():
            return R
        R = R2


def anc_scores(G, D):
    """Per true edge (i, j): 1 if i is an ancestor of j in D and not vice versa, 0 if the reverse, 1/2 otherwise."""
    R = closure(D)
    E = np.argwhere(G)
    f, b = R[E[:, 0], E[:, 1]], R[E[:, 1], E[:, 0]]
    return E, np.where(f & ~b, 1.0, np.where(b & ~f, 0.0, 0.5)), f, b


def eval_graph(Gp, Mdir, Mund, sl):
    """Metrics of an estimated graph (permuted frame). sl: slice index of every permuted column."""
    n_true = max(int(Gp.sum()), 1)
    excl = mask_inside_first_slice(sl)
    D = Mdir & ~Mund
    E, v, f, b = anc_scores(Gp, D)
    keep = sl[E[:, 0]] != 0
    out = dict(shd_n=shd(Gp, Mdir, Mund) / n_true, shd_x1n=shd(Gp, Mdir, Mund, exclude=excl) / n_true,
               aoa=float(v.mean()) if len(v) else float("nan"), aoa_x1=float(v[keep].mean()) if keep.any() else float("nan"),
               fwd=float((f & ~b).mean()), bwd=float((b & ~f).mean()), nedge=int(Mdir.sum() + np.triu(Mund, 1).sum()),
               dag=int(not closure(D).diagonal().any()))
    sn = float("nan")
    if out["dag"] == 1 and not np.diag(D).any():       # SID is defined for DAG estimates only (self-loops and cycles: NaN)
        try:
            sn, sc = sid(Gp, D)
        except BaseException as e:                     # noqa: BLE001  gadjid raises pyo3 PanicException (a BaseException) on malformed graphs
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            sn = float("nan")
    out["sidn"] = sn
    return out


def eval_order(Gp, order, sl):
    return dict(oa_full=cy.order_accuracy(Gp, order), oa_x1=cy.oa_x1(Gp, order, sl))


def with_timeout(fn, seconds):
    """Run fn() in a forked child with a wall-clock limit. Returns (value, status): status 'ok', 'timeout' or 'err:...'.
    A learner that does not finish is recorded as missing (never silently dropped); the shards count them."""
    import multiprocessing as mp, queue
    ctx = mp.get_context("fork"); q = ctx.Queue()

    def target():
        try:
            q.put(("ok", fn()))
        except Exception as e:                         # noqa: BLE001
            q.put(("err", repr(e)[:200]))
    pr = ctx.Process(target=target); pr.start()
    try:
        kind, val = q.get(timeout=seconds)
    except queue.Empty:
        pr.kill(); pr.join(); return None, "timeout"
    pr.join()
    return (val, "ok") if kind == "ok" else (None, "err:" + val)


NAN_METRICS = dict(shd_n=float("nan"), shd_x1n=float("nan"), aoa=float("nan"), aoa_x1=float("nan"), fwd=float("nan"),
                   bwd=float("nan"), nedge=float("nan"), dag=float("nan"), sidn=float("nan"))
