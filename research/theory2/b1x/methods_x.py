"""B'1 methods: continuous-optimisation DAG learners and a labels-using time-series reference.

- DAGMA linear: the authors' PyPI package `dagma` 1.1.1, numpy class `dagma.linear.DagmaLinear(loss_type="l2")`,
  `fit` with the package defaults (lambda1 = 0.03, w_threshold = 0.3, T = 5, s = [1, .9, .8, .7, .6], warm_iter 3e4,
  max_iter 6e4, lr 3e-4). Bello, Aragam, Ravikumar, NeurIPS 2022.
- NOTEARS linear: line-by-line copy of the authors' reference implementation `notears/linear.py`
  (github.com/xunzheng/notears, Apache-2.0; Zheng, Aragam, Ravikumar, Xing, NeurIPS 2018), with the README example
  settings lambda1 = 0.1, loss_type = "l2", max_iter 100, h_tol 1e-8, rho_max 1e16, w_threshold = 0.3.
- GOLEM-EV / GOLEM-NV: numpy port of the authors' TensorFlow code (github.com/ignavierng/golem; Ng, Ghassami, Zhang,
  NeurIPS 2020). Same objective, same optimiser (Adam, lr 1e-3, 1e5 iterations, B initialised at 0; NV initialised at
  the EV solution), same defaults (EV: lambda1 = 2e-2, lambda2 = 5; NV: lambda1 = 2e-3, lambda2 = 5), same
  post-processing (threshold 0.3, then remove the smallest remaining edges until acyclic). The only change is that
  the data enter through the sample covariance (the likelihood depends on X only through it), which makes an
  iteration O(d^3) instead of O(n d^2); the gradients are analytic instead of TF autodiff.
- PCMCI+ (Runge, UAI 2020) via tigramite 5.2 with ParCorr, tau_max = 1, pc_alpha = 0.01, on the time-indexed data
  (the n trajectories as n datasets, analysis_mode = "multiple"). Reference only: it uses the time labels.
"""
import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt


# ----------------------------------------------------------------------------- NOTEARS (reference code)
def notears_linear(X, lambda1=0.1, loss_type="l2", max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """Copy of notears/linear.py::notears_linear (l2 loss only path used)."""
    def _loss(W):
        M = X @ W
        if loss_type == "l2":
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        else:
            raise ValueError("unknown loss type")
        return loss, G_loss

    def _h(W):
        E = slin.expm(W * W)
        h = np.trace(E) - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == "l2":
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method="L-BFGS-B", jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


# ----------------------------------------------------------------------------- DAGMA (package)
def dagma_linear(X):
    from dagma.linear import DagmaLinear
    import contextlib, io
    m = DagmaLinear(loss_type="l2", verbose=False)
    with contextlib.redirect_stderr(io.StringIO()):     # tqdm progress bar
        W = m.fit(X)                                     # package defaults, w_threshold = 0.3
    return W


# ----------------------------------------------------------------------------- GOLEM (numpy port)
def _golem_adam(S, d, lambda1, lambda2, equal_variances, B_init, num_iter, lr=1e-3, b1=0.9, b2=0.999, eps=1e-7):
    """Minimise the GOLEM score with Adam (TF defaults beta1 0.9, beta2 0.999, eps 1e-7 in TF; 1e-8 here is
    immaterial). S = X^T X / n of centred X. Score (n-free form, as in the paper):
      EV: d/2 log tr((I-B)^T S (I-B)) - log|det(I-B)| + l1 |B|_1 + l2 h(B)
      NV: 1/2 sum_j log [(I-B)^T S (I-B)]_jj - log|det(I-B)| + l1 |B|_1 + l2 h(B)
    (GOLEM's TF code uses ||X - XB||^2 without dividing by n; this changes the score by a constant only.)"""
    off = 1.0 - np.eye(d)
    B = B_init.copy() * off
    m = np.zeros((d, d)); v = np.zeros((d, d)); I = np.eye(d)
    for it in range(1, int(num_iter) + 1):
        IB = I - B
        SIB = S @ IB
        if equal_variances:
            r = np.sum(IB * SIB)                          # tr((I-B)^T S (I-B))
            g_lik = -d * SIB / r                          # d/dB of d/2 log r
        else:
            r = np.sum(IB * SIB, axis=0)                  # diag((I-B)^T S (I-B))
            g_lik = -SIB / r[None, :]
        g_det = np.linalg.inv(IB).T                       # d/dB of -log|det(I-B)| = (I-B)^{-T}
        E = slin.expm(B * B)
        g_h = 2.0 * B * E.T
        g = (g_lik + g_det + lambda1 * np.sign(B) + lambda2 * g_h) * off
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** it); vh = v / (1 - b2 ** it)
        B = B - lr * mh / (np.sqrt(vh) + eps)
    return B


def _is_dag(W):
    d = W.shape[0]
    A = (W != 0).astype(float)
    return np.trace(slin.expm(A)) - d < 1e-8 if d <= 200 else None


def golem_postprocess(B, graph_thres=0.3):
    """GOLEM's postprocess: threshold, then remove edges with the smallest |weight| until acyclic."""
    B = B.copy()
    B[np.abs(B) <= graph_thres] = 0
    nz = np.argwhere(B != 0)
    order = np.argsort(np.abs(B[nz[:, 0], nz[:, 1]]))
    k = 0
    while not _is_dag(B):
        i, j = nz[order[k]]
        B[i, j] = 0
        k += 1
    return B


def golem(X, num_iter=100000):
    """Returns (B_EV, B_NV) after post-processing."""
    Xc = X - X.mean(0, keepdims=True)
    n, d = Xc.shape
    S = Xc.T @ Xc / n
    B_ev = _golem_adam(S, d, 2e-2, 5.0, True, np.zeros((d, d)), num_iter)
    B_nv = _golem_adam(S, d, 2e-3, 5.0, False, B_ev, num_iter)
    return golem_postprocess(B_ev), golem_postprocess(B_nv)


# ----------------------------------------------------------------------------- PCMCI+ (labels used; reference)
def pcmciplus_window(X, p, T, pc_alpha=0.01):
    """X: (n, T*p) in time order (column t*p+v). Runs PCMCI+ on the n trajectories as multiple datasets and maps the
    estimated summary graph to the window: a lag-1 link i -> j gives (t,i) -> (t+1,j) for every t; a contemporaneous
    link i - j (o-o or x-x or -->) gives the corresponding within-slice edge in every slice.
    Returns Mdir, Mund (bool, window order)."""
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    import contextlib, io
    n = X.shape[0]
    data = {k: X[k].reshape(T, p) for k in range(n)}
    df = pp.DataFrame(data, analysis_mode="multiple")
    pcm = PCMCI(dataframe=df, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
    with contextlib.redirect_stdout(io.StringIO()):
        res = pcm.run_pcmciplus(tau_min=0, tau_max=1, pc_alpha=pc_alpha)
    G = res["graph"]                                      # (p, p, 2) strings
    d = T * p
    Mdir = np.zeros((d, d), bool); Mund = np.zeros((d, d), bool)
    for i in range(p):
        for j in range(p):
            lag1 = G[i, j, 1]
            if lag1 == "-->":
                for t in range(T - 1):
                    Mdir[t * p + i, (t + 1) * p + j] = True
            if i != j:
                c = G[i, j, 0]
                for t in range(T):
                    a, b = t * p + i, t * p + j
                    if c == "-->":
                        Mdir[a, b] = True
                    elif c in ("o-o", "x-x", "<->"):
                        Mund[a, b] = Mund[b, a] = True
    return Mdir, Mund
