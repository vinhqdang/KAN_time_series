"""Step-0 additions to the b1y learners (PLAN.md Addendum 1): pooled VAR-LiNGAM, DYNOTEARS client, GOLEM wrapper, Tetrad-free
conversions. Imports nothing heavy at module level. Returns (Mdir, Mund) bool (d, d) in the frame of the input columns.

Time-labelled references use the time-ordered window X (column t*p+v) and map the estimated summary graph to the window:
lag-1 edge i -> j gives (t, i) -> (t+1, j) in every transition; a contemporaneous edge i -> j gives (t, i) -> (t, j) in every slice.
"""
import os, sys, pickle, struct, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "b1x")); sys.path.insert(0, HERE)

PY310 = os.environ.get("B1Y_PY310", "/home/user/venv310/bin/python")


def pooled_pairs(X, p, T):
    """All (x_t, x_{t+1}) pairs over trajectories: returns prev, nxt of shape (n*(T-1), p), plus Z standardisation."""
    cols = lambda t: X[:, t * p:(t + 1) * p]
    prev = np.vstack([cols(t) for t in range(T - 1)]); nxt = np.vstack([cols(t + 1) for t in range(T - 1)])
    return prev, nxt


def window_from_summary(W, A, p, T, thr_w=0.0, thr_a=0.0):
    """W[i, j]: i -> j within every slice; A[i, j]: i(t) -> j(t+1) for every transition."""
    d = T * p; Mdir = np.zeros((d, d), bool)
    Wb = np.abs(W) > thr_w; Ab = np.abs(A) > thr_a
    np.fill_diagonal(Wb, False)
    for t in range(T):
        Mdir[t * p:(t + 1) * p, t * p:(t + 1) * p] = Wb
    for t in range(T - 1):
        Mdir[t * p:(t + 1) * p, (t + 1) * p:(t + 2) * p] = Ab
    return Mdir, np.zeros((d, d), bool)


def varlingam_pooled_summary(X, p, T, thr_a=0.1, random_state=0):
    """Pooled VAR-LiNGAM for replicated short trajectories (Hyvarinen, Zhang, Shimizu, Hoyer 2010, applied to pooled transitions):
      1. standardise columns of the pooled (x_t, x_{t+1}) pairs; M = OLS of x_{t+1} on x_t (column convention x_{t+1} = M x_t + u);
      2. DirectLiNGAM (lingam 1.13.0, package defaults incl. adaptive-lasso pruning) on the pooled residuals u -> B0
         (lingam convention u_i = sum_j B0[i, j] u_j + n_i, so B0[i, j] != 0 means j -> i);
      3. B1 = (I - B0) M (lag-1 effects of the structural model x_t = B0 x_t + B1 x_{t-1} + n_t); lag-1 edge j -> i if |B1[i, j]| > thr_a.
    This is NOT the packaged lingam.VARLiNGAM, which fits one long series. Returns (W, A) in the W[i, j]: i -> j convention."""
    import lingam
    prev, nxt = pooled_pairs(X, p, T)
    mu = np.vstack([prev, nxt]).mean(0); sdv = np.vstack([prev, nxt]).std(0)
    Z0 = (prev - mu) / sdv; Z1 = (nxt - mu) / sdv
    M = np.linalg.lstsq(Z0, Z1, rcond=None)[0].T            # (p, p): x_{t+1}[i] = sum_j M[i, j] x_t[j]
    U = Z1 - Z0 @ M.T
    dl = lingam.DirectLiNGAM(random_state=random_state); dl.fit(U)
    B0 = dl.adjacency_matrix_
    B1 = (np.eye(p) - B0) @ M
    return B0.T, B1.T                                       # [i, j]: i -> j


def varlingam_pooled(X, p, T, thr_a=0.1):
    W, A = varlingam_pooled_summary(X, p, T, thr_a)
    return window_from_summary(W, A, p, T, thr_w=0.0, thr_a=thr_a)


class DynotearsServer:
    """Persistent Python 3.10 subprocess running dynotears_worker.py (causalnex 0.12.1)."""
    def __init__(self):
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONPATH="")
        self.proc = subprocess.Popen([PY310, "-I", os.path.join(HERE, "dynotears_worker.py")], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, env=env)

    def call(self, req):
        b = pickle.dumps(req, protocol=4)
        self.proc.stdin.write(struct.pack("<Q", len(b))); self.proc.stdin.write(b); self.proc.stdin.flush()
        n = struct.unpack("<Q", self.proc.stdout.read(8))[0]
        rep = pickle.loads(self.proc.stdout.read(n))
        if "error" in rep:
            raise RuntimeError(rep["error"])
        return rep

    def close(self):
        try:
            self.proc.stdin.close(); self.proc.wait(timeout=10)
        except Exception:                                    # noqa: BLE001
            self.proc.kill()


_SERVER = None


def dynotears_summary(X, p, T, lambda_w=0.1, lambda_a=0.1, w_threshold=0.1):
    """DYNOTEARS (lag 1) on pooled standardised transitions; defaults of causalnex except w_threshold = 0.1 on standardised data."""
    global _SERVER
    if _SERVER is None:
        _SERVER = DynotearsServer()
    prev, nxt = pooled_pairs(X, p, T)
    mu = np.vstack([prev, nxt]).mean(0); sdv = np.vstack([prev, nxt]).std(0)
    enc = lambda a: (np.ascontiguousarray(a, dtype=np.float64).tobytes(), a.shape)
    rep = _SERVER.call(dict(Z=enc((nxt - mu) / sdv), Zlag=enc((prev - mu) / sdv), lambda_w=lambda_w, lambda_a=lambda_a,
                            w_threshold=w_threshold))
    return np.array(rep["W"]), np.array(rep["A"])


def dynotears(X, p, T, **kw):
    W, A = dynotears_summary(X, p, T, **kw)
    return window_from_summary(W, A, p, T, thr_w=0.0, thr_a=0.0)


class TetradServer:
    """Persistent subprocess running tetrad_worker.py (Tetrad 7.6.10 via JPype1; JVM -Xmx2g)."""
    def __init__(self):
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        self.proc = subprocess.Popen([sys.executable, os.path.join(HERE, "tetrad_worker.py")], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)

    def call(self, method, X):
        b = pickle.dumps(dict(method=method, X=np.ascontiguousarray(X)), protocol=4)
        self.proc.stdin.write(struct.pack("<Q", len(b))); self.proc.stdin.write(b); self.proc.stdin.flush()
        h = self.proc.stdout.read(8)
        if len(h) < 8:
            raise RuntimeError("tetrad worker died")
        rep = pickle.loads(self.proc.stdout.read(struct.unpack("<Q", h)[0]))
        if "error" in rep:
            raise RuntimeError(rep["error"])
        return rep["Mdir"], rep["Mund"]

    def close(self):
        try:
            self.proc.stdin.close(); self.proc.wait(timeout=10)
        except Exception:                                    # noqa: BLE001
            self.proc.kill()


_TETRAD = None


def tetrad_call(method, X):
    global _TETRAD
    if _TETRAD is None:
        _TETRAD = TetradServer()
    return _TETRAD.call(method, X)
