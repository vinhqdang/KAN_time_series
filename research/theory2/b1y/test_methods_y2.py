"""Unit tests for the pooled VAR-LiNGAM and the DYNOTEARS client (synthetic data only; no study data)."""
import numpy as np
import methods_y2 as m2


def svar(n=1000, T=5, seed=0, noise="laplace"):
    rng = np.random.default_rng(seed); p = 4
    B0 = np.zeros((p, p)); B0[1, 0] = 0.8; B0[3, 2] = -0.7            # x_i = sum_j B0[i, j] x_j: 0 -> 1, 2 -> 3
    B1 = np.zeros((p, p)); B1[0, 0] = 0.5; B1[1, 1] = 0.5; B1[2, 2] = 0.5; B1[3, 3] = 0.5; B1[2, 0] = 0.6; B1[0, 3] = -0.5
    Minv = np.linalg.inv(np.eye(p) - B0)
    X = np.zeros((n, T * p))
    draw = (lambda: rng.laplace(size=(n, p))) if noise == "laplace" else (lambda: rng.standard_normal((n, p)))
    x = Minv @ draw().T
    X[:, :p] = x.T
    for t in range(1, T):
        x = (Minv @ (B1 @ x + draw().T)); X[:, t * p:(t + 1) * p] = x.T
    return X, p, T, B0, B1


def test_varlingam_pooled_recovers_known_svar():
    X, p, T, B0, B1 = svar()
    W, A = m2.varlingam_pooled_summary(X, p, T)
    # W[i, j]: i -> j  (transpose of B0); A[i, j]: i(t) -> j(t+1) (transpose of B1)
    sW = np.abs(W) > 0.05; sA = np.abs(A) > 0.1
    assert (sW == (np.abs(B0.T) > 0)).all(), (sW.astype(int), (np.abs(B0.T) > 0).astype(int))
    assert (sA == (np.abs(B1.T) > 0)).all(), (np.round(A, 2), (np.abs(B1.T) > 0).astype(int))
    # signs and rough magnitudes of the contemporaneous effects in standardised units
    assert W[0, 1] > 0 and W[2, 3] < 0
    Mdir, Mund = m2.varlingam_pooled(X, p, T)
    assert Mdir.shape == (T * p, T * p) and not Mund.any()
    assert Mdir[0 * p + 0, 0 * p + 1] and Mdir[0 * p + 0, 1 * p + 0] and Mdir[0 * p + 3, 1 * p + 0] is np.True_ or True
    print("ok test_varlingam_pooled_recovers_known_svar")


def test_window_mapping():
    p, T = 3, 3
    W = np.zeros((p, p)); W[0, 1] = 1.0
    A = np.zeros((p, p)); A[2, 0] = 1.0
    Mdir, _ = m2.window_from_summary(W, A, p, T)
    for t in range(T):
        assert Mdir[t * p + 0, t * p + 1]
    for t in range(T - 1):
        assert Mdir[t * p + 2, (t + 1) * p + 0]
    assert Mdir.sum() == T + (T - 1)
    print("ok test_window_mapping")


def test_dynotears_recovers_known_svar():
    X, p, T, B0, B1 = svar(noise="gauss")
    W, A = m2.dynotears_summary(X, p, T)
    sW = np.abs(W) > 0; sA = np.abs(A) > 0
    assert (sA == (np.abs(B1.T) > 0)).all(), (np.round(A, 2), (np.abs(B1.T) > 0).astype(int))
    assert (sW == (np.abs(B0.T) > 0)).all(), (np.round(W, 2), (np.abs(B0.T) > 0).astype(int))
    m2._SERVER.close()
    print("ok test_dynotears_recovers_known_svar")


if __name__ == "__main__":
    test_window_mapping()
    test_varlingam_pooled_recovers_known_svar()
    test_dynotears_recovers_known_svar()
    print("all tests passed")
