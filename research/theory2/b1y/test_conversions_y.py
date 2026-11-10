"""Step-0 checks of the learner output conversions on a toy graph with a known CPDAG (synthetic data only; no study data)."""
import numpy as np
import methods_y as my
import methods_y2 as m2
import methods_tetrad as mt


def toy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n); x1 = rng.standard_normal(n); x2 = 0.8 * x0 + 0.8 * x1 + rng.standard_normal(n); x3 = 0.7 * x2 + rng.standard_normal(n)
    return np.c_[x0, x1, x2, x3]      # 0 -> 2 <- 1, 2 -> 3: the v-structure orients every edge


def edges(D):
    return sorted(map(tuple, np.argwhere(D).tolist()))


def test_cpdag_conversions():
    X = toy(); want = [(0, 2), (1, 2), (2, 3)]
    for name, f in (("causallearn_ges", my.ges_cpdag), ("causallearn_grasp", my.grasp_cpdag), ("causallearn_boss", my.boss_cpdag),
                    ("tetrad_fges", mt.fges_tetrad), ("tetrad_boss", mt.boss_tetrad), ("tetrad_grasp", mt.grasp_tetrad)):
        D, U = f(X)
        assert edges(D) == want and not U.any(), (name, edges(D), edges(np.triu(U)))
    # an undirected chain a - b - c (no v-structure): both edges must come back undirected
    rng = np.random.default_rng(1); n = 4000
    a = rng.standard_normal(n); b = 0.8 * a + rng.standard_normal(n); c = 0.8 * b + rng.standard_normal(n); Y = np.c_[a, b, c]
    for name, f in (("causallearn_ges", my.ges_cpdag), ("causallearn_boss", my.boss_cpdag), ("tetrad_fges", mt.fges_tetrad)):
        D, U = f(Y)
        assert not D.any() and edges(np.triu(U)) == [(0, 1), (1, 2)], (name, edges(D), edges(np.triu(U)))
    print("ok test_cpdag_conversions")


def test_directlingam_direction():
    rng = np.random.default_rng(2); n = 3000
    a = rng.laplace(size=n); b = 0.9 * a + rng.laplace(size=n); c = -0.8 * b + rng.laplace(size=n)
    D, U = my.directlingam_dag(np.c_[a, b, c])
    assert edges(D) == [(0, 1), (1, 2)] and not U.any(), edges(D)
    print("ok test_directlingam_direction")


def test_var_ols_labelled_mapping():
    rng = np.random.default_rng(3); n = 1500; p = 3; T = 4
    M = np.zeros((p, p)); M[0, 1] = 0.7; M[2, 2] = 0.6        # x_{t+1}[j] depends on x_t[i] with M[i, j]: 0 -> 1 and 2 -> 2
    X = np.zeros((n, T * p)); x = rng.standard_normal((n, p)); X[:, :p] = x
    for t in range(1, T):
        x = x @ M + rng.standard_normal((n, p)); X[:, t * p:(t + 1) * p] = x
    D, U = my.var_ols_labelled(X, p, T)
    for t in range(T - 1):
        assert D[t * p + 0, (t + 1) * p + 1] and D[t * p + 2, (t + 1) * p + 2]
    assert D.sum() == 2 * (T - 1) and not U.any()
    print("ok test_var_ols_labelled_mapping")


if __name__ == "__main__":
    test_cpdag_conversions(); test_directlingam_direction(); test_var_ols_labelled_mapping(); print("all tests passed")
