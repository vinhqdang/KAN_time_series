"""Learner wrappers for Q2 and Q5. STATUS: written before any run; the conversions below are UNTESTED on data and are
checked in step 0 of PLAN.md (a toy graph with a known CPDAG, no study data). Importing this module runs nothing.
Every wrapper takes the (n, d) data matrix in the permuted-column frame and returns (Mdir, Mund) as boolean (d, d)
matrices: Mdir[i, j] = edge i -> j, Mund symmetric undirected part, as in b1x/run_b1x.py.
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "b1x"))


def dagma_hp(X, lambda1=0.03, w_threshold=0.3):
    from dagma.linear import DagmaLinear
    import contextlib, io
    m = DagmaLinear(loss_type="l2", verbose=False)
    with contextlib.redirect_stderr(io.StringIO()):
        W = m.fit(X, lambda1=lambda1, w_threshold=w_threshold)
    return W != 0, np.zeros(W.shape, bool)


def notears_hp(X, lambda1=0.1, w_threshold=0.3):
    from methods_x import notears_linear
    W = notears_linear(X, lambda1=lambda1, w_threshold=w_threshold)
    return W != 0, np.zeros(W.shape, bool)


def _from_general_graph(Gr, d):
    """causal-learn GeneralGraph.graph: G[j, i] = 1 and G[i, j] = -1 means i -> j; G[i, j] = G[j, i] = -1 undirected."""
    M = np.asarray(Gr.graph)
    Mdir = np.zeros((d, d), bool); Mund = np.zeros((d, d), bool)
    for i in range(d):
        for j in range(d):
            if M[j, i] == 1 and M[i, j] == -1: Mdir[i, j] = True
            elif M[i, j] == -1 and M[j, i] == -1: Mund[i, j] = True
    return Mdir, Mund


def ges_cpdag(X):
    from causallearn.search.ScoreBased.GES import ges
    r = ges(X, score_func="local_score_BIC", parameters={"lambda_value": 2.0}); return _from_general_graph(r["G"], X.shape[1])


def grasp_cpdag(X):
    from causallearn.search.PermutationBased.GRaSP import grasp
    return _from_general_graph(grasp(X, score_func="local_score_BIC_from_cov", depth=3, verbose=False, parameters={"lambda_value": 2.0}), X.shape[1])


def boss_cpdag(X):
    from causallearn.search.PermutationBased.BOSS import boss
    return _from_general_graph(boss(X, score_func="local_score_BIC_from_cov", verbose=False, parameters={"lambda_value": 2.0}), X.shape[1])


def directlingam_dag(X):
    import lingam
    m = lingam.DirectLiNGAM(random_state=0); m.fit(X)
    B = m.adjacency_matrix_                      # B[i, j] != 0: j -> i
    return (B != 0).T, np.zeros(B.shape, bool)


def var_ols_labelled(X, p, T, threshold=0.1):
    """Labelled reference (time labels used): pooled lag-1 OLS over the T-1 transitions of every trajectory; edge i -> j in
    every slice transition if |M[i, j]| > threshold on standardised data. Stands in for DYNOTEARS/VAR-LiNGAM, which are
    not installable here (see PLAN.md Q5)."""
    Z = (X - X.mean(0)) / X.std(0)
    prev = np.vstack([Z[:, t * p:(t + 1) * p] for t in range(T - 1)]); nxt = np.vstack([Z[:, (t + 1) * p:(t + 2) * p] for t in range(T - 1)])
    M = np.linalg.lstsq(prev, nxt, rcond=None)[0]        # nxt ~ prev @ M, M[i, j]: i at t -> j at t+1
    Wp = np.abs(M) > threshold
    d = T * p; Mdir = np.zeros((d, d), bool)
    for t in range(T - 1):
        Mdir[t * p:(t + 1) * p, (t + 1) * p:(t + 2) * p] = Wp
    return Mdir, np.zeros((d, d), bool)
