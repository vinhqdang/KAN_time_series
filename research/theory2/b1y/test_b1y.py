"""Unit tests (tiny cases, population quantities and scoring conventions only; no learner is run, no study data).
Run: OMP_NUM_THREADS=1 /usr/bin/python3 test_b1y.py
"""
import numpy as np
from common_y import *
import common_y as cy
import seeds as sd_


def tiny(fam="chain", p=3, seed=0):
    rng = np.random.default_rng([401, FAMILIES.index(fam), p, seed])
    S = summary_graph(fam, p, rng); A, sd = weights(S, rng, "het"); return S, A, sd


def test_cf_endpoints_and_psd():
    for fam in ("chain", "ring", "intree", "ER2", "symER"):
        S, A, sd = tiny(fam, 4, 1); T = 3; D = np.diag(sd ** 2)
        assert np.allclose(cy.start_cov("G1", A, sd, T), D)
        assert np.allclose(cy.start_cov("G2s", A, sd, T), stationary_cov(A, sd))
        # G2 burn-in covariance equals the stationary one up to the mixing error
        assert np.abs(cy.start_cov("G2", A, sd, T) - stationary_cov(A, sd)).max() < 1e-6
        for arm in ARMS:
            S1 = cy.start_cov(arm, A, sd, T)
            assert np.linalg.eigvalsh(S1).min() > 0, (fam, arm)


def test_patterns():
    """CF1: population pattern = G1's; CF2 (any lam>0): population pattern = G2's; lam_star keeps new pairs undetectable."""
    eps = cy.eps_threshold()
    for fam in ("chain", "intree", "ER2", "ring"):
        S, A, sd = tiny(fam, 4, 2)
        for T in (3, 5):
            C1 = window_cov(A, sd, T, np.diag(sd ** 2)); P1 = population_pattern(C1)
            C2 = window_cov(A, sd, T, stationary_cov(A, sd)); P2 = population_pattern(C2)
            for arm in CF1_C:
                Ca = window_cov(A, sd, T, cy.start_cov(arm, A, sd, T))
                assert (population_pattern(Ca) == P1).all(), (fam, T, arm)
            lam, degen = cy.lam_star(A, sd, T)
            assert not degen and 0 < lam <= 1
            for arm in CF2_M:
                Ca = window_cov(A, sd, T, cy.start_cov(arm, A, sd, T))
                assert (population_pattern(Ca) == P2).all(), (fam, T, arm)
            # undetectability of the new pairs at lam_star
            C = window_cov(A, sd, T, cy.start_cov("CF2_m1", A, sd, T)); R = np.abs(corr_of(C))
            Z = zero_pairs_G1(A, sd, T)
            assert R[Z].max() <= cy.FRAC_EPS * eps * 1.0000001
            # detectable pattern of CF2_m1 vs G1: identical except for pairs near the threshold (reported, not asserted)
            R1 = np.abs(corr_of(C1))
            flips = ((R > eps) != (R1 > eps)).sum() / 2
            assert flips <= 0.1 * R.size, (fam, T, flips)      # sanity bound only; the study reports the exact number


def test_exact_cov_matches_simulation():
    S, A, sd = tiny("intree", 3, 3); T = 3
    rng = np.random.default_rng(12345)
    for arm in ("G1", "G2", "CF1_2", "CF2_m3"):
        X = cy.simulate_arm(arm, A, sd, T, 400000, rng)
        Cemp = np.cov(X, rowvar=False)
        Cpop = window_cov(A, sd, T, cy.start_cov(arm, A, sd, T))
        assert np.abs(Cemp - Cpop).max() < 0.02 * np.abs(Cpop).max() + 0.01, (arm, np.abs(Cemp - Cpop).max())


def test_vm_factors():
    S, A, sd = tiny("chain", 4, 4); T = 5
    f = cy.vm_factors(A, sd, T, "G1", "G2")
    v1 = cy.marginal_var(A, sd, T, "G1"); v2 = cy.marginal_var(A, sd, T, "G2")
    assert np.allclose(v1 * f ** 2, v2)
    C1 = window_cov(A, sd, T, np.diag(sd ** 2)); Cs = C1 * np.outer(f, f)
    assert np.allclose(corr_of(C1), corr_of(Cs))               # scale invariance of correlations
    # G1's variance grows along the slices, G2's is constant over slices (it still grows along the chain nodes):
    # G1 is more var-sortable than G2, and after matching the two profiles are identical
    G = true_dag(S, T)
    s1 = sortability(v1, G, tol=1e-9 * v1.max()); s2 = sortability(v2, G, tol=1e-9 * v2.max())
    assert s1 > s2 + 0.05, (s1, s2)
    assert abs(sortability(v1 * f ** 2, G, tol=1e-9 * v2.max()) - s2) < 1e-12


def test_scoring_conventions():
    """Truth gives 1, its transpose gives 0 (guards the below-chance diagnosis); x1 masks are right."""
    S, A, sd = tiny("intree", 4, 5); T = 3; p = 4
    G = true_dag(S, T); rng = np.random.default_rng(0)
    perm = rng.permutation(T * p); Gp = G[np.ix_(perm, perm)]; sl = slice_of_cols(perm, p)
    assert anc_oa(Gp, Gp) == 1.0 and anc_oa(Gp, Gp.T) == 0.0 and anc_oa(Gp, np.zeros_like(Gp)) == 0.5
    true_order = np.argsort(sl, kind="stable")                  # time order (labels used: reference only)
    assert order_accuracy(Gp, true_order) == 1.0 and oa_x1(Gp, true_order, sl) == 1.0
    assert order_accuracy(Gp, true_order[::-1]) == 0.0 and oa_x1(Gp, true_order[::-1], sl) == 0.0
    m = mask_inside_first_slice(sl)
    assert m.sum() == p * p and (m == m.T).all()
    E = np.argwhere(Gp); n_keep = (sl[E[:, 0]] != 0).sum()
    assert 0 < n_keep < len(E)                                 # edges leaving slice 1 are dropped, others kept


def test_exact_gap_matches_theory():
    """Population oracle with eps -> 0 reproduces the exact s_rho of Thm 1.5 / 1.9 (graph check, as in b1x/oracle_b3.py)."""
    for fam in ("chain", "ring", "ER2", "symER"):
        S, A, sd = tiny(fam, 5, 6)
        for T in (3, 5):
            G = true_dag(S, T)
            cnt1 = (np.abs(corr_of(window_cov(A, sd, T, np.diag(sd ** 2)))) > 1e-12).sum(0).astype(float)
            cnt2 = (np.abs(corr_of(window_cov(A, sd, T, cy.start_cov("G2s", A, sd, T)))) > 1e-12).sum(0).astype(float)
            assert abs(sortability(cnt1, G) - srho_G1_exact(S, T)) < 1e-12, (fam, T)
            assert abs(sortability(cnt2, G) - srho_stationary(S)) < 1e-12, (fam, T)


def test_seed_disjointness():
    new_prefixes = {sd_.KEY_GRAPH, sd_.KEY_DATA, sd_.KEY_Q5_GRAPH, sd_.KEY_Q5_DATA}
    new_seeds = set().union(*[set(v) for v in sd_.SEEDS.values()])
    for name, e in sd_.EARLIER.items():
        assert not (new_prefixes & e["prefixes"]), name
        assert not (new_seeds & e["seeds"]), name
    # no seed is shared between the new studies' test and validation sets
    tests = [set(sd_.SEEDS[k]) for k in ("q1_test", "q1_dose", "q2_test", "q5_test")]
    vals = [set(sd_.SEEDS[k]) for k in ("q1_val", "q2_val", "q5_val")]
    for i, a in enumerate(tests + vals):
        for b in (tests + vals)[i + 1:]:
            assert not (a & b)
    # streams: first draws of all new keys differ from those of all earlier keys (spot check on a grid)
    fam_idx = range(7); ps = (5, 10, 20)
    old = set()
    for fi in fam_idx:
        for p in ps:
            for s in list(range(1000, 1030)) + list(range(9000, 9005)):
                old.add(tuple(np.random.default_rng([fi, p, s]).random(3)))
            for s in range(3000, 3012):
                old.add(tuple(np.random.default_rng([301, fi, p, s]).random(3)))
            for key in (101, 102):
                for s in range(2000, 2025):
                    old.add(tuple(np.random.default_rng([key, fi, p, s]).random(3)))
    for fam in FAMILIES:
        for p in ps:
            for s in sd_.SEEDS["q1_test"][:5]:
                assert tuple(sd_.graph_rng(fam, p, s).random(3)) not in old


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
    print("all tests passed")
