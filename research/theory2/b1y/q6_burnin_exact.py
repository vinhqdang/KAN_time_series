"""Q6 (graph-only exact computation, no data, no learner): per-graph minimal sufficient burn-in B_min versus
Delta = largest finite shortest-path distance of the summary graph.

For a summary graph S (self-loop on every node) and horizon T, rel_window_enum(S, T, B) (b1/common.py, bitset
enumeration, independent of the theorems) returns the number of relatives of every window node when the window is the
last T slices of the unrolled DAG G_{B+T}. Reference: B_ref = Delta + 2p >= Delta (by Thm 1.9 the stationary relatives).
  * B_min_cnt: smallest B such that the relatives counts of all window nodes equal those at B_ref for every B' in [B, B_ref];
  * B_min_s:   smallest B such that the window rel-sortability s_rho (ties 1/2, on the T-slice DAG G_T) equals its
               value at B_ref for every B' in [B, B_ref].
Reported: violations of B_min <= Delta (the theorem), how often B = Delta-1 is insufficient, and B_min by family.
Run: OMP_NUM_THREADS=1 /usr/bin/python3 q6_burnin_exact.py > results/q6_burnin_exact.txt
"""
import sys, os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "b1"))
from common import rel_window_enum, delta_max, sortability, true_dag, summary_graph

TS = (2, 3, 4, 5)


def profile(S, T):
    p = S.shape[0]; Dl = delta_max(S); Bref = Dl + 2 * p
    G = true_dag(S, T)
    cnts = [rel_window_enum(S, T, B) for B in range(Bref + 1)]
    sv = [sortability(c.astype(float), G) for c in cnts]
    ref_c, ref_s = cnts[Bref], sv[Bref]
    bmin_c = Bref
    for B in range(Bref, -1, -1):
        if np.array_equal(cnts[B], ref_c): bmin_c = B
        else: break
    bmin_s = Bref
    for B in range(Bref, -1, -1):
        if abs(sv[B] - ref_s) < 1e-12: bmin_s = B
        else: break
    mono = all((cnts[B + 1] >= cnts[B]).all() for B in range(Bref))
    return Dl, bmin_c, bmin_s, mono


def chain(p):
    S = np.zeros((p, p), bool)
    for i in range(p - 1): S[i, i + 1] = True
    np.fill_diagonal(S, True); return S


def ring(p):
    S = np.zeros((p, p), bool)
    for i in range(p): S[i, (i + 1) % p] = True
    np.fill_diagonal(S, True); return S


def main():
    out = {}
    print("Q6: B_min (relatives counts equal stationary) and B_min_s (s_rho equals stationary) vs Delta")
    for name, mk in (("chain", chain), ("ring", ring)):
        print(f"\n{name} (self-loops on every node); columns: p Delta | T -> (B_min_cnt, B_min_s) | monotone")
        for p in range(2, 9):
            S = mk(p); row = []
            for T in TS:
                Dl, bc, bs, mono = profile(S, T)
                row.append((T, bc, bs, mono))
            print(f"p={p} Delta={Dl} | " + "  ".join(f"T={T}:({bc},{bs})" for T, bc, bs, m in row) + f" | monotone={all(m for *_, m in row)}")
            out[f"{name}_{p}"] = dict(Delta=Dl, rows=[dict(T=T, B_min_cnt=bc, B_min_s=bs) for T, bc, bs, m in row])
    # random summary graphs (exploratory, graph-only; fresh RNG key 499)
    rng = np.random.default_rng([499, 0])
    n_g = 0; viol = 0; tight_c = 0; tight_s = 0; cases = 0; slack_c = {}; slack_s = {}
    first_viol = None
    for i in range(400):
        p = int(rng.integers(3, 8)); c = float(rng.choice([1.0, 2.0, 3.0]))
        S = rng.random((p, p)) < c / p; np.fill_diagonal(S, False)
        if not S.any(): continue
        np.fill_diagonal(S, True); n_g += 1
        for T in (3, 5):
            Dl, bc, bs, mono = profile(S, T); cases += 1
            if bc > Dl or bs > Dl:
                viol += 1; first_viol = first_viol or (S.astype(int).tolist(), T)
            tight_c += (bc == Dl and Dl >= 1); tight_s += (bs == Dl and Dl >= 1)
            slack_c[Dl - bc] = slack_c.get(Dl - bc, 0) + 1; slack_s[Dl - bs] = slack_s.get(Dl - bs, 0) + 1
    print(f"\nrandom graphs: {n_g} graphs x T in (3,5) = {cases} cases; violations of B_min <= Delta: {viol} {first_viol or ''}")
    print(f"B_min_cnt == Delta (tight, Delta>=1) in {tight_c}/{cases}; B_min_s == Delta in {tight_s}/{cases}")
    print("slack Delta - B_min_cnt histogram:", dict(sorted(slack_c.items())))
    print("slack Delta - B_min_s   histogram:", dict(sorted(slack_s.items())))
    out["random"] = dict(graphs=n_g, cases=cases, violations=viol, tight_cnt=tight_c, tight_s=tight_s,
                         slack_cnt=slack_c, slack_s=slack_s)
    json.dump(out, open(os.path.join(HERE, "results", "q6_burnin_exact.json"), "w"), indent=1, default=int)


if __name__ == "__main__":
    main()
