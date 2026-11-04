"""Graph-only predictions (no data): full-graph and subgraph predicted rel-sortability, relative counts,
non-relative pairs, for the WT random-walk system and the LT white-noise system."""
import sys, itertools
import numpy as np
sys.path.insert(0, sys.argv[1]); import cc_graph as G
for ch, seeds in [("wt", range(9002, 9017)), ("lt", range(9100, 9110))]:
    nodes, edges = G.system(ch); p = len(nodes); R = G.related(p, edges)
    ind, val = G.predicted_edges(range(p), edges, R)
    nonrel = [(a, b) for a, b in itertools.combinations(range(p), 2) if not R[a, b]]
    print(f"[{ch}] p={p} edges={len(edges)} predicted s_full={np.mean(val):.4f} edges valued 1/2: "
          f"{[(nodes[a], nodes[b]) for (a, b), v in zip(ind, val) if v == 0.5]}")
    print(f"  rho: {dict((nodes[v], int(R[v].sum())) for v in range(p))}")
    print(f"  pairs {p*(p-1)//2}, non-relative {len(nonrel)}")
    for d in (5, 8):
        s = []
        for sd in seeds:
            for S in G.sample_subgraphs(p, edges, d, 10, sd * 10 + d):
                s.append(np.mean(G.predicted_edges(S, edges, R)[1]))
        s = np.array(s)
        print(f"  d={d}: {len(s)} subgraphs, predicted s mean {s.mean():.3f}, sd {s.std():.3f}, frac >0.5 {np.mean(s>0.5):.2f}, frac ==0.5 {np.mean(s==0.5):.2f}, distinct {len(np.unique(s))}")
