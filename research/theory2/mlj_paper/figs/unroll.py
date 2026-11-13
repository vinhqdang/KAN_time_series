"""Graph helpers for the figures (pure Python, no numpy): time-unrolled DAGs, ancestors, relatives.

Node (v, t) with v in range(p), t in range(1, T+1).  Edge (i,t)->(j,t+1) for every summary edge (i,j) (self-loops
included).  `rel_counts` is independent of the theorems in the paper: it enumerates ancestors and descendants.
"""
from itertools import product


def with_loops(p, edges):
    return sorted(set(edges) | {(i, i) for i in range(p)})


def unrolled(p, S, T):
    """Return (nodes, edges) of G_T for the summary edge list S (self-loops added if missing)."""
    S = with_loops(p, S)
    nodes = [(v, t) for t in range(1, T + 1) for v in range(p)]
    edges = [((i, t), (j, t + 1)) for t in range(1, T) for (i, j) in S]
    return nodes, edges


def _closure(nodes, edges, forward):
    nxt = {n: set() for n in nodes}
    for a, b in edges:
        if forward:
            nxt[a].add(b)
        else:
            nxt[b].add(a)
    out = {}
    for n in nodes:
        seen, stack = {n}, [n]
        while stack:
            x = stack.pop()
            for y in nxt[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        out[n] = seen
    return out


def anc_desc(nodes, edges):
    return _closure(nodes, edges, False), _closure(nodes, edges, True)


def relatives(nodes, edges):
    """rel(y) = descendants of ancestors of y (contains y)."""
    anc, desc = anc_desc(nodes, edges)
    rel = {}
    for y in nodes:
        r = set()
        for a in anc[y]:
            r |= desc[a]
        rel[y] = r
    return rel


def rel_counts(p, S, T, B=0, window=True):
    """Counts of relatives (inside the window, relatives computed in the full graph G_{B+T}).
    B = 0 is the fixed start.  Returns {(v, t): count} with t = 1..T indexing the window."""
    nodes, edges = unrolled(p, S, B + T)
    rel = relatives(nodes, edges)
    wmask = {(v, t) for (v, t) in nodes if t > B}
    return {(v, t - B): len(rel[(v, t)] & wmask) for (v, t) in nodes if t > B}


def srho(p, S, T, B=0):
    """Edge-based rel-sortability of the T-slice window DAG, ties 1/2."""
    S = with_loops(p, S)
    cnt = rel_counts(p, S, T, B)
    tot = 0.0
    m = 0
    for t in range(1, T):
        for (i, j) in S:
            a, b = cnt[(i, t)], cnt[(j, t + 1)]
            tot += 1.0 if a < b else (0.5 if a == b else 0.0)
            m += 1
    return tot / m


def dist_matrix(p, S):
    INF = 10 ** 9
    d = [[INF] * p for _ in range(p)]
    for i in range(p):
        d[i][i] = 0
    for (i, j) in S:
        if i != j:
            d[i][j] = min(d[i][j], 1)
    for k in range(p):
        for i in range(p):
            for j in range(p):
                if d[i][k] + d[k][j] < d[i][j]:
                    d[i][j] = d[i][k] + d[k][j]
    return d, INF


def delta(p, S):
    d, INF = dist_matrix(p, S)
    return max(x for r in d for x in r if x < INF)


# ---- summary graph families used in the figures -----------------------------------------------
def chain(p):
    return [(i, i + 1) for i in range(p - 1)]


def ring(p):
    return [(i, (i + 1) % p) for i in range(p)]


def intree(depth):
    """Complete binary in-tree: leaves -> root. Node 0 is the root."""
    p = 2 ** (depth + 1) - 1
    return p, [(c, (c - 1) // 2) for c in range(1, p)]


def outtree(depth):
    p = 2 ** (depth + 1) - 1
    return p, [((c - 1) // 2, c) for c in range(1, p)]


if __name__ == "__main__":
    p, S = 3, chain(3)
    c = rel_counts(p, S, 4)
    for v in range(p):
        print("abc"[v], [c[(v, t)] for t in range(1, 5)])
    print("srho T=4 fixed start", srho(p, S, 4), "limit check T=40", srho(p, S, 40), "stationary", srho(p, S, 6, B=10))
    rel = relatives(*unrolled(3, chain(3), 4))
    print(sorted(rel[(2, 2)]))
    print(sorted(rel[(0, 2)]))
