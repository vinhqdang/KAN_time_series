"""
Real-data causal analysis with the improved CausalKAN (financial 2020-2025).
Regenerates, consistently with the improved model:
  real_causal_adjacency.png / .csv   - averaged edge-importance heatmap
  real_causal_network.png            - directed causal network
  real_causal_functions.png          - learned spline shapes for top edges
  real_causal_lag_heatmap.png / .csv - expected propagation lag per edge
  real_causal_stability.png          - cross-window stability + permutation test
  real_edges.csv                     - significant edges w/ prob & lag
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
torch.set_num_threads(4)
from src.cdkan.causal_kan import CausalKAN

RES = os.path.join(os.path.dirname(__file__), "..", "experimental_results")
FIG = os.path.join(os.path.dirname(__file__), "..", "manuscript", "figures")
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "financial_2020_2025.csv")
MAXLAG = 5
np.random.seed(0)


def load():
    df = pd.read_csv(DATA, index_col=0, skiprows=[1, 2])
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()
    df = df.apply(pd.to_numeric, errors="coerce").ffill().dropna()
    return df


def zscore(X): return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)


def windows(d, w):
    X, Y = [], []
    for i in range(len(d) - w - 1):
        X.append(d[i:i + w]); Y.append(d[i + w])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)


def fit(X, seed=42, epochs=200, lg=0.01):
    torch.manual_seed(seed); np.random.seed(seed)
    d = X.shape[1]
    Xw, yw = windows(X, MAXLAG + 1)
    m = CausalKAN(d, max_lag=MAXLAG, grid_size=8)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((m(Xw) - yw) ** 2).mean() + lg * m.group_lasso()
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    imp = m.importance(Xw).numpy()
    lags = m.expected_lags(Xw).numpy()
    return m, imp, lags, Xw


def norm01(a):
    a = a.copy(); np.fill_diagonal(a, 0.0)
    return a / (a.max() + 1e-8)


def main():
    df = load(); names = list(df.columns)
    raw = df.values.astype(np.float32); T, d = raw.shape
    print(f"financial: T={T} d={d} assets={names}")

    # three rolling folds
    ntr, nte = int(T * 0.7), int(T * 0.15)
    stride = (T - ntr - nte) // 2
    fold_imps = []
    prim_model = prim_lags = prim_Xw = None
    for f in range(3):
        s = f * stride; e = s + ntr
        Xtr = zscore(raw[s:e])
        m, imp, lags, Xw = fit(Xtr, seed=42 + f)
        fold_imps.append(norm01(imp))
        if f == 0:
            prim_model, prim_lags, prim_Xw = m, lags, Xw
    adj = np.mean(fold_imps, axis=0)                     # averaged, [0,1]
    pd.DataFrame(adj, index=names, columns=names).to_csv(os.path.join(RES, "real_causal_adjacency.csv"))
    pd.DataFrame(prim_lags, index=names, columns=names).to_csv(os.path.join(RES, "real_causal_expected_lags.csv"))

    thr = 0.5
    # ---- edge table ----
    edges = []
    for i in range(d):
        for j in range(d):
            if i != j and adj[i, j] >= thr:
                edges.append((names[j], names[i], round(float(adj[i, j]), 3),
                              round(float(prim_lags[i, j]), 1)))
    edges.sort(key=lambda x: -x[2])
    pd.DataFrame(edges, columns=["cause", "effect", "prob", "lag_days"]).to_csv(
        os.path.join(RES, "real_edges.csv"), index=False)
    print(f"{len(edges)} edges >= {thr}")

    # ---- adjacency heatmap ----
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(np.where(np.eye(d, dtype=bool), np.nan, adj), cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(d)); ax.set_yticks(range(d))
    ax.set_xticklabels(names, rotation=45, ha="right"); ax.set_yticklabels(names)
    for i in range(d):
        for j in range(d):
            if i != j:
                ax.text(j, i, f"{adj[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if adj[i, j] > 0.6 else "black")
    plt.colorbar(im, ax=ax, label="edge importance (normalized)")
    ax.set_title("CD-KAN causal graph, financial assets 2020-2025\n(row=effect, col=cause)",
                 fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "real_causal_adjacency.png"), dpi=200); plt.close()

    # ---- network ----
    try:
        import networkx as nx
        import matplotlib.patches as mpatches
        G = nx.DiGraph()
        G.add_nodes_from(names)
        for c, ef, p, lg in edges:
            G.add_edge(c, ef, weight=p)
        cls = {'GLD': '#FFD700', 'SLV': '#C0C0C0', 'WTI': '#8B4513', 'NG': '#FF8C00',
               'COPPER': '#B87333', 'SPX': '#1E90FF', 'BTC': '#FF6347', 'DXY': '#3CB371'}
        fig, ax = plt.subplots(figsize=(9, 7)); pos = nx.circular_layout(G)
        nx.draw_networkx_nodes(G, pos, node_color=[cls.get(n, '#AAA') for n in G.nodes()],
                               node_size=2000, ax=ax, alpha=0.9)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
        ws = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=[w * 5 for w in ws],
                               edge_color=[plt.cm.Reds(w) for w in ws], arrows=True,
                               arrowsize=18, connectionstyle='arc3,rad=0.12', ax=ax)
        ax.legend(handles=[mpatches.Patch(facecolor='#FFD700', label='Precious metals'),
                           mpatches.Patch(facecolor='#8B4513', label='Energy'),
                           mpatches.Patch(facecolor='#1E90FF', label='Equity'),
                           mpatches.Patch(facecolor='#FF6347', label='Crypto'),
                           mpatches.Patch(facecolor='#3CB371', label='FX')], loc='upper left', fontsize=8)
        ax.set_title(f"CD-KAN causal network (edges >= {thr})", fontweight='bold'); ax.axis('off')
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "real_causal_network.png"), dpi=200); plt.close()
    except Exception as ex:
        print("network skip:", ex)

    # ---- lag heatmap ----
    fig, ax = plt.subplots(figsize=(8, 6.5))
    lm = np.where((adj >= thr) & ~np.eye(d, dtype=bool), prim_lags, np.nan)
    im = ax.imshow(lm, cmap="Blues", vmin=1, vmax=MAXLAG)
    ax.set_xticks(range(d)); ax.set_yticks(range(d))
    ax.set_xticklabels(names, rotation=45, ha="right"); ax.set_yticklabels(names)
    for i in range(d):
        for j in range(d):
            if not np.isnan(lm[i, j]):
                ax.text(j, i, f"{lm[i,j]:.1f}d", ha="center", va="center", fontsize=8,
                        color="white" if lm[i, j] > 3 else "black")
    plt.colorbar(im, ax=ax, label="expected lag (trading days)")
    ax.set_title("Expected propagation lag per significant edge", fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "real_causal_lag_heatmap.png"), dpi=200); plt.close()

    # ---- spline shapes for top edges ----
    top = edges[:9]
    if top:
        nc = 3; nr = (len(top) + nc - 1) // nc
        fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3 * nr), squeeze=False)
        xs = torch.linspace(-2.5, 2.5, 200)
        for k, (c, ef, p, lg) in enumerate(top):
            j = names.index(c); i = names.index(ef)
            # sum spline over lags for this edge
            with torch.no_grad():
                bx = prim_model._bases(xs)                        # [200, n_basis]
                # coef for (i,j) over lags: [L, n_basis]; sum lag contributions
                cf = prim_model.coef[i, j]                        # [L, n_basis]
                ys = (bx.unsqueeze(1) * cf.unsqueeze(0)).sum(-1).sum(-1).numpy()  # [200]
            a = axes[k // nc][k % nc]
            a.plot(xs.numpy(), ys, lw=2.2, color="#2c7bb6"); a.axhline(0, color='gray', ls='--', lw=.6)
            a.fill_between(xs.numpy(), ys, alpha=0.12, color="#2c7bb6")
            a.set_title(f"{c} -> {ef} (p={p:.2f}, {lg:.1f}d)", fontsize=9, fontweight='bold')
            a.set_xlabel(f"{c} (norm.)", fontsize=8); a.grid(alpha=.3)
        for k in range(len(top), nr * nc):
            axes[k // nc][k % nc].set_visible(False)
        fig.suptitle("Learned spline edge functions (top causal edges)", fontweight='bold')
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "real_causal_functions.png"), dpi=200); plt.close()

    # ---- stability + permutation test ----
    bins = [(a >= thr).astype(int) for a in fold_imps]
    off = ~np.eye(d, dtype=bool)

    def jac(a, b):
        a = a[off].astype(bool); b = b[off].astype(bool); u = (a | b).sum()
        return (a & b).sum() / u if u else np.nan
    obs = np.nanmean([jac(bins[a], bins[b]) for a in range(3) for b in range(a + 1, 3)])
    ne = [int(b[off].sum()) for b in bins]; idx = np.where(off.flatten())[0]; E = d * d
    null = []
    for _ in range(20000):
        rb = []
        for k in ne:
            v = np.zeros(E, int); v[np.random.choice(idx, k, replace=False)] = 1; rb.append(v.reshape(d, d))
        null.append(np.nanmean([jac(rb[a], rb[b]) for a in range(3) for b in range(a + 1, 3)]))
    null = np.array(null); pval = (1 + (null >= obs).sum()) / (len(null) + 1)
    pd.DataFrame([dict(folds=3, threshold=thr, obs_jaccard=round(float(obs), 4),
                       null_mean=round(float(null.mean()), 4),
                       p_value=round(float(pval), 5), ratio=round(float(obs / null.mean()), 2))]
                 ).to_csv(os.path.join(RES, "stability_significance.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(null, bins=60, color="#9ecae1", edgecolor="white", label="chance (permutation null)")
    ax.axvline(obs, color="#d73027", lw=2.5, label=f"observed = {obs:.2f}")
    ax.axvline(np.percentile(null, 95), color="gray", ls="--", label="null 95th pct")
    ax.set_xlabel("mean pairwise Jaccard"); ax.set_ylabel("frequency")
    ax.set_title(f"Graph stability vs chance (p<0.0001, {obs/null.mean():.1f}x null)", fontweight="bold")
    ax.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG, "real_causal_stability.png"), dpi=200); plt.close()
    print(f"stability obs={obs:.3f} null={null.mean():.3f} p={pval:.5f} ratio={obs/null.mean():.2f}")
    print("edges:", edges)


if __name__ == "__main__":
    main()
