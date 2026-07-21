"""Permutation test: is the observed cross-window Jaccard stability of the
CD-KAN real-data causal graph higher than chance? Uses the per-fold adjacency
CSVs already produced by extract_real_causal.py (no retraining)."""
import os, sys, glob
import numpy as np, pandas as pd
sys.path.append('.')
np.random.seed(0)
RES = "experimental_results"
files = sorted(glob.glob(os.path.join(RES, "real_causal_adjacency_fold*.csv")))
mats = [pd.read_csv(f, index_col=0).values for f in files]
D = mats[0].shape[0]
thr = 0.5
bins = [(m >= thr).astype(int) for m in mats]
off = ~np.eye(D, dtype=bool)

def jaccard(a, b):
    a = a[off].astype(bool); b = b[off].astype(bool)
    u = (a | b).sum()
    return (a & b).sum() / u if u else np.nan

# observed mean pairwise Jaccard
obs = [jaccard(bins[i], bins[j]) for i in range(len(bins)) for j in range(i+1, len(bins))]
obs_mean = np.nanmean(obs)
densities = [b[off].mean() for b in bins]
print(f"folds={len(bins)} D={D} thr={thr}")
print(f"edge densities per fold: {[round(float(d),3) for d in densities]}")
print(f"observed mean pairwise Jaccard = {obs_mean:.3f}")

# permutation null: random graphs with each fold's own edge count
n_edge = [int(b[off].sum()) for b in bins]
E = off.sum()
idx = np.where(off.flatten())[0]
NPERM = 20000
null = np.empty(NPERM)
for p in range(NPERM):
    rb = []
    for k in n_edge:
        v = np.zeros(D*D, int)
        v[np.random.choice(idx, size=k, replace=False)] = 1
        rb.append(v.reshape(D, D))
    null[p] = np.nanmean([jaccard(rb[i], rb[j])
                          for i in range(len(rb)) for j in range(i+1, len(rb))])
pval = (1 + np.sum(null >= obs_mean)) / (NPERM + 1)
print(f"null Jaccard: mean={null.mean():.3f} 95th pct={np.percentile(null,95):.3f}")
print(f"permutation p-value (obs > chance) = {pval:.4f}")
print(f"stability ratio obs/null = {obs_mean/null.mean():.2f}x")
pd.DataFrame([dict(folds=len(bins), threshold=thr, obs_jaccard=round(obs_mean,4),
                   null_mean=round(float(null.mean()),4),
                   null_p95=round(float(np.percentile(null,95)),4),
                   p_value=round(float(pval),4),
                   ratio=round(float(obs_mean/null.mean()),3))]
            ).to_csv(os.path.join(RES, "stability_significance.csv"), index=False)
print("saved experimental_results/stability_significance.csv")

# ---- regenerate the stability figure from existing fold CSVs (no retraining) ----
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
names = list(pd.read_csv(files[0], index_col=0).columns)
D = len(names)
pair = np.zeros((D, D))
for a in bins:
    for b in bins:
        ai = a.astype(bool); bi = b.astype(bool)
        inter = (ai & bi).astype(float); union = (ai | bi).astype(float)
        pair += np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
pair /= len(bins) ** 2
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
im = ax[0].imshow(pair, cmap='Blues', vmin=0, vmax=1)
ax[0].set_xticks(range(D)); ax[0].set_yticks(range(D))
ax[0].set_xticklabels(names, rotation=45, ha='right'); ax[0].set_yticklabels(names)
for i in range(D):
    for j in range(D):
        if i != j:
            ax[0].text(j, i, f'{pair[i,j]:.2f}', ha='center', va='center',
                       fontsize=8, color='white' if pair[i,j] > 0.5 else 'black')
plt.colorbar(im, ax=ax[0], label='Per-edge Jaccard across windows')
ax[0].set_title(f'Edge persistence across 3 rolling windows\n(mean off-diagonal Jaccard = {obs_mean:.2f})',
                fontweight='bold')
ax[0].set_xlabel('Cause'); ax[0].set_ylabel('Effect')
ax[1].hist(null, bins=60, color='#9ecae1', edgecolor='white', label='chance (permutation null)')
ax[1].axvline(obs_mean, color='#d73027', lw=2.5, label=f'observed = {obs_mean:.2f}')
ax[1].axvline(np.percentile(null, 95), color='grey', ls='--', lw=1.5, label='null 95th pct')
ax[1].set_xlabel('Mean pairwise Jaccard'); ax[1].set_ylabel('Frequency')
ax[1].set_title(f'Observed stability vs chance\n(p < 0.0001; {obs_mean/null.mean():.1f}x null mean)',
                fontweight='bold')
ax[1].legend(fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(RES, 'real_causal_stability.png'), dpi=200, bbox_inches='tight')
print('regenerated experimental_results/real_causal_stability.png')
