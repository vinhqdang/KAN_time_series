import pandas as pd, glob, numpy as np
from scipy.stats import wilcoxon
d = pd.concat([pd.read_csv(f) for f in glob.glob('results/cells/*.csv')] + [pd.read_csv('results/test_d10_all_with_basil.csv')])
d = d[~d.method.isin(['SPIRE_c2_m0', 'SPIRE_c2_m0.3', 'R2-sortnregress', 'BASIL_g0.05_p1'])].drop_duplicates(['d', 'n', 'noise', 'seed', 'method'])
d.to_csv('results/all_cells.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
lines = []
for (dd, n, nz), g in d.groupby(['d', 'n', 'noise']):
    ref = 'SPADE-adaptive'; seeds = g.pivot(index='seed', columns='method', values='auroc').dropna(subset=['BASIL_g1_p1', ref]).index
    t = g[g.seed.isin(seeds)].groupby('method')[['auroc', 'auprc', 'skel_auroc', 'orient', 'slope', 'cover', 'ece', 'time']].mean().round(3)
    lines.append(f'\n=== d={dd} n={n} noise={nz}  paired datasets={len(seeds)} ===\n' + t.to_string())
    for col in ['auroc', 'auprc', 'orient', 'slope']:
        p = g.pivot(index='seed', columns='method', values=col).loc[seeds]; df = (p['BASIL_g1_p1'] - p[ref]).dropna()
        if len(df) < 3: continue
        bs = np.random.RandomState(0).choice(df.values, (4000, len(df))).mean(1)
        pv = wilcoxon(df).pvalue if df.abs().sum() > 0 else float('nan')
        lines.append(f'  BASIL-minus-{ref} {col:6s} {df.mean():+.3f} CI[{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}] p={pv:.3g}')
open('results/all_cells_analysis.txt', 'w').write('\n'.join(lines)); print('\n'.join(lines))
