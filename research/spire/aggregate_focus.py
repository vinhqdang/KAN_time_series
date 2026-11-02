import pandas as pd, glob, numpy as np
d = pd.concat([pd.read_csv(f) for f in glob.glob('results/scale/F_d*_s*.csv')])
d.to_csv('results/focus_all.csv', index=False); pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
out = []
for dd, g in d.groupby('d'):
    t = g.groupby('method')[['auroc', 'auprc', 'orient', 'nrmse', 'slope', 'curve_r2', 'cover', 'ece', 'recall', 'time']].mean().round(3)
    out.append(f'\n=== d={dd} (datasets per method: {g.groupby("method").seed.nunique().to_dict()}) ===\n' + t.to_string())
    p = g.pivot(index='seed', columns='method', values='nrmse')
    for m in p.columns:
        if m in ('SSC',): continue
        r = (p[m] / p['SSC']).dropna(); df = (p[m] - p['SSC']).dropna()
        out.append(f'  NRMSE {m} / SSC = {r.mean():.2f} (per-dataset range {r.min():.2f}-{r.max():.2f})')
    if 'BASILW' in p:
        for m in [x for x in p.columns if x.startswith('SSR')]:
            df = (p['BASILW'] - p[m]).dropna(); out.append(f'  NRMSE BASILW minus {m} = {df.mean():+.3f} (range {df.min():+.3f}..{df.max():+.3f}, n={len(df)})')
    for col in ['auroc', 'orient']:
        pp = g.pivot(index='seed', columns='method', values=col)
        if 'BASILW' in pp: df = (pp['BASILW'] - pp['SSC']).dropna(); out.append(f'  {col} BASILW minus SSC = {df.mean():+.3f} (range {df.min():+.3f}..{df.max():+.3f})')
open('results/focus_analysis.txt', 'w').write('\n'.join(out)); print('\n'.join(out))
