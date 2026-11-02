import pandas as pd, glob, numpy as np
d = pd.concat([pd.read_csv(f) for f in glob.glob('results/do/*.csv')]); d['cell'] = d.d.astype(str) + '|' + d.noise + '|' + d.mech + '|n' + d.n.astype(str)
d.to_csv('results/do_all.csv', index=False); pd.set_option('display.width', 260); pd.set_option('display.max_columns', 30)
out = []
for cell, g in d.groupby('cell', sort=False):
    t = g.groupby('method')[['nrmse', 'do_direct', 'do_indirect', 'slope', 'auroc', 'auprc', 'orient', 'cover', 'ece', 'time']].mean().round(3)
    k = g.groupby('method').seed.nunique().to_dict()
    out.append(f'\n=== {cell} datasets={k} ===\n' + t.to_string())
    p = g.pivot(index='seed', columns='method', values='nrmse')
    if 'SSC' in p and 'BASILW' in p:
        out.append('  edge NRMSE ratio BASILW/SSC = %.2f ; BASILW-SSR = %+.3f' % ((p['BASILW'] / p['SSC']).mean(), (p['BASILW'] - p['SSR_k1.5']).mean()))
    for col in ['do_direct', 'do_indirect']:
        q = g.pivot(index='seed', columns='method', values=col)
        if 'SSC' in q and 'BASILW' in q: out.append('  %s ratio BASILW/SSC = %.2f (range %.2f-%.2f) ; BASILW-SSR = %+.3f' % (col, (q['BASILW'] / q['SSC']).mean(), (q['BASILW'] / q['SSC']).min(), (q['BASILW'] / q['SSC']).max(), (q['BASILW'] - q['SSR_k1.5']).mean()))
    a = g.pivot(index='seed', columns='method', values='auroc')
    if 'SSC' in a and 'BASILW' in a: out.append('  AUROC BASILW-SSC = %+.3f' % (a['BASILW'] - a['SSC']).mean())
open('results/do_analysis.txt', 'w').write('\n'.join(out)); print('\n'.join(out))
