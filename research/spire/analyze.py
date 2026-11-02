import pandas as pd, glob, sys, numpy as np
from scipy.stats import wilcoxon
pat = sys.argv[1]; ref = sys.argv[2]
d = pd.concat([pd.read_csv(f) for f in glob.glob(pat)]); pd.set_option('display.width', 220)
print(d.groupby('method')[['auroc','auprc','skel_auroc','orient','curve_r2','slope','nnz','time']].mean().round(3))
print('datasets per method:', d.groupby('method').seed.nunique().to_dict())
for col in ['auroc', 'orient', 'slope']:
    p = d.pivot(index='seed', columns='method', values=col)
    print('\npaired vs', ref, col)
    for m in p.columns:
        if m == ref: continue
        df = (p[m] - p[ref]).dropna()
        if len(df) < 5 or df.abs().sum() == 0: continue
        bs = np.random.RandomState(0).choice(df.values, (5000, len(df))).mean(1)
        print('  %-20s mean diff %+.3f  CI[%+.3f,%+.3f]  wilcoxon p=%.3g  n=%d' % (m, df.mean(), *np.percentile(bs, [2.5, 97.5]), wilcoxon(df).pvalue, len(df)))
