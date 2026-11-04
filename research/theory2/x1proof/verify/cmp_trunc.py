import re,collections,math
def kv(l): return {a:float(b) for a,b in re.findall(r'(P\(\w+\)|\w+)=([-\d.e+]+)',l)}
mc={}
for l in open('results/gw_trunc.txt'):
    d=kv(l); mc[(d['c'],int(d['K']))]=d
acc=collections.defaultdict(lambda: collections.defaultdict(float))
for fn in ['results/x1check_p100000.txt','results/x1check_p20000.txt']:
    cur=None
    for l in open(fn):
        if l.startswith('p='): cur=kv(l); p=int(cur['p']); c=cur['c']
        elif l.startswith('K='):
            d=kv(l); K=int(d['K']); a=acc[(c,K,p)]; a['m']+=cur['m']; 
            for k in ['Ntilde','Psi','Psi1','Psi2','up']: a[k]+=d[k]
print("finite-p exact fractions (per non-loop edge) vs Monte Carlo of the GW local limit (1e6 samples, se ~0.0005):")
for (c,K,p),a in sorted(acc.items()):
    if (c,K) not in mc: continue
    M=mc[(c,K)]
    print(f"c={c} K={K} p={p}: A_K {a['Ntilde']/a['m']:.4f} vs {M['P(A_K)']:.4f}; clause1 {a['Psi1']/a['m']:.4f} vs {M['P(clause1_K)']:.4f}; clause2 {a['Psi2']/a['m']:.4f} vs {M['P(clause2_K)']:.4f}; Psi_K {a['Psi']/a['m']:.4f} vs {M['P(Psi_K)']:.4f}; E_up {a['up']/a['m']:.4f}")
