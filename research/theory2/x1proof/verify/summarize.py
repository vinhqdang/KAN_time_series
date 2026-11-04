"""Summarise x1check output files: per (p,c) means over seeds against the limits of X1PROOF.md."""
import sys, math, re, collections
def x_surv(c):
    if c<=1: return 0.0
    x=0.9
    for _ in range(20000): x=1-math.exp(-c*x)
    return x
def d_fix(c,x):
    d=0.0
    for _ in range(200000):
        nd=1+math.exp(-c)-math.exp(-c*(1-x)*d)
        if abs(nd-d)<1e-16: break
        d=nd
    return d
def theta(c,K):  # P(total progeny of Poisson(c) GW >= K)
    s=0.0
    for n in range(1,K):
        s+=math.exp(-c*n+(n-1)*math.log(c*n)-math.lgamma(n+1))
    return 1-s
def kv(line): return {a:b for a,b in re.findall(r'(\w+)=([-\d.e+]+)',line)}
recs=[]; cur=None
for fn in sys.argv[1:]:
    for line in open(fn):
        if line.startswith('p='):
            cur=kv(line); cur['K']={}; cur['dich']={}; recs.append(cur)
        elif line.startswith('k_src'): cur.update(kv(line))
        elif line.startswith('K='):
            d=kv(line); cur['K'][int(d['K'])]=d
        elif line.startswith('reach_giant'): cur.update(kv(line))
        elif line.startswith('dichot'):
            d=kv(line); cur['dich'][int(d['K'])]=d
groups=collections.defaultdict(list)
for r in recs: groups[(int(r['p']),float(r['c']))].append(r)
def mean(v): return sum(v)/len(v)
def sd(v): return math.sqrt(sum((a-mean(v))**2 for a in v)/(len(v)-1)) if len(v)>1 else float('nan')
print("== global quantities (mean over seeds [sd]) vs limits")
for (p,c),rs in sorted(groups.items(), key=lambda t:(t[0][1],t[0][0])):
    x=x_surv(c); d=d_fix(c,x); m=[float(r['m']) for r in rs]
    Fx=1-x*x; Fup=(1-x)*(1-math.exp(-c))+x*(1-x)*(1-math.exp(-c*(1-x)*d))
    Lp=0.5+c*Fx/(2*(1+c)); Sp=0.5+c*Fup/(2*(1+c))
    L=[float(r['L']) for r in rs]; S=[float(r['S']) for r in rs] if 'S' in rs[0] else []
    gi=[float(r['giant'])/p for r in rs]; N=[float(r['N']) for r in rs]; sec=[float(r['second']) for r in rs]
    print(f"p={p} c={c} n={len(rs)}: giant/p {mean(gi):.4f} (x^2={x*x:.4f}); second SCC {mean(sec):.1f} max {max(sec):.0f} (log p={math.log(p):.1f}); N/p {mean(N)/p:.4f} (c x^2={c*x*x:.4f}); L {mean(L):.5f}[{sd(L):.5f}] pred {Lp:.5f}", end='')
    if S:
        nup=[float(r['nup']) for r in rs]
        print(f"; S {mean(S):.5f}[{sd(S):.5f}] pred {Sp:.5f}; nup/m {mean([a/b for a,b in zip(nup,m)]):.4f} (F_up={Fup:.4f}); up_not_confluence total {sum(float(r['up_not_confluence']) for r in rs):.0f}; nontrivial source SCCs {mean([float(r['nontrivial_src_scc']) for r in rs]):.2f} (nodes {mean([float(r['nodes_in_nontrivial_src']) for r in rs]):.2f})")
    else: print()
print("\n== structure: maxSmallOut/In (largest out/in-set of a node that does not reach / is not reached from the giant), reach/from giant fraction vs x")
for (p,c),rs in sorted(groups.items(), key=lambda t:(t[0][1],t[0][0])):
    if 'maxSmallOut' not in rs[0]: continue
    x=x_surv(c)
    print(f"p={p} c={c}: maxSmallOut {max(float(r['maxSmallOut']) for r in rs):.0f}, maxSmallIn {max(float(r['maxSmallIn']) for r in rs):.0f}, log p {math.log(p):.1f}; reach_giant {mean([float(r['reach_giant']) for r in rs]):.4f}, from_giant {mean([float(r['from_giant']) for r in rs]):.4f}, x={x:.4f}; both = giant {mean([float(r['both']) for r in rs]):.4f} (giantfrac {mean([float(r['giantfrac']) for r in rs]) if 'giantfrac' in rs[0] else float('nan'):.4f}) vs x^2 {x*x:.4f}")
print("\n== local functional Ntilde_K = sum_e 1[out(j)>=K, in(i)>=K] vs N, and Psi_K vs E_up (edge by edge); sums over seeds")
for (p,c),rs in sorted(groups.items(), key=lambda t:(t[0][1],t[0][0])):
    x=x_surv(c); d=d_fix(c,x)
    for K in sorted(rs[0]['K']):
        ks=[r['K'][K] for r in rs]; m=sum(float(r['m']) for r in rs)
        row=f"p={p} c={c} K={K}: (Ntilde-N)/m {sum(float(k['Ntilde']) for k in ks)/m - sum(float(r['N']) for r in rs)/m:+.5f}, Z1A0/m {sum(float(k['Z1A0']) for k in ks)/m:.5f}, Z0A1/m {sum(float(k['Z0A1']) for k in ks)/m:.5f}"
        row+=f"; frac out>=K {mean([float(k['frac_out_big']) if 'frac_out_big' in k else float('nan') for k in ks]):.4f} theta_K {theta(c,K):.4f}"
        if 'Psi' in ks[0]:
            row+=f"; up_not_Psi/m {sum(float(k['up_not_Psi']) for k in ks)/m:.5f}, Psi_not_up/m {sum(float(k['Psi_not_up']) for k in ks)/m:.5f}, Psi1_not_up {sum(float(k['Psi1_not_up']) for k in ks):.0f}, Psi2_not_up {sum(float(k['Psi2_not_up']) for k in ks):.0f}, up_bigin_with_bigwitness {sum(float(k['up_bigin_with_bigwitness']) for k in ks):.0f}/{sum(float(k['up_bigin']) for k in ks):.0f}"
        print(row)
print("\n== dichotomy (3000 random nodes per seed, exact sizes, eps=0.02): P(K<=|out|<eps p) and P(K<=|in|<eps p) vs K")
for (p,c),rs in sorted(groups.items(), key=lambda t:(t[0][1],t[0][0])):
    if not rs[0]['dich']: continue
    print(f"p={p} c={c}: "+"; ".join(f"K={K}: meso out {mean([float(r['dich'][K]['P_meso_out']) for r in rs]):.4f} in {mean([float(r['dich'][K]['P_meso_in']) for r in rs]):.4f} (P(out>=K) {mean([float(r['dich'][K]['P_out_ge_K']) for r in rs]):.4f} vs theta_K {theta(c,K):.4f})" for K in sorted(rs[0]['dich'])))
