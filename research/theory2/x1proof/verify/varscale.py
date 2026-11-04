import re, math, collections, sys
def x_surv(c):
    x=0.9
    for _ in range(20000): x=1-math.exp(-c*x)
    return x
def load(fn):
    rs=[]
    for l in open(fn):
        if l.startswith('p='): rs.append({a:float(b) for a,b in re.findall(r'(\w+)=([-\d.e+]+)',l)})
    return rs
for fn,c in [('results/x1check_varscale_c2.txt',2.0),('results/x1check_vl_c1.5_p100000.txt',1.5)]:
    x=x_surv(c); g=collections.defaultdict(list)
    for r in load(fn): g[int(r['p'])].append(r)
    print(f"c={c}: x^2={x*x:.5f}, c x^2={c*x*x:.5f}, L_pred={0.5+c*(1-x*x)/(2*(1+c)):.5f}")
    for p,rs in sorted(g.items()):
        n=len(rs); Np=[r['N']/p for r in rs]; Lv=[r['L'] for r in rs]; Gp=[r['giant']/p for r in rs]
        mu=sum(Np)/n; sd=math.sqrt(sum((a-mu)**2 for a in Np)/(n-1))
        lm=sum(Lv)/n; lsd=math.sqrt(sum((a-lm)**2 for a in Lv)/(n-1)); gm=sum(Gp)/n
        print(f"  p={p} n={n}: mean N/p={mu:.5f} (se {sd/math.sqrt(n):.5f}), sd(N/p)={sd:.5f}, sd*sqrt(p)={sd*math.sqrt(p):.3f}; mean L={lm:.5f} (se {lsd/math.sqrt(n):.5f}); giant/p={gm:.5f}")
