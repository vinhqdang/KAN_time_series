import numpy as np, sanity_flow_lib as s
for (pi,meas,tau) in ((0.03,0.0,0.001),(0.01,0.0,0.001),(0.01,0.02,0.001),(0.01,0.02,0.1)):
    o={"cf_lp":[],"corr+mean-order":[]}
    for seed in range(8):
        X,G=s.gen(seed,pi_loc=pi,meas=meas)
        o["cf_lp"].append(s.auc(s.cf_lp(X,tau=tau),G)); mu=X.mean(1)
        o["corr+mean-order"].append(s.auc(s.corr(X)*(mu[:,None]<mu[None,:]),G))
    print("pi",pi,"meas",meas,"tau",tau,{k:round(float(np.mean(v)),3) for k,v in o.items()})
