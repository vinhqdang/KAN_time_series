# sanity: Gaussian second-order identifiability of a k=2 subsampled VAR(1) with diagonal D
import numpy as np, itertools
r=np.random.default_rng(0); p=4; cnt=[]
for rep in range(200):
    V=r.standard_normal((p,p)); lam=r.uniform(0.2,0.9,p)*r.choice([-1,1],p)
    M=V@np.diag(lam)@np.linalg.inv(V); D=np.diag(r.uniform(0.5,2,p))
    P=M@M; Om=D+M@D@M.T
    w,U=np.linalg.eig(P); w=w.real; U=U.real; ok=0
    for s in itertools.product([-1,1],repeat=p):
        Mp=U@np.diag(np.array(s)*np.sqrt(w))@np.linalg.inv(U)
        # solve diag d: d_i e_i e_i^T + d_i m_i m_i^T = Om (least squares over symmetric entries)
        A=np.array([(np.outer(np.eye(p)[i],np.eye(p)[i])+np.outer(Mp[:,i],Mp[:,i]))[np.triu_indices(p)] for i in range(p)]).T
        d,res,_,_=np.linalg.lstsq(A,Om[np.triu_indices(p)],rcond=None)
        if np.linalg.norm(A@d-Om[np.triu_indices(p)])<1e-8 and (d>0).all(): ok+=1
    cnt.append(ok)
print("p",p,"number of observationally equivalent roots per draw (200 draws):",np.unique(cnt,return_counts=True))
