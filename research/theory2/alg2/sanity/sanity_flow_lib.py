# Tiny sanity computation (not an experiment): synthetic conserved-flow in-tree with a multi-filter latent driver.
import numpy as np, time, sys
from scipy.optimize import linprog, nnls
from scipy.sparse import hstack, eye, csr_matrix
from sklearn.metrics import roc_auc_score
def gen(seed, p=8, T=1500, pi_loc=0.01, L=2, meas=0.02):
    r=np.random.default_rng(seed)
    child=np.full(p,-1)
    for i in range(p-1): child[i]=r.integers(i+1,p)      # in-forest, node p-1 is the outlet
    R=(r.random(T+200)<0.08)*r.gamma(2.0,1.0,T+200)       # regional rain pulses
    X=np.zeros((p,T+200)); K={}
    for j in range(p):
        tau=r.uniform(2,12); h=np.exp(-np.arange(40)/tau); h/=h.sum()
        d=r.integers(0,3)                                  # node-specific delay of the driver
        common=np.convolve(np.roll(R,d),h)[:T+200]
        size=r.lognormal(0,0.7)
        loc=(r.random(T+200)<pi_loc)*r.gamma(2.0,2.0,T+200)
        hl=np.exp(-np.arange(20)/r.uniform(1,4)); hl/=hl.sum()
        U=size*(0.3+common+np.convolve(loc,hl)[:T+200])
        x=U.copy()
        for i in np.where(child==j)[0]:
            k=r.dirichlet(np.ones(L+1)*0.7); k[0]*=0.3; k/=k.sum(); K[(i,j)]=k
            for l in range(L+1): x[l:]+=k[l]*X[i,:T+200-l]
        X[j]=x
    X=X[:,200:]*(1+meas*r.standard_normal((p,T)))
    G=np.zeros((p,p),int)
    for i in range(p-1): G[i,child[i]]=1
    return X,G
def design(X,j,L):
    p,T=X.shape; cols=[];idx=[]
    for i in range(p):
        if i==j: continue
        for l in range(L+1):
            cols.append(X[i,L-l:T-l]); idx.append(i)
    return np.array(cols).T, X[j,L:], np.array(idx)
def cf_lp(X,L=2,tau=0.02):
    p=X.shape[0]; S=np.zeros((p,p)); mu=X.mean(1)
    for j in range(p):
        Z,y,idx=design(X,j,L); n,m=Z.shape
        # min tau*sum(u)+(1-tau)*sum(v)  s.t. Z k + u - v = y, k,u,v>=0, sum_l k_il<=1
        c=np.r_[np.zeros(m),tau*np.ones(n),(1-tau)*np.ones(n)]
        A=hstack([csr_matrix(Z),eye(n),-eye(n)]).tocsr()
        Aub=np.zeros((p-1,m+2*n)); 
        for a,i in enumerate([i for i in range(p) if i!=j]): Aub[a,:m]=(idx==i)
        res=linprog(c,A_ub=csr_matrix(Aub),b_ub=np.ones(p-1),A_eq=A,b_eq=y,bounds=(0,None),method="highs")
        k=res.x[:m]
        for i in range(p):
            if i!=j: S[i,j]=k[idx==i].sum()*mu[i]/mu[j]
    return S
def nn_ls(X,L=2):
    p=X.shape[0]; S=np.zeros((p,p)); mu=X.mean(1)
    for j in range(p):
        Z,y,idx=design(X,j,L); k,_=nnls(np.c_[Z,np.ones(len(y))],y); k=k[:-1]
        for i in range(p):
            if i!=j: S[i,j]=k[idx==i].sum()*mu[i]/mu[j]
    return S
def var_ols(X,L=2):
    Xs=(X-X.mean(1,keepdims=True))/X.std(1,keepdims=True); p,T=X.shape; S=np.zeros((p,p))
    Z=np.array([Xs[i,L-l:T-l] for i in range(p) for l in range(1,L+1)]).T
    for j in range(p):
        b=np.linalg.lstsq(Z,Xs[j,L:],rcond=None)[0].reshape(p,L)
        S[:,j]=np.abs(b).max(1)
    return S
def corr(X,L=2):
    Xs=(X-X.mean(1,keepdims=True))/X.std(1,keepdims=True); p,T=X.shape; S=np.zeros((p,p))
    for i in range(p):
        for j in range(p):
            S[i,j]=max(abs(np.mean(Xs[i,:T-l]*Xs[j,l:])) for l in range(L+1))
    return S
def auc(S,G):
    m=~np.eye(len(G),dtype=bool); return roc_auc_score(G[m],S[m])
