// x1check.cpp: exact structural checks for D(p,c/p) (non-loop edges), used to test every intermediate claim of X1PROOF.md.
// Usage: x1check p c seed  -> prints key=value lines (one record).  Single thread.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll; typedef unsigned long long u64;
int p; double c; int m;
vector<int> ei, ej;               // edges
vector<int> fo, fa, bo, ba;       // CSR forward (offset, adjacency) and backward
vector<int> stampv; int stampc=0;
static int cappedBFS(int v, bool fwd, int cap, vector<int>* out=nullptr){
  const vector<int>&off = fwd?fo:bo; const vector<int>&adj = fwd?fa:ba;
  ++stampc; vector<int> q; q.push_back(v); stampv[v]=stampc; size_t h=0;
  while(h<q.size() && (int)q.size()<cap){ int u=q[h++]; for(int k=off[u];k<off[u+1];k++){int w=adj[off[u]+ (k-off[u])]; if(stampv[w]!=stampc){stampv[w]=stampc;q.push_back(w); if((int)q.size()>=cap) break;}}}
  int s=q.size(); if(out) *out=q; return s;
}
int main(int argc,char**argv){
  p=atoi(argv[1]); c=atof(argv[2]); u64 seed=atoll(argv[3]); bool light=(argc>4); int Ks[]={5,10,20,40};
  mt19937_64 rng(seed*1000003ULL+12345);
  ll npairs=(ll)p*(p-1); binomial_distribution<ll> bd(npairs,c/p); m=(int)bd(rng);
  vector<u64> keys; 
  while((int)keys.size()<m){ int need=(m-keys.size())*1.05+10; for(int t=0;t<need;t++){ u64 a=rng()%p,b=rng()%p; if(a!=b) keys.push_back(a*p+b);} sort(keys.begin(),keys.end()); keys.erase(unique(keys.begin(),keys.end()),keys.end()); }
  shuffle(keys.begin(),keys.end(),rng); keys.resize(m);
  ei.resize(m); ej.resize(m); for(int e=0;e<m;e++){ei[e]=keys[e]/p; ej[e]=keys[e]%p;}
  // CSR
  fo.assign(p+1,0); bo.assign(p+1,0); for(int e=0;e<m;e++){fo[ei[e]+1]++; bo[ej[e]+1]++;}
  for(int v=0;v<p;v++){fo[v+1]+=fo[v]; bo[v+1]+=bo[v];}
  fa.resize(m); ba.resize(m); {vector<int> pf(fo.begin(),fo.end()-1), pb(bo.begin(),bo.end()-1); for(int e=0;e<m;e++){fa[pf[ei[e]]++]=ej[e]; ba[pb[ej[e]]++]=ei[e];}}
  vector<int> indeg(p), outdeg(p); for(int v=0;v<p;v++){indeg[v]=bo[v+1]-bo[v]; outdeg[v]=fo[v+1]-fo[v];}
  stampv.assign(p,0);
  // iterative Tarjan
  vector<int> idx(p,-1), low(p,0), comp(p,-1), st, itp(p,0); vector<char> onst(p,0); int counter=0, nc=0;
  for(int r=0;r<p;r++) if(idx[r]<0){
    vector<int> cs; cs.push_back(r); idx[r]=low[r]=counter++; st.push_back(r); onst[r]=1; itp[r]=fo[r];
    while(!cs.empty()){
      int u=cs.back();
      if(itp[u]<fo[u+1]){ int w=fa[itp[u]++]; if(idx[w]<0){idx[w]=low[w]=counter++; st.push_back(w); onst[w]=1; itp[w]=fo[w]; cs.push_back(w);} else if(onst[w]) low[u]=min(low[u],idx[w]); }
      else { if(low[u]==idx[u]){ while(true){int w=st.back(); st.pop_back(); onst[w]=0; comp[w]=nc; if(w==u)break;} nc++; }
        cs.pop_back(); if(!cs.empty()){int pr=cs.back(); low[pr]=min(low[pr],low[u]);} }
    }
  }
  vector<int> csz(nc,0); for(int v=0;v<p;v++) csz[comp[v]]++;
  int giant=0; for(int k=0;k<nc;k++) if(csz[k]>csz[giant]) giant=k;
  int second=0; for(int k=0;k<nc;k++) if(k!=giant) second=max(second,csz[k]);
  ll N=0; for(int e=0;e<m;e++) if(comp[ei[e]]==comp[ej[e]]) N++;
  ll ES=p+(ll)m; double L=0.5+(double)(m-N)/(2.0*ES);
  printf("p=%d c=%g seed=%llu m=%d nc=%d giant=%d second=%d N=%lld L=%.6f\n",p,c,(unsigned long long)seed,m,nc,csz[giant],second,N,L);
  if(argc>5) return 0;
  if(!light){
  // source SCCs and Src bitsets
  vector<int> crossIn(nc,0); for(int e=0;e<m;e++) if(comp[ei[e]]!=comp[ej[e]]) crossIn[comp[ej[e]]]++;
  vector<int> sid(nc,-1); int k=0; for(int u=0;u<nc;u++) if(crossIn[u]==0) sid[u]=k++;
  int W=(k+63)/64; vector<u64> bits((size_t)nc*W,0);
  for(int u=0;u<nc;u++) if(sid[u]>=0) bits[(size_t)u*W+sid[u]/64] |= 1ULL<<(sid[u]%64);
  // nodes by comp
  vector<int> cstart(nc+1,0); for(int v=0;v<p;v++) cstart[comp[v]+1]++; for(int u=0;u<nc;u++) cstart[u+1]+=cstart[u];
  vector<int> cn(p), cp(cstart.begin(),cstart.end()-1); for(int v=0;v<p;v++) cn[cp[comp[v]]++]=v;
  for(int u=nc-1;u>=0;u--){ for(int t=cstart[u];t<cstart[u+1];t++){int v=cn[t]; for(int q=fo[v];q<fo[v+1];q++){int w=fa[q]; int cw=comp[w]; if(cw!=u){ u64*bw=&bits[(size_t)cw*W]; u64*bu=&bits[(size_t)u*W]; for(int z=0;z<W;z++) bw[z]|=bu[z]; }}}}
  vector<int> cnt(nc); for(int u=0;u<nc;u++){int s=0; for(int z=0;z<W;z++) s+=__builtin_popcountll(bits[(size_t)u*W+z]); cnt[u]=s;}
  vector<char> up(m,0); ll nup=0, viol_conf=0;
  for(int e=0;e<m;e++){ if(cnt[comp[ei[e]]]!=cnt[comp[ej[e]]]){up[e]=1;nup++; if(indeg[ej[e]]<2) viol_conf++;} }
  double S=0.5+(double)nup/(2.0*ES);
  ll nsrcnodes=0; for(int v=0;v<p;v++) if(indeg[v]==0) nsrcnodes++;
  int ntsrc=0; for(int u=0;u<nc;u++) if(sid[u]>=0 && csz[u]>1) ntsrc++;
  ll ntsrcnodes=0; for(int u=0;u<nc;u++) if(sid[u]>=0 && csz[u]>1) ntsrcnodes+=csz[u];
  printf("k_src=%d nontrivial_src_scc=%d nodes_in_nontrivial_src=%lld nup=%lld S=%.6f up_not_confluence=%lld\n",k,ntsrc,ntsrcnodes,nup,S,viol_conf);
  // big-source bitset (out >= 200 capped) 
  const int BIGCAP=200; vector<u64> bigsrc(W,0);
  for(int u=0;u<nc;u++) if(sid[u]>=0){ int s=cappedBFS(cn[cstart[u]],true,BIGCAP); if(s>=BIGCAP) bigsrc[sid[u]/64]|=1ULL<<(sid[u]%64); }
  // per K tests
  for(int K: Ks){
    vector<int> incap(p), outcap(p);
    for(int v=0;v<p;v++){ incap[v]=cappedBFS(v,false,K); outcap[v]=cappedBFS(v,true,K); }
    ll Nt=0; for(int e=0;e<m;e++) if(outcap[ej[e]]>=K && incap[ei[e]]>=K) Nt++;
    ll za=0, zb=0; for(int e=0;e<m;e++){ bool Z=comp[ei[e]]==comp[ej[e]]; bool A=(outcap[ej[e]]>=K && incap[ei[e]]>=K); if(Z&&!A) za++; if(!Z&&A) zb++; }
    // Psi
    vector<char> psi1(m,0), psi2(m,0);
    for(int e=0;e<m;e++) if(incap[ei[e]]<K && indeg[ej[e]]>=2) psi1[e]=1;
    vector<int> O; vector<int> mark(p,-1);
    // need edge ids: build id lookup via backward CSR with edge ids
    // (recompute with explicit ids)
    vector<int> bid(m); { vector<int> pb(bo.begin(),bo.end()-1); for(int e=0;e<m;e++) bid[pb[ej[e]]++]=e; }
    fill(mark.begin(),mark.end(),-1);
    for(int s=0;s<p;s++) if(indeg[s]==0){ int sz=cappedBFS(s,true,K,&O); if(sz<K){ for(int v:O) mark[v]=s; for(int j:O) for(int q=bo[j];q<bo[j+1];q++){ int i=ba[q]; if(mark[i]!=s) psi2[bid[q]]=1; } } }
    ll n1=0,n2=0,np=0,upn=0,m10=0,m01=0, m10_bigin=0, m10_bigwit=0, up_bigin=0, up_bigin_bigwit=0, p1_notup=0, p2_notup=0;
    for(int e=0;e<m;e++){ bool P=psi1[e]||psi2[e]; n1+=psi1[e]; n2+=psi2[e]; np+=P; upn+=up[e];
      if(up[e]&&!P){m10++; if(incap[ei[e]]>=K) m10_bigin++; }
      if(!up[e]&&P) m01++; if(psi1[e]&&!up[e]) p1_notup++; if(psi2[e]&&!up[e]) p2_notup++;
      if(up[e] && incap[ei[e]]>=K){ up_bigin++; u64*bj=&bits[(size_t)comp[ej[e]]*W]; u64*bi=&bits[(size_t)comp[ei[e]]*W]; bool bw=false; for(int z=0;z<W;z++) if(bj[z]&~bi[z]&bigsrc[z]){bw=true;break;} if(bw) up_bigin_bigwit++; }
    }
    ll cnt_inbig=0, cnt_outbig=0; for(int v=0;v<p;v++){cnt_inbig+= incap[v]>=K; cnt_outbig+= outcap[v]>=K;}
    printf("K=%d Ntilde=%lld Z1A0=%lld Z0A1=%lld frac_in_big=%.5f frac_out_big=%.5f | Psi=%lld Psi1=%lld Psi2=%lld up=%lld up_not_Psi=%lld Psi_not_up=%lld Psi1_not_up=%lld Psi2_not_up=%lld up_bigin=%lld up_bigin_with_bigwitness=%lld\n",
      K,Nt,za,zb,(double)cnt_inbig/p,(double)cnt_outbig/p,np,n1,n2,upn,m10,m01,p1_notup,p2_notup,up_bigin,up_bigin_bigwit);
  }
  } // end !light
  if(light){ int Ks2[]={10,20,40}; for(int K:Ks2){ vector<int> incap(p),outcap(p); for(int v=0;v<p;v++){incap[v]=cappedBFS(v,false,K);outcap[v]=cappedBFS(v,true,K);} ll Nt=0,za=0,zb=0; for(int e=0;e<m;e++){bool Z=comp[ei[e]]==comp[ej[e]]; bool A=(outcap[ej[e]]>=K&&incap[ei[e]]>=K); Nt+=A; if(Z&&!A)za++; if(!Z&&A)zb++;} printf("K=%d Ntilde=%lld Z1A0=%lld Z0A1=%lld\n",K,Nt,za,zb);} }
  // structure: exact |out(v)| for v not reaching giant, |in(v)| for v not reachable from giant
  vector<char> reachG(p,0), fromG(p,0);
  { vector<int> q; for(int v=0;v<p;v++) if(comp[v]==giant){q.push_back(v); reachG[v]=1;} for(size_t h=0;h<q.size();h++){int u=q[h]; for(int t=bo[u];t<bo[u+1];t++){int w=ba[t]; if(!reachG[w]){reachG[w]=1;q.push_back(w);}}} }
  { vector<int> q; for(int v=0;v<p;v++) if(comp[v]==giant){q.push_back(v); fromG[v]=1;} for(size_t h=0;h<q.size();h++){int u=q[h]; for(int t=fo[u];t<fo[u+1];t++){int w=fa[t]; if(!fromG[w]){fromG[w]=1;q.push_back(w);}}} }
  int maxSmallOut=0,maxSmallIn=0; ll nOutBig=0,nInBig=0,nBoth=0;
  for(int v=0;v<p;v++){ nOutBig+=reachG[v]; nInBig+=fromG[v]; nBoth+=(reachG[v]&&fromG[v]);
    if(!reachG[v]){ int s=cappedBFS(v,true,p+1); maxSmallOut=max(maxSmallOut,s);} 
    if(!fromG[v]){ int s=cappedBFS(v,false,p+1); maxSmallIn=max(maxSmallIn,s);} }
  printf("reach_giant=%.5f from_giant=%.5f both=%.5f giantfrac=%.5f maxSmallOut=%d maxSmallIn=%d logp=%.2f\n",(double)nOutBig/p,(double)nInBig/p,(double)nBoth/p,(double)csz[giant]/p,maxSmallOut,maxSmallIn,log((double)p));
  // dichotomy by sampling (exact sizes)
  int ns=min(p,3000); vector<int> so(ns), si(ns); for(int t=0;t<ns;t++){ int v=rng()%p; so[t]=cappedBFS(v,true,p+1); si[t]=cappedBFS(v,false,p+1); }
  for(int K: Ks){ double eps=0.02; int a=0,b=0,a2=0,b2=0; for(int t=0;t<ns;t++){ if(so[t]>=K && so[t]<eps*p) a++; if(si[t]>=K && si[t]<eps*p) b++; if(so[t]>=K) a2++; if(si[t]>=K) b2++; }
    printf("dichot K=%d eps=0.02 P_meso_out=%.5f P_meso_in=%.5f P_out_ge_K=%.5f P_in_ge_K=%.5f (n=%d)\n",K,(double)a/ns,(double)b/ns,(double)a2/ns,(double)b2/ns,ns); }
  return 0;
}
