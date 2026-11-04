// Monte Carlo of the K-truncated local functional Psi_K in the Poisson(c) edge-rooted GW tree:
//  clause1_K = [ total progeny of the ancestor tree of i < K ] and [j has >= 1 other parent]
//  clause2_K = exists other parent chain k=u_0 <- u_1 <- ... <- u_r (u_r has no parents) with |out(u_r)| < K,
//              where out(u_r) = path nodes + j + all other-children trees of path nodes + out-tree of j.
// Also A_K = [|out(j)|>=K and |in(i)|>=K]. Output compared with the finite-p exact counts of x1check.
#include <bits/stdc++.h>
using namespace std;
mt19937_64 rng(777); poisson_distribution<int>* PD; int K;
inline int P(){ return (*PD)(rng); }
int gwsize(int cap){ // total progeny capped at cap
  long long total=1,cur=1; while(cur>0 && total<cap){ long long nxt=0; for(long long t=0;t<cur;t++) nxt+=P(); total+=nxt; cur=nxt; if(total>=cap) return cap; } return (int)min<long long>(total,cap); }
int othersize(int cap){ int n=P(); int s=0; for(int t=0;t<n;t++){ s+=gwsize(cap); if(s>=cap) return cap; } return s; }
bool rec(int used,int depth){ // current path node v: add 1 + other children sizes
  used+=1+othersize(K); if(used>=K) return false;
  int n=P(); if(n==0) return true; if(depth>K) return false;
  for(int t=0;t<n;t++) if(rec(used,depth+1)) return true; return false; }
int main(int argc,char**argv){
  double c=atof(argv[1]); K=atoi(argv[2]); long long n=atoll(argv[3]); PD=new poisson_distribution<int>(c);
  long long c1=0,c2=0,ps=0,A=0;
  for(long long s=0;s<n;s++){
    int inI=gwsize(K); bool inBig=(inI>=K);            // |in(i)| >= K
    int sj=1+othersize(K); bool outBig=(sj>=K);          // |out(j)| >= K
    int nk=P();
    bool cl1=(!inBig && nk>=1);
    bool cl2=false; if(!outBig){ for(int t=0;t<nk;t++){ if(rec(sj,0)){cl2=true;break;} } }
    A+=(inBig&&outBig); c1+=cl1; c2+=cl2; ps+=(cl1||cl2);
  }
  printf("c=%g K=%d n=%lld P(A_K)=%.5f P(clause1_K)=%.5f P(clause2_K)=%.5f P(Psi_K)=%.5f se=%.5f\n",c,K,n,(double)A/n,(double)c1/n,(double)c2/n,(double)ps/n,sqrt((double)ps/n*(1-(double)ps/n)/n));
}
